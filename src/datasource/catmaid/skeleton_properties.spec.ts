/**
 * @license
 * Copyright 2026 Google Inc.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

import { describe, expect, it, vi } from "vitest";

import type { CatmaidSkeletonPropertyClient } from "#src/datasource/catmaid/skeleton_properties.js";
import { CatmaidSkeletonPropertyProvider } from "#src/datasource/catmaid/skeleton_properties.js";
import type { SegmentPropertyMap } from "#src/segmentation_display_state/property_map.js";

function makeClient(
  skeletonIds: number[],
  neuronNames: Map<number, string>,
  annotationNames: Map<number, string[]>,
): CatmaidSkeletonPropertyClient {
  return {
    listSkeletons: vi.fn().mockResolvedValue(skeletonIds),
    fetchNeuronNames: vi.fn().mockResolvedValue(neuronNames),
    fetchSkeletonAnnotations: vi.fn().mockResolvedValue(annotationNames),
  };
}

function nextPublishedMap(provider: CatmaidSkeletonPropertyProvider) {
  return new Promise<SegmentPropertyMap | undefined>((resolve) => {
    const removeListener = provider.segmentPropertyMap.changed.add(() => {
      removeListener();
      resolve(provider.segmentPropertyMap.value);
    });
  });
}

function getIds(map: SegmentPropertyMap | undefined) {
  return [...map!.inlineProperties!.ids].map(Number);
}

function getColumn(map: SegmentPropertyMap | undefined, id: string) {
  const property = map!.inlineProperties!.properties.find(
    (property) => property.id === id,
  );
  if (property === undefined) {
    throw new Error(`Expected column ${id}.`);
  }
  return property;
}

describe("CatmaidSkeletonPropertyProvider", () => {
  it("lists every skeleton before names and annotations have loaded", async () => {
    const provider = new CatmaidSkeletonPropertyProvider(
      makeClient(
        [2, 1],
        new Map([
          [1, "neuron one"],
          [2, "neuron two"],
        ]),
        new Map(),
      ),
    );

    await provider.load();

    const map = provider.segmentPropertyMap.value;
    expect(getIds(map)).toEqual([1, 2]);
    expect(getColumn(map, "neuron_name").values).toEqual(["", ""]);
  });

  it("labels each skeleton with its neuron name and its sorted annotations", async () => {
    const provider = new CatmaidSkeletonPropertyProvider(
      makeClient(
        [2, 1],
        new Map([
          [1, "neuron one"],
          [2, "neuron two"],
        ]),
        new Map([[1, ["right", "left"]]]),
      ),
    );

    await provider.load();
    const map = await nextPublishedMap(provider);

    expect(getIds(map)).toEqual([1, 2]);
    expect(getColumn(map, "neuron_name")).toEqual({
      id: "neuron_name",
      type: "label",
      values: ["neuron one", "neuron two"],
    });
    expect(getColumn(map, "annotations")).toEqual({
      id: "annotations",
      type: "string",
      values: ["left right", ""],
    });
  });

  it("fails to load when the skeletons cannot be listed", async () => {
    const client = makeClient([], new Map(), new Map());
    client.listSkeletons = vi.fn().mockRejectedValue(new Error("offline"));
    const provider = new CatmaidSkeletonPropertyProvider(client);

    await expect(provider.load()).rejects.toThrow("offline");
    expect(provider.segmentPropertyMap.value).toBeUndefined();
  });

  it("drops skeletons the server no longer knows and updates the rest on refresh", async () => {
    const client = makeClient(
      [1, 2],
      new Map([
        [1, "neuron one"],
        [2, "neuron two"],
      ]),
      new Map(),
    );
    const provider = new CatmaidSkeletonPropertyProvider(client);
    await provider.load();
    await nextPublishedMap(provider);
    client.fetchNeuronNames = vi
      .fn()
      .mockResolvedValue(new Map([[1, "renamed"]]));
    client.fetchSkeletonAnnotations = vi
      .fn()
      .mockResolvedValue(new Map([[1, ["0000002", "0000001"]]]));

    await provider.refreshSegmentProperties([1, 2]);

    const map = provider.segmentPropertyMap.value;
    expect(getIds(map)).toEqual([1]);
    expect(getColumn(map, "neuron_name").values).toEqual(["renamed"]);
    expect(getColumn(map, "annotations").values).toEqual(["0000001 0000002"]);
  });

  it("keeps previous annotations when only the annotation fetch fails on refresh", async () => {
    const warnSpy = vi.spyOn(console, "warn").mockImplementation(() => {});
    const client = makeClient(
      [1],
      new Map([[1, "neuron one"]]),
      new Map([[1, ["0000001"]]]),
    );
    const provider = new CatmaidSkeletonPropertyProvider(client);
    await provider.load();
    await nextPublishedMap(provider);
    client.fetchNeuronNames = vi
      .fn()
      .mockResolvedValue(new Map([[1, "renamed"]]));
    client.fetchSkeletonAnnotations = vi
      .fn()
      .mockRejectedValue(new Error("timeout"));

    await provider.refreshSegmentProperties([1]);

    const map = provider.segmentPropertyMap.value;
    expect(getColumn(map, "neuron_name").values).toEqual(["renamed"]);
    expect(getColumn(map, "annotations").values).toEqual(["0000001"]);
    expect(warnSpy).toHaveBeenCalledTimes(1);
    warnSpy.mockRestore();
  });

  it("keeps refreshed values when an older load finishes later", async () => {
    const loadNames = Promise.withResolvers<Map<number, string>>();
    const client = makeClient([2, 1], new Map(), new Map());
    client.fetchNeuronNames = vi.fn((skeletonIds: readonly number[]) =>
      skeletonIds.length === 2
        ? loadNames.promise
        : Promise.resolve(new Map([[1, "renamed"]])),
    );
    const provider = new CatmaidSkeletonPropertyProvider(client);

    await provider.load();
    await provider.refreshSegmentProperties([1]);
    const loaded = nextPublishedMap(provider);
    loadNames.resolve(
      new Map([
        [1, "stale"],
        [2, "neuron two"],
      ]),
    );
    const map = await loaded;

    expect(getColumn(map, "neuron_name").values).toEqual([
      "renamed",
      "neuron two",
    ]);
  });

  it("keeps the skeleton list when the names fetch fails", async () => {
    const warnSpy = vi.spyOn(console, "warn").mockImplementation(() => {});
    const client = makeClient([1, 2], new Map(), new Map([[1, ["left"]]]));
    client.fetchNeuronNames = vi.fn().mockRejectedValue(new Error("offline"));
    const provider = new CatmaidSkeletonPropertyProvider(client);

    await provider.load();
    const map = await nextPublishedMap(provider);

    expect(warnSpy).toHaveBeenCalledTimes(1);
    expect(getIds(map)).toEqual([1, 2]);
    expect(getColumn(map, "neuron_name").values).toEqual(["", ""]);
    expect(getColumn(map, "annotations").values).toEqual(["left", ""]);
    warnSpy.mockRestore();
  });
});
