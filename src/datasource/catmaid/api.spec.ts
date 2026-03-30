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

import { CatmaidClient } from "#src/datasource/catmaid/api.js";

describe("CatmaidClient skeleton editing methods", () => {
  it("parses compact-detail labels returned as label-to-node-id maps", async () => {
    const client = new CatmaidClient("https://example.invalid", 1);
    const fetchMock = vi.fn().mockResolvedValue([
      [
        [22107946, null, 2, 23697030.0, 15055839.0, 16651262.0, 2000.0, 5],
        [22107955, 22107954, 2, 23705874.0, 15093672.0, 16682375.0, 2000.0, 5],
        [22107959, 22107958, 2, 23704520.0, 15085237.0, 16708998.0, 2000.0, 5],
      ],
      [],
      {
        "afonso reviewed it": [22107946],
        "test 123 4": [22107955],
        ends: [22107959],
      },
      [],
      [],
    ]);
    (client as any).fetch = fetchMock;

    await expect(client.getSkeleton(2)).resolves.toEqual([
      {
        id: 22107946,
        parent_id: null,
        x: 23697030,
        y: 15055839,
        z: 16651262,
        skeleton_id: 2,
        radius: 2000,
        confidence: 100,
        labels: ["afonso reviewed it"],
      },
      {
        id: 22107955,
        parent_id: 22107954,
        x: 23705874,
        y: 15093672,
        z: 16682375,
        skeleton_id: 2,
        radius: 2000,
        confidence: 100,
        labels: ["test 123 4"],
      },
      {
        id: 22107959,
        parent_id: 22107958,
        x: 23704520,
        y: 15085237,
        z: 16708998,
        skeleton_id: 2,
        radius: 2000,
        confidence: 100,
        labels: ["ends"],
      },
    ]);
  });

  it("merges skeletons using from/to treenode ids", async () => {
    const client = new CatmaidClient("https://example.invalid", 1);
    const fetchMock = vi.fn().mockResolvedValue({
      result_skeleton_id: 17,
      deleted_skeleton_id: 21,
      stable_annotation_swap: false,
    });
    (client as any).fetch = fetchMock;

    await expect(client.mergeSkeletons(101, 202)).resolves.toEqual({
      resultSkeletonId: 17,
      deletedSkeletonId: 21,
      stableAnnotationSwap: false,
    });

    expect(fetchMock).toHaveBeenCalledTimes(1);
    expect(fetchMock.mock.calls[0]?.[0]).toBe("skeleton/join");
    expect(
      (fetchMock.mock.calls[0]?.[1] as { body: URLSearchParams }).body.get(
        "from_id",
      ),
    ).toBe("101");
    expect(
      (fetchMock.mock.calls[0]?.[1] as { body: URLSearchParams }).body.get(
        "to_id",
      ),
    ).toBe("202");
  });

  it("returns treenode and skeleton ids from addNode", async () => {
    const client = new CatmaidClient("https://example.invalid", 1);
    const fetchMock = vi
      .fn()
      .mockResolvedValue({ treenode_id: 88, skeleton_id: 13 });
    (client as any).fetch = fetchMock;

    await expect(client.addNode(13, 1, 2, 3, 7)).resolves.toEqual({
      treenodeId: 88,
      skeletonId: 13,
    });
  });

  it("reroots skeletons using treenode ids", async () => {
    const client = new CatmaidClient("https://example.invalid", 1);
    const fetchMock = vi
      .fn()
      .mockResolvedValue({ newroot: 202, skeleton_id: 17 });
    (client as any).fetch = fetchMock;

    await expect(client.rerootSkeleton(202)).resolves.toBeUndefined();

    expect(fetchMock).toHaveBeenCalledTimes(1);
    expect(fetchMock.mock.calls[0]?.[0]).toBe("skeleton/reroot");
    expect(
      (fetchMock.mock.calls[0]?.[1] as { body: URLSearchParams }).body.get(
        "treenode_id",
      ),
    ).toBe("202");
    expect(
      (fetchMock.mock.calls[0]?.[1] as { body: URLSearchParams }).body.get(
        "state",
      ),
    ).toBe(JSON.stringify({ nocheck: true }));
  });
});
