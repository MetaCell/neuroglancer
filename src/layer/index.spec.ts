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

import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import {
  CoordinateSpaceCombiner,
  TrackableCoordinateSpace,
} from "#src/coordinate_transform.js";
import type { LayerListSpecification } from "#src/layer/index.js";
import { GlobalToolBinder } from "#src/ui/tool.js";
import type { MultiToolPaletteState } from "#src/ui/tool_palette.js";
import { RefCounted } from "#src/util/disposable.js";
import { NullarySignal } from "#src/util/signal.js";

if (!("WebGL2RenderingContext" in globalThis)) {
  Object.defineProperty(globalThis, "WebGL2RenderingContext", {
    value: new Proxy(class WebGL2RenderingContext {} as any, {
      get(target, property, receiver) {
        if (Reflect.has(target, property)) {
          return Reflect.get(target, property, receiver);
        }
        return 0;
      },
    }),
    configurable: true,
  });
}

const {
  LayerManager,
  LayerSelectedValues,
  ManagedUserLayer,
  MouseSelectionState,
  TrackableDataSelectionState,
  UserLayer,
} = await import("#src/layer/index.js");

function makeSelectionFixture() {
  const scope = new RefCounted();
  const toolBinder = scope.registerDisposer(
    new GlobalToolBinder(() => {}, {} as MultiToolPaletteState),
  );
  const manager = scope.registerDisposer(new LayerManager());
  scope.registerDisposer(manager.useDirectly());
  const coordinateSpace = new TrackableCoordinateSpace();
  const specification = {
    rootLayers: manager,
    root: {
      coordinateSpaceCombiner: new CoordinateSpaceCombiner(
        coordinateSpace,
        () => true,
      ),
      display: { updateStarted: new NullarySignal(), scheduleRedraw: vi.fn() },
      toolBinder,
    },
  } as unknown as LayerListSpecification;
  const mouse = new MouseSelectionState();
  mouse.active = true;
  const selected = scope.registerDisposer(
    new LayerSelectedValues(manager, mouse),
  );
  const persistent = scope.registerDisposer(
    new TrackableDataSelectionState(coordinateSpace, selected),
  );

  function createLayer(name: string) {
    const managed = new ManagedUserLayer(name, specification);
    const layer = new UserLayer(managed);
    managed.layer = layer;
    manager.addManagedLayer(managed);
    return layer;
  }

  return { scope, manager, mouse, selected, persistent, createLayer };
}

// Refocus restores visible layers before initializationDone. Exercise that
// lifecycle with live selection so restored layers are skipped, then sampled.
describe("layer selection during restoration", () => {
  let fixture: ReturnType<typeof makeSelectionFixture>;

  beforeEach(() => {
    vi.useFakeTimers();
    fixture = makeSelectionFixture();
  });

  afterEach(() => {
    fixture.scope.dispose();
    vi.clearAllTimers();
    vi.useRealTimers();
  });

  it("omits an uninitialized layer from selection reads and serialization", () => {
    const layer = fixture.createLayer("restored");

    expect(fixture.selected.get(layer)).toBeUndefined();
    expect(fixture.selected.toJSON()).toEqual({});

    layer.initializationDone();

    expect(fixture.selected.get(layer)?.localPositionValid).toBe(true);
    expect(fixture.selected.toJSON()).toEqual({ restored: {} });
  });

  it("resamples live selection after initialization without mouse movement", () => {
    fixture.persistent.pin.value = false;
    vi.runAllTimers();
    let layer: InstanceType<typeof UserLayer>;
    expect(() => {
      layer = fixture.createLayer("restored");
    }).not.toThrow();
    vi.runAllTimers();

    expect(fixture.persistent.value?.layers).toEqual([]);
    layer!.initializationDone();
    vi.runAllTimers();

    expect(
      fixture.persistent.value?.layers.map((entry) => entry.layer),
    ).toEqual([layer!]);
    expect(fixture.persistent.value?.layers[0].state.localPositionValid).toBe(
      true,
    );
  });

  it("resamples initialized selection while data sources are still loading", () => {
    const layer = fixture.createLayer("loading");
    const finishLoading = layer.markLoading();
    fixture.persistent.pin.value = false;
    vi.runAllTimers();

    layer.initializationDone();
    vi.runAllTimers();

    expect(layer.isReady).toBe(false);
    expect(
      fixture.persistent.value?.layers.map((entry) => entry.layer),
    ).toEqual([layer]);
    finishLoading();
    expect(layer.isReady).toBe(true);
  });

  it("keeps sampling other initialized layers while a restored layer is pending", () => {
    const pending = fixture.createLayer("pending");
    const initialized = fixture.createLayer("initialized");
    initialized.initializationDone();
    const capture = vi.spyOn(initialized, "captureSelectionState");

    fixture.mouse.setActive(true);

    expect(fixture.selected.get(pending)).toBeUndefined();
    expect(fixture.selected.get(initialized)?.localPositionValid).toBe(true);
    expect(capture).toHaveBeenCalledOnce();

    initialized.managedLayer.setVisible(false);
    expect(fixture.selected.get(initialized)).toBeUndefined();
    fixture.mouse.setActive(false);
    expect(fixture.selected.get(initialized)).toBeUndefined();
  });
});
