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

import { describe, expect, it } from "vitest";
import { emptyInvalidCoordinateSpace } from "#src/coordinate_transform.js";
import type { DisplayContext } from "#src/display_context.js";
import type { MouseSelectionState } from "#src/layer/index.js";
import type { PanelOverlaySource } from "#src/panel_overlay.js";
import {
  PanelOverlayManager,
  PickingIndicatorOverlay,
  projectToViewport,
} from "#src/panel_overlay.js";
import type { ProjectionParameters } from "#src/projection_parameters.js";
import { WatchableSet, WatchableValue } from "#src/trackable_value.js";
import { mat4 } from "#src/util/geom.js";
import { NullarySignal } from "#src/util/signal.js";

function makeParameters(options: {
  projectionMat: mat4;
  focalDistance: number;
  logicalWidth: number;
  logicalHeight: number;
}): ProjectionParameters {
  const { projectionMat, focalDistance, logicalWidth, logicalHeight } = options;
  const viewMatrix = mat4.fromTranslation(mat4.create(), [
    0,
    0,
    -focalDistance,
  ]);
  return {
    projectionMat,
    viewProjectionMat: mat4.multiply(mat4.create(), projectionMat, viewMatrix),
    logicalWidth,
    logicalHeight,
    displayDimensionRenderInfo: {
      displayDimensionIndices: Int32Array.of(0, 1, 2),
    },
  } as unknown as ProjectionParameters;
}

function makeMouseState(): MouseSelectionState {
  return {
    active: true,
    position: new Float32Array([0, 0, 0]),
    coordinateSpace: emptyInvalidCoordinateSpace,
    changed: new NullarySignal(),
  } as unknown as MouseSelectionState;
}

function makeContext(): DisplayContext {
  return {
    panelOverlays: new WatchableSet<PanelOverlaySource>(),
    scheduleOverlayUpdate: () => {},
  } as unknown as DisplayContext;
}

function makeSource(): PanelOverlaySource {
  return {
    updateNeeded: new NullarySignal(),
    createPanelOverlay: () => ({ update() {}, dispose() {} }),
  };
}

describe("projectToViewport", () => {
  it("puts the focal point at the panel center, on the focal plane", () => {
    for (const projectionMat of [
      mat4.perspective(mat4.create(), Math.PI / 2, 2, 0.5, 1.5),
      mat4.ortho(mat4.create(), -2, 2, -1, 1, 0.5, 1.5),
    ]) {
      const parameters = makeParameters({
        projectionMat,
        focalDistance: 1,
        logicalWidth: 200,
        logicalHeight: 100,
      });
      const point = projectToViewport(parameters, [0, 0, 0])!;
      expect(point.viewportLeft).toBeCloseTo(100);
      expect(point.viewportTop).toBeCloseTo(50);
      expect(point.perspectiveDivideFactor).toBeCloseTo(1);
      expect(point.focalPlaneDepthFraction).toBeCloseTo(0);
    }
  });

  it("magnifies a nearer point only under perspective projection", () => {
    const perspective = makeParameters({
      projectionMat: mat4.perspective(mat4.create(), Math.PI / 2, 2, 0.5, 1.5),
      focalDistance: 1,
      logicalWidth: 200,
      logicalHeight: 100,
    });
    const orthographic = makeParameters({
      projectionMat: mat4.ortho(mat4.create(), -2, 2, -1, 1, 0.5, 1.5),
      focalDistance: 1,
      logicalWidth: 200,
      logicalHeight: 100,
    });
    const halfWayToNear = [0, 0, 0.5];
    expect(
      projectToViewport(perspective, halfWayToNear)!.perspectiveDivideFactor,
    ).toBeCloseTo(2);
    expect(
      projectToViewport(orthographic, halfWayToNear)!.perspectiveDivideFactor,
    ).toBeCloseTo(1);
  });

  it("reports the depth fraction linearly under both projections", () => {
    const perspective = makeParameters({
      projectionMat: mat4.perspective(mat4.create(), Math.PI / 2, 2, 0.5, 1.5),
      focalDistance: 1,
      logicalWidth: 200,
      logicalHeight: 100,
    });
    const orthographic = makeParameters({
      projectionMat: mat4.ortho(mat4.create(), -2, 2, -1, 1, 0.5, 1.5),
      focalDistance: 1,
      logicalWidth: 200,
      logicalHeight: 100,
    });
    const halfWayToFar = [0, 0, -0.25];
    expect(
      projectToViewport(perspective, halfWayToFar)!.focalPlaneDepthFraction,
    ).toBeCloseTo(0.5);
    expect(
      projectToViewport(orthographic, halfWayToFar)!.focalPlaneDepthFraction,
    ).toBeCloseTo(0.5);
  });
});

describe("PickingIndicatorOverlay", () => {
  it("centres one ring on the picked point, scaled by nearness and faded by depth", () => {
    const panel = document.createElement("div");
    const context = makeContext();
    context.panelOverlays.add(
      new PickingIndicatorOverlay(makeMouseState(), new WatchableValue(true)),
    );
    const manager = new PanelOverlayManager(panel, context, () => ({
      viewportLeft: 100,
      viewportTop: 50,
      perspectiveDivideFactor: 1.5,
      focalPlaneDepthFraction: -0.25,
    }));
    manager.update();
    const rings = panel.querySelectorAll<HTMLElement>(
      ".neuroglancer-picking-indicator",
    );
    expect(rings.length).toBe(1);
    const [ring] = rings;
    expect(ring.hidden).toBe(false);
    expect(ring.style.width).toBe("21px");
    expect(ring.style.opacity).toBe("0.75");
    expect(ring.style.transform).toBe("translate(89.5px, 39.5px)");
  });

  it("hides the ring when the picked point is outside the panel", () => {
    const panel = document.createElement("div");
    const context = makeContext();
    context.panelOverlays.add(
      new PickingIndicatorOverlay(makeMouseState(), new WatchableValue(true)),
    );
    const manager = new PanelOverlayManager(panel, context, () => undefined);
    manager.update();
    const ring = panel.querySelector<HTMLElement>(
      ".neuroglancer-picking-indicator",
    )!;
    expect(ring.hidden).toBe(true);
  });

  it("hides the ring while the picking indicator setting is off", () => {
    const panel = document.createElement("div");
    const context = makeContext();
    const showPickingIndicator = new WatchableValue(false);
    context.panelOverlays.add(
      new PickingIndicatorOverlay(makeMouseState(), showPickingIndicator),
    );
    const manager = new PanelOverlayManager(panel, context, () => ({
      viewportLeft: 100,
      viewportTop: 50,
      perspectiveDivideFactor: 1,
      focalPlaneDepthFraction: 0,
    }));
    manager.update();
    const ring = panel.querySelector<HTMLElement>(
      ".neuroglancer-picking-indicator",
    )!;
    expect(ring.closest("[hidden]")).not.toBe(null);

    showPickingIndicator.value = true;
    manager.update();
    expect(ring.closest("[hidden]")).toBe(null);
  });

  it("draws the ring above an overlay of default priority", () => {
    const panel = document.createElement("div");
    const context = makeContext();
    context.panelOverlays.add(
      new PickingIndicatorOverlay(makeMouseState(), new WatchableValue(true)),
    );
    context.panelOverlays.add(makeSource());
    new PanelOverlayManager(panel, context, () => undefined);
    const ring = panel.querySelector<HTMLElement>(
      ".neuroglancer-picking-indicator",
    )!;
    const [ringLayer, otherLayer] = ring.parentElement!.parentElement!
      .children as HTMLCollectionOf<HTMLElement>;
    expect(ringLayer.contains(ring)).toBe(true);
    expect(Number(ringLayer.style.zIndex)).toBeGreaterThan(
      Number(otherLayer.style.zIndex),
    );
  });

  it("removes the ring when the panel is disposed", () => {
    const panel = document.createElement("div");
    const context = makeContext();
    context.panelOverlays.add(
      new PickingIndicatorOverlay(makeMouseState(), new WatchableValue(true)),
    );
    const manager = new PanelOverlayManager(panel, context, () => undefined);
    manager.dispose();
    expect(panel.querySelector(".neuroglancer-picking-indicator")).toBe(null);
  });
});
