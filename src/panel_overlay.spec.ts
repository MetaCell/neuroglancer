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
import type { MouseSelectionState } from "#src/layer/index.js";
import {
  PickingIndicatorOverlay,
  projectToViewport,
} from "#src/panel_overlay.js";
import type { ProjectionParameters } from "#src/projection_parameters.js";
import { WatchableValue } from "#src/trackable_value.js";
import { mat4 } from "#src/util/geom.js";
import { NullarySignal } from "#src/util/signal.js";

function makeParameters(projectionMat: mat4): ProjectionParameters {
  const viewMatrix = mat4.fromTranslation(mat4.create(), [0, 0, -1]);
  return {
    projectionMat,
    viewProjectionMat: mat4.multiply(mat4.create(), projectionMat, viewMatrix),
    logicalWidth: 200,
    logicalHeight: 100,
    displayDimensionRenderInfo: {
      displayDimensionIndices: Int32Array.of(0, 1, 2),
    },
  } as unknown as ProjectionParameters;
}

const perspective = makeParameters(
  mat4.perspective(mat4.create(), Math.PI / 2, 2, 0.5, 1.5),
);
const orthographic = makeParameters(
  mat4.ortho(mat4.create(), -2, 2, -1, 1, 0.5, 1.5),
);

describe("projectToViewport", () => {
  it("puts the focal point at the panel center, on the focal plane", () => {
    for (const parameters of [perspective, orthographic]) {
      const point = projectToViewport(parameters, [0, 0, 0])!;
      expect(point.viewportLeft).toBeCloseTo(100);
      expect(point.viewportTop).toBeCloseTo(50);
      expect(point.perspectiveDivideFactor).toBeCloseTo(1);
      expect(point.focalPlaneDepthFraction).toBeCloseTo(0);
    }
  });

  it("magnifies a nearer point only under perspective projection", () => {
    const nearer = [0, 0, 0.5];
    expect(
      projectToViewport(perspective, nearer)!.perspectiveDivideFactor,
    ).toBeCloseTo(2);
    expect(
      projectToViewport(orthographic, nearer)!.perspectiveDivideFactor,
    ).toBeCloseTo(1);
  });

  it("reports the depth fraction linearly under both projections", () => {
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
  const mouseState = {
    active: true,
    position: new Float32Array([0, 0, 0]),
    coordinateSpace: emptyInvalidCoordinateSpace,
    changed: new NullarySignal(),
  } as unknown as MouseSelectionState;

  it("sizes, fades and places one ring from the projected point", () => {
    const container = document.createElement("div");
    const overlay = new PickingIndicatorOverlay(
      mouseState,
      new WatchableValue(true),
    ).createPanelOverlay(container, () => ({
      viewportLeft: 100,
      viewportTop: 50,
      perspectiveDivideFactor: 1.5,
      focalPlaneDepthFraction: -0.25,
    }));
    overlay.update();
    const ring = container.firstElementChild as HTMLElement;
    expect(container.childElementCount).toBe(1);
    expect(ring.hidden).toBe(false);
    expect(ring.style.width).toBe("21px");
    expect(ring.style.opacity).toBe("0.75");
    expect(ring.style.transform).toBe("translate(89.5px, 39.5px)");

    overlay.dispose();
    expect(container.childElementCount).toBe(0);
  });

  it("hides the ring when nothing is picked", () => {
    const container = document.createElement("div");
    const overlay = new PickingIndicatorOverlay(
      mouseState,
      new WatchableValue(true),
    ).createPanelOverlay(container, () => undefined);
    overlay.update();
    expect((container.firstElementChild as HTMLElement).hidden).toBe(true);
  });
});
