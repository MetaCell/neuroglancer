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
import type { ProjectionParameters } from "#src/projection_parameters.js";
import { mat4 } from "#src/util/geom.js";
import { projectToViewport } from "#src/util/viewport_projection.js";

function makeParameters(projectionMat: mat4): ProjectionParameters {
  // Camera at the origin looking down -z, with the focal point at depth 1.
  const viewMatrix = mat4.fromTranslation(mat4.create(), [0, 0, -1]);
  const viewProjectionMat = mat4.multiply(
    mat4.create(),
    projectionMat,
    viewMatrix,
  );
  return {
    viewProjectionMat,
    logicalWidth: 200,
    logicalHeight: 100,
    displayDimensionRenderInfo: {
      displayDimensionIndices: Int32Array.of(0, 1, 2),
    },
  } as unknown as ProjectionParameters;
}

describe("projectToViewport", () => {
  it("maps the focal point to the viewport center with unit depth scale", () => {
    for (const projectionMat of [
      mat4.perspective(mat4.create(), Math.PI / 2, 2, 0.5, 1.5),
      mat4.ortho(mat4.create(), -2, 2, -1, 1, 0.5, 1.5),
    ]) {
      const point = projectToViewport(makeParameters(projectionMat), [0, 0, 0]);
      expect(point).toBeDefined();
      expect(point!.x).toBeCloseTo(100);
      expect(point!.y).toBeCloseTo(50);
      expect(point!.depthScale).toBeCloseTo(1);
    }
  });

  it("scales with depth only under perspective projection", () => {
    const nearer = [0, 0, 0.5];
    const perspective = makeParameters(
      mat4.perspective(mat4.create(), Math.PI / 2, 2, 0.25, 1.75),
    );
    expect(projectToViewport(perspective, nearer)!.depthScale).toBeCloseTo(2);
    const orthographic = makeParameters(
      mat4.ortho(mat4.create(), -2, 2, -1, 1, 0.25, 1.75),
    );
    expect(projectToViewport(orthographic, nearer)!.depthScale).toBeCloseTo(1);
  });

  it("culls points behind the camera and outside the depth range", () => {
    const parameters = makeParameters(
      mat4.perspective(mat4.create(), Math.PI / 2, 2, 0.5, 1.5),
    );
    expect(projectToViewport(parameters, [0, 0, 2])).toBeUndefined();
    expect(projectToViewport(parameters, [0, 0, -2])).toBeUndefined();
  });
});
