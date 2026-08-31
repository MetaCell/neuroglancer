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
import { mat4 } from "#src/util/geom.js";
import type { GL } from "#src/webgl/context.js";
import { drawQuads } from "#src/webgl/quad.js";
import { defineRaycastCylinderShader } from "#src/webgl/raycast_cylinder.js";
import {
  glsl_raycastFragmentSetup,
  initializeRaycastPrimitiveShader,
} from "#src/webgl/raycast_primitive.js";
import { defineRaycastSphereShader } from "#src/webgl/raycast_sphere.js";
import { ShaderBuilder } from "#src/webgl/shader.js";
import { webglTest } from "#src/webgl/testing.js";
import { defineVertexId, VertexIdHelper } from "#src/webgl/vertex_id.js";

function buildShader(
  definePrimitive: (builder: ShaderBuilder) => void,
  emitPrimitive: string,
) {
  webglTest((gl) => {
    const builder = new ShaderBuilder(gl);
    builder.addOutputBuffer("vec4", "out_color", 0);
    definePrimitive(builder);
    builder.setVertexMain(emitPrimitive);
    // Mirrors how a consumer emits: from a helper function, which can only see
    // the published globals and not main's locals.
    builder.addFragmentCode(`
void emitShaded() {
  out_color = vec4(vec3(raycastLightingFactor), raycastSurfaceDepth);
}
`);
    builder.setFragmentMain(glsl_raycastFragmentSetup + "emitShaded();\n");
    builder.build().dispose();
  });
}

// A camera at the raycast-space origin looking down -z, so a raycast-space z of
// -1 is one unit in front of the camera.
const COVERAGE_VIEWPORT_SIZE = 64;
const COVERAGE_NEAR_BOUND = 0.1;
const COVERAGE_FAR_BOUND = 20;
// Radius of the tube and ball that the shaded tests draw.
const PRIMITIVE_TEST_RADIUS = "0.05";

function renderPrimitive(
  gl: GL,
  definePrimitive: (builder: ShaderBuilder) => void,
  emitPrimitive: string,
  fragmentMain: string,
): Uint8Array {
  const size = COVERAGE_VIEWPORT_SIZE;
  const builder = new ShaderBuilder(gl);
  builder.addOutputBuffer("vec4", "out_color", 0);
  defineVertexId(builder);
  definePrimitive(builder);
  builder.setVertexMain(emitPrimitive);
  builder.setFragmentMain(fragmentMain);
  const shader = builder.build();
  const vertexIdHelper = VertexIdHelper.get(gl);
  try {
    shader.bind();
    vertexIdHelper.enable();
    const projectionMatrix = mat4.perspective(
      mat4.create(),
      Math.PI / 4,
      1,
      COVERAGE_NEAR_BOUND,
      COVERAGE_FAR_BOUND,
    );
    initializeRaycastPrimitiveShader(shader, projectionMatrix, {
      width: size,
      height: size,
    });
    gl.viewport(0, 0, size, size);
    gl.clearColor(0, 0, 0, 0);
    gl.clear(WebGL2RenderingContext.COLOR_BUFFER_BIT);
    drawQuads(gl, 1, 1);
    const pixels = new Uint8Array(size * size * 4);
    gl.readPixels(
      0,
      0,
      size,
      size,
      WebGL2RenderingContext.RGBA,
      WebGL2RenderingContext.UNSIGNED_BYTE,
      pixels,
    );
    return pixels;
  } finally {
    vertexIdHelper.disable();
    shader.dispose();
  }
}

// Fraction of the viewport that the bounding quad rasterises.  The fragment
// shader writes unconditionally, so this measures the vertex stage: an
// out-of-range quad is counted here but discarded by the real shader, making it
// invisible to any test of the shaded result.
function measureQuadCoverage(
  gl: GL,
  definePrimitive: (builder: ShaderBuilder) => void,
  emitPrimitive: string,
): number {
  const pixels = renderPrimitive(
    gl,
    definePrimitive,
    emitPrimitive,
    "out_color = vec4(1.0, 1.0, 1.0, 1.0);\n",
  );
  const size = COVERAGE_VIEWPORT_SIZE;
  let covered = 0;
  for (let i = 0; i < size * size; ++i) {
    if (pixels[i * 4] !== 0) ++covered;
  }
  return covered / (size * size);
}

// `depth` is the raycast-space z, negative for in front of the camera.
function cylinderCoverage(gl: GL, depth: number) {
  return measureQuadCoverage(
    gl,
    defineRaycastCylinderShader,
    `emitRaycastCylinder(vec3(0.0, -0.3, ${depth.toFixed(4)}),
                     vec3(0.0, 0.3, ${depth.toFixed(4)}),
                     ${PRIMITIVE_TEST_RADIUS}, 0.0, 0.0);`,
  );
}

function sphereCoverage(gl: GL, depth: number) {
  return measureQuadCoverage(
    gl,
    defineRaycastSphereShader,
    `emitRaycastSphere(vec3(0.0, 0.0, ${depth.toFixed(4)}), ${PRIMITIVE_TEST_RADIUS});`,
  );
}

describe("raycast primitives", () => {
  it("compiles the sphere shader", () => {
    buildShader(
      defineRaycastSphereShader,
      `emitRaycastSphere(vec3(0.0), getRaycastRadiusForPixels(vec3(0.0), 5.0));`,
    );
  });

  it("compiles the cylinder shader", () => {
    buildShader(
      defineRaycastCylinderShader,
      `emitRaycastCylinder(vec3(0.0), vec3(0.0, 1.0, 0.0),
                    getRaycastRadiusForPixels(vec3(0.0), 2.0), 1.0, 1.0);`,
    );
  });

  it("bounds a cylinder tightly, and culls one behind the camera", () => {
    webglTest((gl) => {
      const visible = cylinderCoverage(gl, -1);
      expect(visible).toBeGreaterThan(0);
      expect(visible).toBeLessThan(0.5);
      expect(cylinderCoverage(gl, 1)).toBe(0);
    });
  });

  // The camera sits inside this tube, whose surface then has no bounded screen
  // footprint. Covering the viewport instead would shade every pixel of a
  // depth-writing fragment shader, once for each such edge.
  it("culls a cylinder that wraps the camera", () => {
    webglTest((gl) => {
      const coverage = measureQuadCoverage(
        gl,
        defineRaycastCylinderShader,
        `emitRaycastCylinder(vec3(-1.0, 0.0, -0.2), vec3(1.0, 0.0, -0.2),
                     0.5, 0.0, 0.0);`,
      );
      expect(coverage).toBe(0);
    });
  });

  // This edge crosses the eye plane, so its midpoint lies behind the camera. A
  // radius read there is zero and the near half of the edge is lost with it.
  it("keeps an edge whose midpoint has passed behind the camera", () => {
    webglTest((gl) => {
      const endpoints = "vec3(-0.3, -0.2, -1.0), vec3(0.5, 0.4, 1.0)";
      const coverage = (radius: string) =>
        measureQuadCoverage(
          gl,
          defineRaycastCylinderShader,
          `emitRaycastCylinder(${endpoints}, ${radius}, 0.0, 0.0);`,
        );
      expect(
        coverage("getRaycastRadiusForPixels(vec3(0.1, 0.1, 0.0), 1.0)"),
      ).toBe(0);
      const atNearEndpoint = coverage(
        `getRaycastSegmentRadiusForPixels(${endpoints}, 1.0)`,
      );
      expect(atNearEndpoint).toBeGreaterThan(0.25);
      expect(atNearEndpoint).toBeLessThan(1);
    });
  });

  // An upright tube one unit in front of the camera, shaded with the axial
  // fraction. Endpoint A is the lower end, and readPixels returns rows bottom up,
  // so the result runs from endpoint A to endpoint B. Values are 0 to 255.
  function shadedCylinderAxialFractionByRow(
    gl: GL,
    clipRadiusA: number,
    clipRadiusB: number,
  ): number[] {
    const size = COVERAGE_VIEWPORT_SIZE;
    const pixels = renderPrimitive(
      gl,
      defineRaycastCylinderShader,
      `emitRaycastCylinder(vec3(0.0, -0.3, -1.0), vec3(0.0, 0.3, -1.0),
                     ${PRIMITIVE_TEST_RADIUS}, ${clipRadiusA.toFixed(4)},
                     ${clipRadiusB.toFixed(4)});`,
      glsl_raycastFragmentSetup +
        "out_color = vec4(raycastCylinderAxialFraction, 1.0, 0.0, 1.0);\n",
    );
    const fractionByRow: number[] = [];
    for (let row = 0; row < size; ++row) {
      for (let column = 0; column < size; ++column) {
        const offset = (row * size + column) * 4;
        if (pixels[offset + 1] !== 0) {
          fractionByRow.push(pixels[offset]);
          break;
        }
      }
    }
    return fractionByRow;
  }

  // A skeleton edge carries a vertex attribute at each end, and the consumer mixes
  // the two by this fraction. A constant value would colour a whole edge from one
  // endpoint, so the test checks that it runs the length of the tube.
  it("reports where a cylinder hit falls between the endpoints", () => {
    webglTest((gl) => {
      const fractionByRow = shadedCylinderAxialFractionByRow(gl, 0, 0);
      expect(fractionByRow.length).toBeGreaterThan(8);
      const [first, last] = [fractionByRow[0], fractionByRow.at(-1)!];
      expect(first).toBeLessThan(16);
      expect(last).toBeGreaterThan(239);
      for (let i = 1; i < fractionByRow.length; ++i) {
        expect(fractionByRow[i]).toBeGreaterThanOrEqual(fractionByRow[i - 1]);
      }
    });
  });

  // The clip radius hands the region around a joint to the ball drawn there. The
  // surface sits one radius from the axis, so a clip radius of 0.15 reaches
  // sqrt(0.15^2 - 0.05^2) = 0.1414 along a 0.6 long axis: the lowest 23.6 percent.
  it("clips the cylinder surface around an endpoint", () => {
    webglTest((gl) => {
      const clipped = shadedCylinderAxialFractionByRow(gl, 0.15, 0);
      expect(clipped.length).toBeGreaterThan(8);
      // 0.236 of the way along, as a 0-to-255 value, is 60.
      expect(clipped[0]).toBeGreaterThan(45);
      expect(clipped[0]).toBeLessThan(78);
      expect(clipped.at(-1)!).toBeGreaterThan(239);
      // A clip radius under the tube radius cannot reach the surface at all.
      const unreachable = shadedCylinderAxialFractionByRow(gl, 0.04, 0);
      expect(unreachable).toEqual(shadedCylinderAxialFractionByRow(gl, 0, 0));
    });
  });

  // A radius of zero has no surface for the fragment shader to hit, and reaching
  // the quad emitters with one leaves the radius vectors degenerate. Both radius
  // helpers return zero for a point at or behind the eye, so this runs every frame
  // on any skeleton with geometry behind the camera.
  it("culls a zero-radius primitive", () => {
    webglTest((gl) => {
      expect(
        measureQuadCoverage(
          gl,
          defineRaycastCylinderShader,
          `emitRaycastCylinder(vec3(0.0, -0.3, -1.0), vec3(0.0, 0.3, -1.0),
                     0.0, 0.0, 0.0);`,
        ),
      ).toBe(0);
      expect(
        measureQuadCoverage(
          gl,
          defineRaycastSphereShader,
          "emitRaycastSphere(vec3(0.0, 0.0, -1.0), 0.0);",
        ),
      ).toBe(0);
    });
  });

  it("bounds a sphere tightly, and culls one behind the camera", () => {
    webglTest((gl) => {
      const visible = sphereCoverage(gl, -1);
      expect(visible).toBeGreaterThan(0);
      expect(visible).toBeLessThan(0.5);
      expect(sphereCoverage(gl, 1)).toBe(0);
    });
  });
});
