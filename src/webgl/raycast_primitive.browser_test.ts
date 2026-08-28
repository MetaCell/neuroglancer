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

// Fraction of the viewport that the bounding quad rasterises.  The fragment
// shader writes unconditionally, so this measures the vertex stage: an
// out-of-range quad is counted here but discarded by the real shader, making it
// invisible to any test of the shaded result.
function measureQuadCoverage(
  gl: GL,
  definePrimitive: (builder: ShaderBuilder) => void,
  emitPrimitive: string,
): number {
  const size = COVERAGE_VIEWPORT_SIZE;
  const builder = new ShaderBuilder(gl);
  builder.addOutputBuffer("vec4", "out_color", 0);
  defineVertexId(builder);
  definePrimitive(builder);
  builder.setVertexMain(emitPrimitive);
  builder.setFragmentMain("out_color = vec4(1.0, 1.0, 1.0, 1.0);\n");
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
    let covered = 0;
    for (let i = 0; i < size * size; ++i) {
      if (pixels[i * 4] !== 0) ++covered;
    }
    return covered / (size * size);
  } finally {
    vertexIdHelper.disable();
    shader.dispose();
  }
}

// `depth` is the raycast-space z, negative for in front of the camera.
function cylinderCoverage(gl: GL, depth: number) {
  return measureQuadCoverage(
    gl,
    defineRaycastCylinderShader,
    `emitRaycastCylinder(vec3(0.0, -0.3, ${depth.toFixed(4)}),
                     vec3(0.0, 0.3, ${depth.toFixed(4)}), 0.05, 0.0, 0.0);`,
  );
}

function sphereCoverage(gl: GL, depth: number) {
  return measureQuadCoverage(
    gl,
    defineRaycastSphereShader,
    `emitRaycastSphere(vec3(0.0, 0.0, ${depth.toFixed(4)}), 0.05);`,
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

  it("bounds a sphere tightly, and culls one behind the camera", () => {
    webglTest((gl) => {
      const visible = sphereCoverage(gl, -1);
      expect(visible).toBeGreaterThan(0);
      expect(visible).toBeLessThan(0.5);
      expect(sphereCoverage(gl, 1)).toBe(0);
    });
  });
});
