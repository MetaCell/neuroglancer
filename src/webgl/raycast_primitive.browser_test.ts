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
import {
  glsl_raycastFragmentSetup,
  initializeRaycastPrimitiveShader,
} from "#src/webgl/raycast_primitive.js";
import { defineRaycastSphereShader } from "#src/webgl/raycast_sphere.js";
import { defineRaycastConeShader } from "#src/webgl/raycast_truncated_cone.js";
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
// Radius of the cone and ball that the shaded tests draw.
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

// Fraction of the viewport the primitive's own surface shades, with the real
// fragment setup so a miss discards. Unlike quad coverage this measures the
// surface, so it falls if a bounding quad clips the primitive.
function measureShadedCoverage(
  gl: GL,
  definePrimitive: (builder: ShaderBuilder) => void,
  emitPrimitive: string,
): number {
  const pixels = renderPrimitive(
    gl,
    definePrimitive,
    emitPrimitive,
    glsl_raycastFragmentSetup + "out_color = vec4(1.0, 1.0, 1.0, 1.0);\n",
  );
  const size = COVERAGE_VIEWPORT_SIZE;
  let shaded = 0;
  for (let i = 0; i < size * size; ++i) {
    if (pixels[i * 4] !== 0) ++shaded;
  }
  return shaded / (size * size);
}

// `depth` is the raycast-space z, negative for in front of the camera.
function coneCoverage(gl: GL, depth: number) {
  return measureQuadCoverage(
    gl,
    defineRaycastConeShader,
    `emitRaycastCone(vec3(0.0, -0.3, ${depth.toFixed(4)}),
                     vec3(0.0, 0.3, ${depth.toFixed(4)}),
                     ${PRIMITIVE_TEST_RADIUS}, ${PRIMITIVE_TEST_RADIUS},
                     0.0, 0.0);`,
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

  it("compiles the cone shader", () => {
    buildShader(
      defineRaycastConeShader,
      `emitRaycastCone(vec3(0.0), vec3(0.0, 1.0, 0.0),
                    getRaycastRadiusForPixels(vec3(0.0), 2.0),
                    getRaycastRadiusForPixels(vec3(0.0, 1.0, 0.0), 2.0),
                    1.0, 1.0);`,
    );
  });

  it("bounds a cone tightly, and culls one behind the camera", () => {
    webglTest((gl) => {
      const visible = coneCoverage(gl, -1);
      expect(visible).toBeGreaterThan(0);
      expect(visible).toBeLessThan(0.5);
      expect(coneCoverage(gl, 1)).toBe(0);
    });
  });

  // The camera sits inside this cone, whose surface then has no bounded screen
  // footprint. Covering the viewport instead would shade every pixel of a
  // depth-writing fragment shader, once for each such edge.
  it("culls a cone that wraps the camera", () => {
    webglTest((gl) => {
      const coverage = measureQuadCoverage(
        gl,
        defineRaycastConeShader,
        `emitRaycastCone(vec3(-1.0, 0.0, -0.2), vec3(1.0, 0.0, -0.2),
                     0.5, 0.5, 0.0, 0.0);`,
      );
      expect(coverage).toBe(0);
    });
  });

  // This edge crosses the eye plane, so one endpoint has no on-screen size and its
  // own radius is zero. Borrowing the other end's radius keeps the visible half.
  it("keeps an edge whose endpoint has passed behind the camera", () => {
    webglTest((gl) => {
      const endpoints = "vec3(-0.3, -0.2, -1.0), vec3(0.5, 0.4, 1.0)";
      const coverage = (radii: string) =>
        measureQuadCoverage(
          gl,
          defineRaycastConeShader,
          `emitRaycastCone(${endpoints}, ${radii}, 0.0, 0.0);`,
        );
      // Endpoint B is behind the eye, so its own radius alone leaves nothing.
      expect(
        coverage("0.0, getRaycastRadiusForPixels(vec3(0.5, 0.4, 1.0), 1.0)"),
      ).toBe(0);
      const borrowed = coverage(
        `getRaycastSegmentRadiiForPixels(${endpoints}, 1.0).x,
         getRaycastSegmentRadiiForPixels(${endpoints}, 1.0).y`,
      );
      expect(borrowed).toBeGreaterThan(0.25);
      expect(borrowed).toBeLessThan(1);
    });
  });

  // An upright cone one unit in front of the camera, shaded with the axial
  // fraction. Endpoint A is the lower end, and readPixels returns rows bottom up,
  // so the result runs from endpoint A to endpoint B. Values are 0 to 255.
  function renderUprightCone(
    gl: GL,
    radiusA: string,
    radiusB: string,
    clipRadiusA: number,
    clipRadiusB: number,
  ): Uint8Array {
    return renderPrimitive(
      gl,
      defineRaycastConeShader,
      `emitRaycastCone(vec3(0.0, -0.3, -1.0), vec3(0.0, 0.3, -1.0),
                     ${radiusA}, ${radiusB}, ${clipRadiusA.toFixed(4)},
                     ${clipRadiusB.toFixed(4)});`,
      glsl_raycastFragmentSetup +
        "out_color = vec4(raycastConeAxialFraction, 1.0, 0.0, 1.0);\n",
    );
  }

  function shadedConeAxialFractionByRow(
    gl: GL,
    clipRadiusA: number,
    clipRadiusB: number,
  ): number[] {
    const size = COVERAGE_VIEWPORT_SIZE;
    const pixels = renderUprightCone(
      gl,
      PRIMITIVE_TEST_RADIUS,
      PRIMITIVE_TEST_RADIUS,
      clipRadiusA,
      clipRadiusB,
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

  // Covered pixels per row, from the endpoint A end to the endpoint B end.
  function coneWidthByRow(gl: GL, radiusA: string, radiusB: string): number[] {
    const size = COVERAGE_VIEWPORT_SIZE;
    const pixels = renderUprightCone(gl, radiusA, radiusB, 0, 0);
    const widthByRow: number[] = [];
    for (let row = 0; row < size; ++row) {
      let width = 0;
      for (let column = 0; column < size; ++column) {
        if (pixels[(row * size + column) * 4 + 1] !== 0) ++width;
      }
      if (width > 0) widthByRow.push(width);
    }
    return widthByRow;
  }

  // A skeleton edge carries a vertex attribute at each end, and the consumer mixes
  // the two by this fraction. A constant value would colour a whole edge from one
  // endpoint, so the test checks that it runs the length of the cone.
  it("reports where a cone hit falls between the endpoints", () => {
    webglTest((gl) => {
      const fractionByRow = shadedConeAxialFractionByRow(gl, 0, 0);
      expect(fractionByRow.length).toBeGreaterThan(8);
      const [first, last] = [fractionByRow[0], fractionByRow.at(-1)!];
      expect(first).toBeLessThan(16);
      expect(last).toBeGreaterThan(239);
      for (let i = 1; i < fractionByRow.length; ++i) {
        expect(fractionByRow[i]).toBeGreaterThanOrEqual(fractionByRow[i - 1]);
      }
    });
  });

  // Equal end radii must leave the taper rate at zero, so the quadratic collapses
  // to the fixed-radius circle test. A cylinder is the common case, and any drift
  // here would show as a width that changes along a cone that should not taper.
  it("draws an exact cylinder when both end radii match", () => {
    webglTest((gl) => {
      const widthByRow = coneWidthByRow(
        gl,
        PRIMITIVE_TEST_RADIUS,
        PRIMITIVE_TEST_RADIUS,
      );
      expect(widthByRow.length).toBeGreaterThan(8);
      const widest = Math.max(...widthByRow);
      const narrowest = Math.min(...widthByRow);
      // One pixel covers where the silhouette falls between sample points.
      expect(widest - narrowest).toBeLessThanOrEqual(1);
    });
  });

  // The taper is what holds one on-screen width along a receding edge. Endpoint A
  // is the lower end here, so the drawn width has to grow from bottom to top.
  //
  // The rows nearest each end are left out. The ends are open, so the rim there
  // projects as an ellipse and the silhouette closes over the last few rows.
  it("tapers between two different end radii", () => {
    webglTest((gl) => {
      // Wide enough that whole-pixel rasterisation does not dominate the ratio.
      const widthByRow = coneWidthByRow(gl, "0.03", "0.12");
      expect(widthByRow.length).toBeGreaterThan(16);
      const interior = widthByRow.slice(
        Math.round(widthByRow.length * 0.15),
        Math.round(widthByRow.length * 0.85),
      );
      // Radius runs 0.0435 to 0.1065 across this slice, a ratio of 2.45.
      expect(interior.at(-1)! / interior[0]).toBeGreaterThan(1.8);
      expect(interior.at(-1)! / interior[0]).toBeLessThan(3.2);
      for (let i = 1; i < interior.length; ++i) {
        expect(interior[i]).toBeGreaterThanOrEqual(interior[i - 1] - 1);
      }
    });
  });

  // Both ends at the same depth ask for the same radius, and the requested pixel
  // radius has to come back as the drawn width. This checks the whole chain from a
  // pixel radius through the per-end radii to the rasterised silhouette.
  it("draws a segment at the requested pixel radius", () => {
    webglTest((gl) => {
      const endpoints = "vec3(0.0, -0.3, -1.0), vec3(0.0, 0.3, -1.0)";
      const radii = `getRaycastSegmentRadiiForPixels(${endpoints}, 6.0)`;
      const widthByRow = coneWidthByRow(gl, `${radii}.x`, `${radii}.y`);
      expect(widthByRow.length).toBeGreaterThan(8);
      // A radius of 6 device pixels is a 12 pixel width, plus or minus a pixel.
      // The test above already covers the width holding along the cone.
      expect(Math.max(...widthByRow)).toBeGreaterThan(10);
      expect(Math.max(...widthByRow)).toBeLessThan(14);
    });
  });

  // The clip radius hands the region around a joint to the ball drawn there. The
  // surface sits one radius from the axis, so a clip radius of 0.15 reaches
  // sqrt(0.15^2 - 0.05^2) = 0.1414 along a 0.6 long axis: the lowest 23.6 percent.
  it("clips the cone surface around an endpoint", () => {
    webglTest((gl) => {
      const clipped = shadedConeAxialFractionByRow(gl, 0.15, 0);
      expect(clipped.length).toBeGreaterThan(8);
      // 0.236 of the way along, as a 0-to-255 value, is 60.
      expect(clipped[0]).toBeGreaterThan(45);
      expect(clipped[0]).toBeLessThan(78);
      expect(clipped.at(-1)!).toBeGreaterThan(239);
      // A clip radius under the cone radius cannot reach the surface at all.
      const unreachable = shadedConeAxialFractionByRow(gl, 0.04, 0);
      expect(unreachable).toEqual(shadedConeAxialFractionByRow(gl, 0, 0));
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
          defineRaycastConeShader,
          `emitRaycastCone(vec3(0.0, -0.3, -1.0), vec3(0.0, 0.3, -1.0),
                     0.0, 0.0, 0.0, 0.0);`,
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

  // The bound is the exact silhouette conic, so the quad is the square around that
  // ellipse and needs no margin. A radius of 0.05 one unit ahead has a silhouette
  // 0.12086 in NDC, which is 3.87 pixels of a 64 pixel viewport, so the square is
  // 59.8 pixels or 0.0146 of it. The disc itself is 0.0115. The projected box this
  // replaced measured 0.0376, most of that its fixed two pixel margin.
  it("bounds a sphere to its silhouette, and culls one behind the camera", () => {
    webglTest((gl) => {
      const visible = sphereCoverage(gl, -1);
      expect(visible).toBeGreaterThan(0.012);
      expect(visible).toBeLessThan(0.017);
      expect(sphereCoverage(gl, 1)).toBe(0);
    });
  });

  // A tighter bound only pays if it still contains the whole surface. The exact
  // silhouette of a sphere of radius r at distance d has radius r / sqrt(d^2 - r^2),
  // which for r of 0.2 at one unit is 4 percent more area than the r / d disc. The
  // conic gives that exactly, so the shaded surface has to exceed the plain disc
  // rather than fall short of it, which is what a quad clipping the sphere would do.
  it("bounds a sphere without clipping its surface", () => {
    webglTest((gl) => {
      const shaded = measureShadedCoverage(
        gl,
        defineRaycastSphereShader,
        "emitRaycastSphere(vec3(0.0, 0.0, -1.0), 0.2);",
      );
      // A radius of 0.2 one unit ahead spans 15.5 pixels of a 64 pixel viewport,
      // so the r / d disc is 0.1831 of it.
      expect(shaded).toBeGreaterThan(0.1831);
      expect(shaded).toBeLessThan(0.21);
    });
  });

  // The conic is an ellipse only while the sphere clears the eye plane. Past that,
  // part of the sphere projects arbitrarily far, so the whole viewport is the only
  // honest bound, and nothing may be lost by taking it.
  it("takes the whole viewport when the sphere crosses the eye plane", () => {
    webglTest((gl) => {
      // Centered 0.3 ahead with a radius of 0.5, so the sphere spans the eye plane.
      const coverage = measureQuadCoverage(
        gl,
        defineRaycastSphereShader,
        "emitRaycastSphere(vec3(0.0, 0.0, -0.3), 0.5);",
      );
      expect(coverage).toBe(1);
    });
  });
});
