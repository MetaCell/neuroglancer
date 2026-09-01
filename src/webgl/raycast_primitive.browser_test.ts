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

// A 64 pixel square viewport, and a camera at the origin looking down -z with a
// 45 degree vertical field of view. So a z of -1 is one unit ahead, and a radius r
// there spans r / tan(22.5 degrees) in NDC, which is 2.414 * r.
const VIEWPORT = 64;
const FIELD_OF_VIEW = Math.PI / 4;

// Writes for every fragment the bounding quad rasterises, so the result measures
// the quad the vertex stage emitted.
const SHADE_WHOLE_QUAD = "out_color = vec4(1.0);\n";
// Runs the real hit test first, so a miss discards and the result measures the
// primitive's own surface.
const SHADE_SURFACE = `${glsl_raycastFragmentSetup}out_color = vec4(1.0);\n`;
// Red carries the axial fraction. Green marks a fragment that survived the hit
// test, since a fraction of zero is indistinguishable from an unwritten pixel.
const SHADE_AXIAL_FRACTION = `${glsl_raycastFragmentSetup}out_color = vec4(raycastConeAxialFraction, 1.0, 0.0, 1.0);\n`;

type Point = readonly [number, number, number];
// A radius may be a plain number, or a GLSL expression for the tests that drive the
// pixel-radius helpers.
type Radius = number | string;

interface ConeSpec {
  readonly endpointA: Point;
  readonly endpointB: Point;
  readonly radiusA: Radius;
  readonly radiusB: Radius;
  readonly clipRadiusA?: Radius;
  readonly clipRadiusB?: Radius;
}

interface SphereSpec {
  readonly center: Point;
  readonly radius: Radius;
}

function glslPoint(point: Point): string {
  return `vec3(${point.map((value) => value.toFixed(4)).join(", ")})`;
}

function glslRadius(radius: Radius): string {
  return typeof radius === "number" ? radius.toFixed(5) : radius;
}

function render(
  gl: GL,
  definePrimitive: (builder: ShaderBuilder) => void,
  emitPrimitive: string,
  fragmentMain: string,
): Uint8Array {
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
    initializeRaycastPrimitiveShader(
      shader,
      mat4.perspective(mat4.create(), FIELD_OF_VIEW, 1, 0.1, 20),
      { width: VIEWPORT, height: VIEWPORT },
    );
    gl.viewport(0, 0, VIEWPORT, VIEWPORT);
    gl.clearColor(0, 0, 0, 0);
    gl.clear(WebGL2RenderingContext.COLOR_BUFFER_BIT);
    drawQuads(gl, 1, 1);
    const pixels = new Uint8Array(VIEWPORT * VIEWPORT * 4);
    gl.readPixels(
      0,
      0,
      VIEWPORT,
      VIEWPORT,
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

function drawCone(gl: GL, spec: ConeSpec, fragmentMain: string): Uint8Array {
  const { clipRadiusA = 0, clipRadiusB = 0 } = spec;
  return render(
    gl,
    defineRaycastConeShader,
    `emitRaycastCone(${glslPoint(spec.endpointA)}, ${glslPoint(spec.endpointB)},
                     ${glslRadius(spec.radiusA)}, ${glslRadius(spec.radiusB)},
                     ${glslRadius(clipRadiusA)}, ${glslRadius(clipRadiusB)});`,
    fragmentMain,
  );
}

function drawSphere(
  gl: GL,
  spec: SphereSpec,
  fragmentMain: string,
): Uint8Array {
  return render(
    gl,
    defineRaycastSphereShader,
    `emitRaycastSphere(${glslPoint(spec.center)}, ${glslRadius(spec.radius)});`,
    fragmentMain,
  );
}

function coveredFraction(pixels: Uint8Array): number {
  let covered = 0;
  for (let i = 0; i < VIEWPORT * VIEWPORT; ++i) {
    if (pixels[i * 4] !== 0) ++covered;
  }
  return covered / (VIEWPORT * VIEWPORT);
}

// readPixels returns rows bottom up, so index 0 is the lowest drawn row.
function coveredWidthByRow(pixels: Uint8Array): number[] {
  const widths: number[] = [];
  for (let row = 0; row < VIEWPORT; ++row) {
    let width = 0;
    for (let column = 0; column < VIEWPORT; ++column) {
      if (pixels[(row * VIEWPORT + column) * 4 + 1] !== 0) ++width;
    }
    if (width > 0) widths.push(width);
  }
  return widths;
}

// Red of the first shaded pixel in each row, bottom up. Values run 0 to 255.
function axialFractionByRow(pixels: Uint8Array): number[] {
  const fractions: number[] = [];
  for (let row = 0; row < VIEWPORT; ++row) {
    for (let column = 0; column < VIEWPORT; ++column) {
      const offset = (row * VIEWPORT + column) * 4;
      if (pixels[offset + 1] !== 0) {
        fractions.push(pixels[offset]);
        break;
      }
    }
  }
  return fractions;
}

describe("raycast cone", () => {
  it("publishes its depth and lighting to a consumer's own emit helper", () => {
    webglTest((gl) => {
      const builder = new ShaderBuilder(gl);
      builder.addOutputBuffer("vec4", "out_color", 0);
      defineRaycastConeShader(builder);
      builder.setVertexMain(
        `emitRaycastCone(vec3(0.0), vec3(0.0, 1.0, 0.0), 0.1, 0.2, 0.0, 0.0);`,
      );
      // A helper sees the published globals, not main's locals.
      builder.addFragmentCode(`
void emitShaded() {
  out_color = vec4(raycastLightingFactor, raycastSurfaceDepth,
                   raycastConeAxialFraction, 1.0);
}
`);
      builder.setFragmentMain(`${glsl_raycastFragmentSetup}emitShaded();\n`);
      builder.build().dispose();
    });
  });

  it("bounds a visible cone to a small part of the viewport", () => {
    webglTest((gl) => {
      // Upright, one unit ahead, 0.6 long and 0.1 across. Its silhouette is about
      // 0.6 by 0.1 in raycast units, which is well under a tenth of the viewport.
      const pixels = drawCone(
        gl,
        {
          endpointA: [0, -0.3, -1],
          endpointB: [0, 0.3, -1],
          radiusA: 0.05,
          radiusB: 0.05,
        },
        SHADE_WHOLE_QUAD,
      );
      expect(coveredFraction(pixels)).toBeGreaterThan(0);
      expect(coveredFraction(pixels)).toBeLessThan(0.5);
    });
  });

  it("culls a cone behind the camera", () => {
    webglTest((gl) => {
      // Positive z is behind the camera, which looks down -z.
      const pixels = drawCone(
        gl,
        {
          endpointA: [0, -0.3, 1],
          endpointB: [0, 0.3, 1],
          radiusA: 0.05,
          radiusB: 0.05,
        },
        SHADE_WHOLE_QUAD,
      );
      expect(coveredFraction(pixels)).toBe(0);
    });
  });

  it("culls a cone that wraps the camera", () => {
    webglTest((gl) => {
      // The axis passes 0.2 in front of the camera and the radius is 0.5, so the
      // camera is inside. That surface has no bounded screen footprint, and
      // covering the viewport instead would shade every pixel of a depth-writing
      // shader once per such edge.
      const pixels = drawCone(
        gl,
        {
          endpointA: [-1, 0, -0.2],
          endpointB: [1, 0, -0.2],
          radiusA: 0.5,
          radiusB: 0.5,
        },
        SHADE_WHOLE_QUAD,
      );
      expect(coveredFraction(pixels)).toBe(0);
    });
  });

  it("culls a cone with no radius", () => {
    webglTest((gl) => {
      // Both radius helpers return zero for a point at or behind the eye, so this
      // runs every frame on a skeleton with geometry behind the camera. There is
      // no surface to hit, so shading its quad would be pure waste.
      const pixels = drawCone(
        gl,
        {
          endpointA: [0, -0.3, -1],
          endpointB: [0, 0.3, -1],
          radiusA: 0,
          radiusB: 0,
        },
        SHADE_WHOLE_QUAD,
      );
      expect(coveredFraction(pixels)).toBe(0);
    });
  });

  it("keeps a segment whose far endpoint has passed behind the camera", () => {
    webglTest((gl) => {
      const endpointA: Point = [-0.3, -0.2, -1];
      const endpointB: Point = [0.5, 0.4, 1];
      // Endpoint B is behind the camera, so its own pixel radius is zero and it
      // alone leaves nothing to draw.
      expect(
        coveredFraction(
          drawCone(
            gl,
            {
              endpointA,
              endpointB,
              radiusA: 0,
              radiusB: `getRaycastRadiusForPixels(${glslPoint(endpointB)}, 1.0)`,
            },
            SHADE_WHOLE_QUAD,
          ),
        ),
      ).toBe(0);

      // The segment helper makes that end borrow the other's radius, which keeps
      // the half that is still in view.
      const radii = `getRaycastSegmentRadiiForPixels(${glslPoint(endpointA)}, ${glslPoint(endpointB)}, 1.0)`;
      const borrowed = coveredFraction(
        drawCone(
          gl,
          {
            endpointA,
            endpointB,
            radiusA: `${radii}.x`,
            radiusB: `${radii}.y`,
          },
          SHADE_WHOLE_QUAD,
        ),
      );
      expect(borrowed).toBeGreaterThan(0.25);
      expect(borrowed).toBeLessThan(1);
    });
  });

  it("draws a constant width when both end radii match", () => {
    webglTest((gl) => {
      // Equal radii must leave the taper rate at zero, so the quadratic collapses
      // to the fixed-radius circle test. Any drift shows as a width that changes
      // along a cone that should not taper.
      const widths = coveredWidthByRow(
        drawCone(
          gl,
          {
            endpointA: [0, -0.3, -1],
            endpointB: [0, 0.3, -1],
            radiusA: 0.05,
            radiusB: 0.05,
          },
          SHADE_SURFACE,
        ),
      );
      expect(widths.length).toBeGreaterThan(8);
      // One pixel covers where the silhouette falls between sample points.
      expect(Math.max(...widths) - Math.min(...widths)).toBeLessThanOrEqual(1);
    });
  });

  it("tapers the width between two different end radii", () => {
    webglTest((gl) => {
      // Endpoint A is the lower end, so the width grows from bottom to top. The
      // radii are wide enough that whole-pixel rasterisation does not dominate.
      const widths = coveredWidthByRow(
        drawCone(
          gl,
          {
            endpointA: [0, -0.3, -1],
            endpointB: [0, 0.3, -1],
            radiusA: 0.03,
            radiusB: 0.12,
          },
          SHADE_SURFACE,
        ),
      );
      expect(widths.length).toBeGreaterThan(16);
      // The rows nearest each end are left out. The ends are open, so the rim
      // there projects as an ellipse and the silhouette closes over them.
      const interior = widths.slice(
        Math.round(widths.length * 0.15),
        Math.round(widths.length * 0.85),
      );
      // Radius runs 0.0435 to 0.1065 across this slice, a ratio of 2.45.
      expect(interior.at(-1)! / interior[0]).toBeGreaterThan(1.8);
      expect(interior.at(-1)! / interior[0]).toBeLessThan(3.2);
      for (let i = 1; i < interior.length; ++i) {
        expect(interior[i]).toBeGreaterThanOrEqual(interior[i - 1] - 1);
      }
    });
  });

  it("draws a segment at the radius its pixel width asks for", () => {
    webglTest((gl) => {
      const endpointA: Point = [0, -0.3, -1];
      const endpointB: Point = [0, 0.3, -1];
      const radii = `getRaycastSegmentRadiiForPixels(${glslPoint(endpointA)}, ${glslPoint(endpointB)}, 6.0)`;
      const widths = coveredWidthByRow(
        drawCone(
          gl,
          {
            endpointA,
            endpointB,
            radiusA: `${radii}.x`,
            radiusB: `${radii}.y`,
          },
          SHADE_SURFACE,
        ),
      );
      expect(widths.length).toBeGreaterThan(8);
      // Both ends sit at the same depth, so both ask for 6 device pixels. That is
      // a 12 pixel width, plus or minus a pixel of rasterisation.
      expect(Math.max(...widths)).toBeGreaterThan(10);
      expect(Math.max(...widths)).toBeLessThan(14);
    });
  });

  it("reports the axial fraction from 0 at endpoint A to 1 at endpoint B", () => {
    webglTest((gl) => {
      // A consumer mixes an attribute's two end values by this fraction, so a
      // constant would colour a whole edge from one endpoint.
      const fractions = axialFractionByRow(
        drawCone(
          gl,
          {
            endpointA: [0, -0.3, -1],
            endpointB: [0, 0.3, -1],
            radiusA: 0.05,
            radiusB: 0.05,
          },
          SHADE_AXIAL_FRACTION,
        ),
      );
      expect(fractions.length).toBeGreaterThan(8);
      expect(fractions[0]).toBeLessThan(16);
      expect(fractions.at(-1)!).toBeGreaterThan(239);
      for (let i = 1; i < fractions.length; ++i) {
        expect(fractions[i]).toBeGreaterThanOrEqual(fractions[i - 1]);
      }
    });
  });

  it("clips the surface around an endpoint", () => {
    webglTest((gl) => {
      // The clip radius hands the region around a joint to the ball drawn there.
      // The surface sits one radius from the axis, so a clip radius of 0.15
      // reaches sqrt(0.15^2 - 0.05^2) = 0.1414 along a 0.6 long axis. That is the
      // lowest 23.6 percent of it, or 60 as a 0 to 255 value.
      const clipped = axialFractionByRow(
        drawCone(
          gl,
          {
            endpointA: [0, -0.3, -1],
            endpointB: [0, 0.3, -1],
            radiusA: 0.05,
            radiusB: 0.05,
            clipRadiusA: 0.15,
          },
          SHADE_AXIAL_FRACTION,
        ),
      );
      expect(clipped.length).toBeGreaterThan(8);
      expect(clipped[0]).toBeGreaterThan(45);
      expect(clipped[0]).toBeLessThan(78);
      expect(clipped.at(-1)!).toBeGreaterThan(239);
    });
  });

  it("leaves the surface alone when the clip radius cannot reach it", () => {
    webglTest((gl) => {
      const spec: ConeSpec = {
        endpointA: [0, -0.3, -1],
        endpointB: [0, 0.3, -1],
        radiusA: 0.05,
        radiusB: 0.05,
      };
      // A clip radius under the cone's own radius never reaches the surface, which
      // is already 0.05 from the axis everywhere.
      expect(
        axialFractionByRow(
          drawCone(gl, { ...spec, clipRadiusA: 0.04 }, SHADE_AXIAL_FRACTION),
        ),
      ).toEqual(axialFractionByRow(drawCone(gl, spec, SHADE_AXIAL_FRACTION)));
    });
  });
});

describe("raycast sphere", () => {
  it("publishes its depth and lighting to a consumer's own emit helper", () => {
    webglTest((gl) => {
      const builder = new ShaderBuilder(gl);
      builder.addOutputBuffer("vec4", "out_color", 0);
      defineRaycastSphereShader(builder);
      builder.setVertexMain("emitRaycastSphere(vec3(0.0, 0.0, -1.0), 0.2);");
      builder.addFragmentCode(`
void emitShaded() {
  out_color = vec4(vec3(raycastLightingFactor), raycastSurfaceDepth);
}
`);
      builder.setFragmentMain(`${glsl_raycastFragmentSetup}emitShaded();\n`);
      builder.build().dispose();
    });
  });

  it("bounds a sphere to the square around its silhouette", () => {
    webglTest((gl) => {
      // A radius of 0.05 one unit ahead has a silhouette 0.12086 in NDC, which is
      // 3.87 pixels of a 64 pixel viewport. The square around that disc is 59.8
      // pixels, or 0.0146 of the viewport. The disc itself is 0.0115. The
      // projected box this replaced measured 0.0376, most of that a fixed two
      // pixel margin the exact bound does not need.
      const pixels = drawSphere(
        gl,
        { center: [0, 0, -1], radius: 0.05 },
        SHADE_WHOLE_QUAD,
      );
      expect(coveredFraction(pixels)).toBeGreaterThan(0.012);
      expect(coveredFraction(pixels)).toBeLessThan(0.017);
    });
  });

  it("culls a sphere behind the camera", () => {
    webglTest((gl) => {
      const pixels = drawSphere(
        gl,
        { center: [0, 0, 1], radius: 0.05 },
        SHADE_WHOLE_QUAD,
      );
      expect(coveredFraction(pixels)).toBe(0);
    });
  });

  it("culls a sphere with no radius", () => {
    webglTest((gl) => {
      const pixels = drawSphere(
        gl,
        { center: [0, 0, -1], radius: 0 },
        SHADE_WHOLE_QUAD,
      );
      expect(coveredFraction(pixels)).toBe(0);
    });
  });

  it("bounds a sphere without clipping its surface", () => {
    webglTest((gl) => {
      // The silhouette of a sphere of radius r at distance d has radius
      // r / sqrt(d^2 - r^2), which for r of 0.2 at one unit is 4 percent more area
      // than the plain r / d disc. The conic gives that exactly, so the shaded
      // surface has to exceed the plain disc. Falling short of it is what a quad
      // clipping the sphere would produce.
      //
      // A radius of 0.2 one unit ahead spans 15.5 pixels of a 64 pixel viewport,
      // so the r / d disc is 0.1831 of it.
      const shaded = coveredFraction(
        drawSphere(gl, { center: [0, 0, -1], radius: 0.2 }, SHADE_SURFACE),
      );
      expect(shaded).toBeGreaterThan(0.1831);
      expect(shaded).toBeLessThan(0.21);
    });
  });

  it("takes the whole viewport when the sphere crosses the eye plane", () => {
    webglTest((gl) => {
      // Centred 0.3 ahead with a radius of 0.5, so the sphere spans the eye plane.
      // The silhouette conic is an ellipse only while the sphere clears that
      // plane. Past it, part of the sphere projects arbitrarily far, so the whole
      // viewport is the only honest bound.
      const pixels = drawSphere(
        gl,
        { center: [0, 0, -0.3], radius: 0.5 },
        SHADE_WHOLE_QUAD,
      );
      expect(coveredFraction(pixels)).toBe(1);
    });
  });
});
