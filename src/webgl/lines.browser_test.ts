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
import type { GL } from "#src/webgl/context.js";
import {
  defineLineShader,
  drawLines,
  initializeLineShader,
} from "#src/webgl/lines.js";
import { ShaderBuilder } from "#src/webgl/shader.js";
import { webglTest } from "#src/webgl/testing.js";
import { defineVertexId, VertexIdHelper } from "#src/webgl/vertex_id.js";

// A 64 pixel square viewport. Endpoints are given in clip space, so a test can put
// one outside the depth range without setting up a projection.
const VIEWPORT = 64;

// Clip space x, y, z, w. A point is inside the depth range when the magnitude of z
// is at most w, so a w of 1 puts the range at -1 to 1.
type ClipPoint = readonly [number, number, number, number];

interface LineSpec {
  readonly endpointA: ClipPoint;
  readonly endpointB: ClipPoint;
  readonly widthInPixels: number;
  readonly clipRadiusInPixels: number;
}

function glslClipPoint(point: ClipPoint): string {
  return `vec4(${point.map((value) => value.toFixed(4)).join(", ")})`;
}

function drawLine(gl: GL, spec: LineSpec): Uint8Array {
  const builder = new ShaderBuilder(gl);
  builder.addOutputBuffer("vec4", "out_color", 0);
  defineVertexId(builder);
  defineLineShader(builder, { endpointClipping: true });
  builder.setVertexMain(
    `emitLine(${glslClipPoint(spec.endpointA)}, ${glslClipPoint(spec.endpointB)},
              ${spec.widthInPixels.toFixed(1)}, ${spec.clipRadiusInPixels.toFixed(1)});`,
  );
  builder.setFragmentMain("out_color = vec4(getLineAlpha());\n");
  const shader = builder.build();
  const vertexIdHelper = VertexIdHelper.get(gl);
  try {
    shader.bind();
    vertexIdHelper.enable();
    initializeLineShader(
      shader,
      { width: VIEWPORT, height: VIEWPORT },
      /*featherWidthInPixels=*/ 0,
    );
    gl.viewport(0, 0, VIEWPORT, VIEWPORT);
    gl.clearColor(0, 0, 0, 0);
    gl.clear(WebGL2RenderingContext.COLOR_BUFFER_BIT);
    drawLines(gl, 1, 1);
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

function coveredCount(pixels: Uint8Array): number {
  let covered = 0;
  for (let i = 0; i < VIEWPORT * VIEWPORT; ++i) {
    if (pixels[i * 4] !== 0) ++covered;
  }
  return covered;
}

function isCovered(pixels: Uint8Array, x: number, y: number): boolean {
  return pixels[(y * VIEWPORT + x) * 4] !== 0;
}

describe("line endpoint clipping", () => {
  it("removes a disc at each endpoint, so a node drawn there has room", () => {
    webglTest((gl) => {
      // Horizontal across the middle, from NDC x of -0.5 to 0.5. Endpoint A lands
      // a quarter across the viewport, at pixel 16.
      const spec = {
        endpointA: [-0.5, 0, 0, 1],
        endpointB: [0.5, 0, 0, 1],
        widthInPixels: 6,
      } as const;
      const unclipped = drawLine(gl, { ...spec, clipRadiusInPixels: 0 });
      const clipped = drawLine(gl, { ...spec, clipRadiusInPixels: 10 });

      expect(coveredCount(clipped)).toBeGreaterThan(0);
      expect(coveredCount(clipped)).toBeLessThan(coveredCount(unclipped));
      expect(isCovered(unclipped, 16, VIEWPORT / 2)).toBe(true);
      expect(isCovered(clipped, 16, VIEWPORT / 2)).toBe(false);
    });
  });

  it("clips at an endpoint inside the depth range but not at one outside it", () => {
    webglTest((gl) => {
      // Endpoint A has z of -1.5, so it is outside the range and the GPU clips away
      // the node quad there. Its disc must not apply. A sits at pixel 24 and the
      // depth range trims the drawn line to start at pixel 32, so a 10 pixel radius
      // would otherwise reach past pixel 33.
      //
      // Endpoint B has z of 0.5, is inside the range, and keeps its disc. It sits
      // at pixel 56, which is the drawn end, so the disc clears pixel 55.
      //
      // Measuring either disc from the clipped end rather than the endpoint as
      // given puts A's disc at pixel 32, which also takes pixel 33.
      const spec = {
        endpointA: [-0.25, 0, -1.5, 1],
        endpointB: [0.75, 0, 0.5, 1],
        widthInPixels: 6,
      } as const;
      const unclipped = drawLine(gl, { ...spec, clipRadiusInPixels: 0 });
      const clipped = drawLine(gl, { ...spec, clipRadiusInPixels: 10 });

      expect(isCovered(unclipped, 33, VIEWPORT / 2)).toBe(true);
      expect(isCovered(clipped, 33, VIEWPORT / 2)).toBe(true);
      expect(isCovered(unclipped, 55, VIEWPORT / 2)).toBe(true);
      expect(isCovered(clipped, 55, VIEWPORT / 2)).toBe(false);
    });
  });

  it("rejects endpoint clipping on a rounded line", () => {
    webglTest((gl) => {
      // The clip lives in getLineAlpha, which a rounded line never calls, so the
      // pair would silently ignore the clip radius.
      expect(() =>
        defineLineShader(new ShaderBuilder(gl), {
          rounded: true,
          endpointClipping: true,
        }),
      ).toThrow(/rounded/);
    });
  });
});
