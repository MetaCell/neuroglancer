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

const VIEWPORT_SIZE = 64;
const LINE_WIDTH_IN_PIXELS = 6;
const CLIP_RADIUS_IN_PIXELS = 10;

// `endpointsClip` gives both endpoints in clip space, so a test can put an endpoint
// outside the depth range without setting up a projection.
function drawClippedLine(
  gl: GL,
  endpointsClip: string,
  clipRadiusInPixels: number,
): Uint8Array {
  const size = VIEWPORT_SIZE;
  const builder = new ShaderBuilder(gl);
  builder.addOutputBuffer("vec4", "out_color", 0);
  defineVertexId(builder);
  defineLineShader(builder, /*rounded=*/ false, /*endpointClipping=*/ true);
  builder.setVertexMain(
    `emitLine(${endpointsClip}, ${LINE_WIDTH_IN_PIXELS.toFixed(1)}, ` +
      `${clipRadiusInPixels.toFixed(1)});`,
  );
  builder.setFragmentMain("out_color = vec4(getLineAlpha());\n");
  const shader = builder.build();
  const vertexIdHelper = VertexIdHelper.get(gl);
  try {
    shader.bind();
    vertexIdHelper.enable();
    initializeLineShader(
      shader,
      { width: size, height: size },
      /*featherWidthInPixels=*/ 0,
    );
    gl.viewport(0, 0, size, size);
    gl.clearColor(0, 0, 0, 0);
    gl.clear(WebGL2RenderingContext.COLOR_BUFFER_BIT);
    drawLines(gl, 1, 1);
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
    const covered = new Uint8Array(size * size);
    for (let i = 0; i < size * size; ++i) {
      covered[i] = pixels[i * 4] !== 0 ? 1 : 0;
    }
    return covered;
  } finally {
    vertexIdHelper.disable();
    shader.dispose();
  }
}

function countCovered(covered: Uint8Array): number {
  let total = 0;
  for (const value of covered) total += value;
  return total;
}

function isCovered(covered: Uint8Array, x: number, y: number): boolean {
  return covered[y * VIEWPORT_SIZE + x] === 1;
}

describe("line endpoint clipping", () => {
  // A clip disc belongs at each endpoint, so that a node drawn there has room.
  it("removes a disc at each endpoint", () => {
    webglTest((gl) => {
      const endpoints = "vec4(-0.5, 0.0, 0.0, 1.0), vec4(0.5, 0.0, 0.0, 1.0)";
      const unclipped = drawClippedLine(gl, endpoints, 0);
      const clipped = drawClippedLine(gl, endpoints, CLIP_RADIUS_IN_PIXELS);
      expect(countCovered(clipped)).toBeGreaterThan(0);
      expect(countCovered(clipped)).toBeLessThan(countCovered(unclipped));
      // Endpoint A sits at NDC x of -0.5, which is a quarter across the viewport.
      const endpointAX = VIEWPORT_SIZE / 4;
      const centerY = VIEWPORT_SIZE / 2;
      expect(isCovered(unclipped, endpointAX, centerY)).toBe(true);
      expect(isCovered(clipped, endpointAX, centerY)).toBe(false);
    });
  });

  // Measuring the disc from the depth-clipped ends would eat the drawn line where
  // no node exists, the node itself having been clipped away with the segment.
  it("measures from the given endpoints, not the depth-clipped ones", () => {
    webglTest((gl) => {
      // z runs from -3 to 3, so only the middle third survives the depth range.
      // Both given endpoints end up more than one clip radius clear of what is
      // drawn, so the discs must remove nothing.
      const endpoints = "vec4(-1.0, 0.0, -3.0, 1.0), vec4(1.0, 0.0, 3.0, 1.0)";
      const unclipped = drawClippedLine(gl, endpoints, 0);
      const clipped = drawClippedLine(gl, endpoints, CLIP_RADIUS_IN_PIXELS);
      expect(countCovered(unclipped)).toBeGreaterThan(0);
      expect(countCovered(clipped)).toBe(countCovered(unclipped));
    });
  });
});
