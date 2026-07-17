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
import { ChunkLayout } from "#src/sliceview/chunk_layout.js";
import { mat4, vec3 } from "#src/util/geom.js";
import { detectNestedOctree } from "#src/volume_rendering/coverage.js";

describe("volume rendering scale nesting (octree detection)", () => {
  it("detects a strict power-of-two octree", () => {
    const outerLayout = new ChunkLayout(
      vec3.fromValues(64, 64, 64),
      mat4.fromValues(2, 0, 0, 0, 0, 2, 0, 0, 0, 0, 2, 0, 0, 0, 0, 1),
      3,
    );
    const innerLayout = new ChunkLayout(
      vec3.fromValues(64, 64, 64),
      mat4.create(),
      3,
    );
    expect(detectNestedOctree(outerLayout, innerLayout)).toBe(true);
  });

  it("detects a nested octree with different chunk sizes", () => {
    const outerLayout = new ChunkLayout(
      vec3.fromValues(64, 64, 64),
      mat4.fromValues(2, 0, 0, 0, 0, 2, 0, 0, 0, 0, 2, 0, 0, 0, 0, 1),
      mat4.create(),
      3,
    );
    const innerLayout = new ChunkLayout(
      vec3.fromValues(32, 32, 32),
      mat4.create(),
      3,
    );
    expect(detectNestedOctree(outerLayout, innerLayout)).toBe(true);
  });

  it("rejects if any scaling factor is different", () => {
    const outerLayout = new ChunkLayout(
      vec3.fromValues(64, 64, 64),
      mat4.fromValues(4, 0, 0, 0, 0, 2, 0, 0, 0, 0, 2, 0, 0, 0, 0, 1),
      3,
    );
    const innerLayout = new ChunkLayout(
      vec3.fromValues(64, 64, 64),
      mat4.create(),
      3,
    );
    expect(detectNestedOctree(outerLayout, innerLayout)).toBe(false);
  });

  it("rejects a non-power of two scaling", () => {
    const outerLayout = new ChunkLayout(
      vec3.fromValues(64, 64, 64),
      mat4.fromValues(3, 0, 0, 0, 0, 3, 0, 0, 0, 0, 3, 0, 0, 0, 0, 1),
      3,
    );
    const innerLayout = new ChunkLayout(
      vec3.fromValues(64, 64, 64),
      mat4.create(),
      3,
    );
    expect(detectNestedOctree(outerLayout, innerLayout)).toBe(false);
  });

  it("rejects a non-axis-aligned (sheared) scale", () => {
    const outerLayout = new ChunkLayout(
      vec3.fromValues(64, 64, 64),
      mat4.fromValues(2, 1, 0, 0, 0, 2, 0, 0, 0, 0, 2, 0, 0, 0, 0, 1),
      3,
    );
    const innerLayout = new ChunkLayout(
      vec3.fromValues(64, 64, 64),
      mat4.create(),
      3,
    );
    expect(detectNestedOctree(outerLayout, innerLayout)).toBe(false);
  });

  it("rejects scales with non-aligned origins", () => {
    const outerLayout = new ChunkLayout(
      vec3.fromValues(64, 64, 64),
      mat4.fromValues(2, 0, 0, 0, 0, 2, 0, 0, 0, 0, 2, 0, 1, 0, 0, 1),
      3,
    );
    const innerLayout = new ChunkLayout(
      vec3.fromValues(64, 64, 64),
      mat4.create(),
      3,
    );
    expect(detectNestedOctree(outerLayout, innerLayout)).toBe(false);
  });
});
