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

import type { ChunkLayout } from "#src/sliceview/chunk_layout.js";
import { mat4 } from "#src/util/geom.js";
import { nearlyEqual, nearlyInteger, isPowerOfTwo } from "#src/util/number.js";

export function detectNestedOctree(
  outerLayout: ChunkLayout,
  innerLayout: ChunkLayout,
): boolean {
  // Reject if rank is less than 3 for either source
  if (outerLayout.finiteRank < 3 || innerLayout.finiteRank < 3) return false;

  // Determine the map from outer chunk-layout coordinates to inner
  // chunk-layout coordinates (outer-local -> global -> inner-local).
  const m = mat4.create();
  mat4.multiply(m, innerLayout.invTransform, outerLayout.transform);

  // First check the diagonal elements are all the same
  // and are a power of two
  // scale represents how many inner units per outer unit
  const scale = m[0];
  if (!nearlyInteger(scale) || !isPowerOfTwo(scale)) return false;
  if (!nearlyEqual(scale, m[5]) || !nearlyEqual(scale, m[10])) return false;

  // Check off diagonal elements are approximately zero, as that would
  // indicate some kind of rotation or shear needed to align the chunks
  const offDiagonals = [1, 2, 4, 6, 8, 9];
  for (const i of offDiagonals) {
    if (!nearlyEqual(m[i], 0)) return false;
  }

  // We need the chunks to nest cleanly, first based on just size
  for (let i = 0; i < 3; ++i) {
    const innerChunkSize = innerLayout.size[i];
    const scaledOuterChunkSize = outerLayout.size[i] * Math.round(scale);
    if (scaledOuterChunkSize % innerChunkSize !== 0) return false;
    // TODO what translation requirements do we have?
    // is 0 required, or would translation % innerSize being 0 be enough?
    // for now let's assume we need the root / origin to match
    if (!nearlyEqual(m[12 + i], 0)) return false;
  }

  return true;
}

/**
 * Given ChunkLayouts ordered finest (index 0) to coarsest (index length-1),
 * returns an array of the same length where entry `i` (for `i >= 1`) is
 * `detectNestedOctree(chunkLayouts[i], chunkLayouts[i - 1])` -- whether scale
 * `i` nests inside its immediate finer neighbor `i - 1`. Entry 0 is true.
 */
export function detectNestedOctreeLevels(
  chunkLayouts: readonly ChunkLayout[],
): boolean[] {
  const result = new Array<boolean>(chunkLayouts.length);
  result[0] = true; // For the finest chunk, it has no "parent" level
  for (let i = 1; i < chunkLayouts.length; ++i) {
    result[i] = detectNestedOctree(chunkLayouts[i], chunkLayouts[i - 1]);
  }
  return result;
}
