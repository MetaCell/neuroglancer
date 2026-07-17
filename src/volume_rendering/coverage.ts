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

  // Map the inner layout voxel coords to the outer layout voxel coords
  // first map inner to global, then global to outer
  const m = mat4.create();
  mat4.multiply(m, outerLayout.transform, innerLayout.transform);

  // First check the diagonal elements are all the same
  // and are a power of two
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
