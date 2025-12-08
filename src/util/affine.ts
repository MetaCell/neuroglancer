/**
 * @license
 * Copyright 2025 Google Inc.
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

import { multiply } from "#src/util/matrix.js";

export function extractScalesFromAffineMatrix(
  affineTransform: Float64Array,
  rank: number,
): Float64Array {
  // For now, use the L2 norm method. But should replace this by either
  // SVD, or exploring how the transform affects a known object
  // such as a unit sphere
  // TODO address the above
  const scales = new Float64Array(rank);
  for (let i = 0; i < rank; ++i) {
    let sumSquares = 0;
    for (let j = 0; j < rank; ++j) {
      const val = affineTransform[j * (rank + 1) + i];
      sumSquares += val * val;
    }
    scales[i] = Math.sqrt(sumSquares);
  }
  return scales;
}

export function makeAffineRelativeToBaseTransform(
  affineTransform: Float64Array,
  baseTransformInverse: Float64Array,
  rank: number,
): Float64Array {
  const relativeTransform = new Float64Array(baseTransformInverse.length);
  multiply(
    relativeTransform,
    rank + 1,
    baseTransformInverse,
    rank + 1,
    affineTransform,
    rank + 1,
    rank + 1,
    rank + 1,
    rank + 1,
  );
  return relativeTransform;
}
