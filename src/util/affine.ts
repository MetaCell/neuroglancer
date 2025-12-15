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
import type { mat4 } from "#src/util/geom.js";
import * as matrix from "#src/util/matrix.js";

/**
 * Extracts scale factors along each axis from an affine transformation matrix.
 * To do this, we extract the upper-left `rank x rank` submatrix, compute its inverse,
 * and determine how much it scales unit vectors along each axis.
 * This is equivalent to 
 */
export function extractScalesFromAffineMatrix(
  affineTransform: Float64Array | mat4,
  rank: number,
): Float64Array {
  // Extract just the upper left rank x rank submatrix
  const upperLeft = new Float64Array(rank * rank);
  for (let i = 0; i < rank; ++i) {
    for (let j = 0; j < rank; ++j) {
      upperLeft[i * rank + j] =
        affineTransform[i * (rank + 1) + j];
    }
  }

  const inverseTransform = new Float64Array(upperLeft.length);
  // TODO need non-invertible check here?
  matrix.inverse(
    inverseTransform,
    rank,
    upperLeft,
    rank,
    rank,
  );
  const scales = new Float64Array(rank);
  for (let i = 0; i < rank; ++i) {
    let sumSquares = 0;
    const unitVector = new Float64Array(rank);
    unitVector[i] = 1;
    // Apply inverse transform to unit vector along axis i
    const transformedVector = new Float64Array(rank);
    matrix.transformVector(
      transformedVector,
      inverseTransform,
      rank,
      unitVector,
      rank,
    )
    for (let j = 0; j < rank; ++j) {
      const val = transformedVector[j];
      sumSquares += val * val;
    }
    scales[i] = 1 / Math.sqrt(sumSquares);
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
