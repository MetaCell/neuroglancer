/**
 * @license
 * Copyright 2023 Google Inc.
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

import {getMemoizedBuffer} from 'neuroglancer/webgl/buffer';
import {GL} from 'neuroglancer/webgl/context';
import {VERTICES_PER_QUAD} from 'neuroglancer/webgl/quad';

/**
 * Create a Float32Array of vertices gridded in a rectangle
 */
export function createGriddedRectangleArray(
    numGrids: number, startX: number = -1, endX: number = 1, startY: number = 1,
    endY: number = -1): Float32Array {
  const result = new Float32Array(numGrids * VERTICES_PER_QUAD * 2);
  const step = (endX - startX) / numGrids;
  let currentx = startX;
  for (let i = 0; i < numGrids; ++i) {
    const index = i * VERTICES_PER_QUAD * 2;

    // Triangle 1 - top-left, top-right, bottom-right
    result[index] = currentx;             // top-left x
    result[index + 1] = startY;           // top-left y
    result[index + 2] = currentx + step   // top-right x
    result[index + 3] = startY;           // top-right y
    result[index + 4] = currentx + step;  // bottom-right x
    result[index + 5] = endY;             // bottom-right y

    // Triangle 2 - top-left, bottom-right, bottom-left
    result[index + 6] = currentx;         // top-left x
    result[index + 7] = startY;           // top-left y
    result[index + 8] = currentx + step;  // bottom-right x
    result[index + 9] = endY;             // bottom-right y
    result[index + 10] = currentx;        // bottom-left x
    result[index + 11] = endY;            // bottom-left y
    currentx += step;
  }
  return result;
}

/**
 * Get a buffer of vertices gridded in a rectangle, useful for drawing grids, e.g. for a histogram
 * or a lookup table / heatmap
 */
export function getGriddedRectangleBuffer(
    gl: GL, numGrids: number, startX: number = -1, endX: number = 1, startY: number = 1, endY: number = -1) {
  return getMemoizedBuffer(
             gl, WebGL2RenderingContext.ARRAY_BUFFER, createGriddedRectangleArray, numGrids, startX,
             endX, startY, endY)
      .value;
}