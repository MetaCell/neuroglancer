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

import type { ProjectionParameters } from "#src/projection_parameters.js";
import { vec4 } from "#src/util/geom.js";

export interface ViewportPoint {
  /** Logical CSS pixels from the left edge of the viewport. */
  readonly x: number;
  /** Logical CSS pixels from the top edge of the viewport. */
  readonly y: number;
  /** Normalized device depth in [-1, 1]. */
  readonly ndcZ: number;
  /**
   * Apparent size relative to the focal plane: 1 at the focal plane, larger
   * nearer the camera, smaller farther away, and always 1 under orthographic
   * projection.
   */
  readonly depthScale: number;
}

const tempClip = vec4.create();

/**
 * Projects `position` (in the coordinate space of `parameters`) to viewport
 * pixels.  Returns `undefined` when the point is behind the camera or outside
 * the clip volume.
 */
export function projectToViewport(
  parameters: ProjectionParameters,
  position: ArrayLike<number>,
): ViewportPoint | undefined {
  const {
    viewProjectionMat,
    logicalWidth,
    logicalHeight,
    displayDimensionRenderInfo: { displayDimensionIndices },
  } = parameters;
  const clip = tempClip;
  for (let i = 0; i < 3; ++i) {
    const index = displayDimensionIndices[i];
    clip[i] = index >= 0 ? position[index] : 0;
  }
  clip[3] = 1;
  vec4.transformMat4(clip, clip, viewProjectionMat);
  const w = clip[3];
  if (w <= 0) return undefined;
  const ndcZ = clip[2] / w;
  if (ndcZ < -1 || ndcZ > 1) return undefined;
  return {
    x: ((clip[0] / w) * 0.5 + 0.5) * logicalWidth,
    y: (0.5 - (clip[1] / w) * 0.5) * logicalHeight,
    ndcZ,
    // The view matrices place the focal point at clip-space w = 1 under both
    // perspective and orthographic projection.
    depthScale: 1 / w,
  };
}
