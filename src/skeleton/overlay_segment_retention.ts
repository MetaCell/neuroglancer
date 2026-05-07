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

import type { SpatialSkeletonId } from "#src/skeleton/api.js";
import { compareUint64Ids } from "#src/util/bigint.js";

export const DEFAULT_MAX_RETAINED_OVERLAY_SEGMENTS = 4;

export function mergeSpatiallyIndexedSkeletonOverlaySegmentIds(
  activeSegmentIds: readonly SpatialSkeletonId[],
  retainedSegmentIds: readonly SpatialSkeletonId[],
) {
  const mergedSegmentIds = new Set<SpatialSkeletonId>();
  for (const segmentId of [...activeSegmentIds, ...retainedSegmentIds]) {
    if (segmentId <= 0n) continue;
    mergedSegmentIds.add(segmentId);
  }
  return [...mergedSegmentIds].sort(compareUint64Ids);
}

export function retainSpatiallyIndexedSkeletonOverlaySegment(
  retainedSegmentIds: readonly SpatialSkeletonId[],
  segmentId: SpatialSkeletonId,
  options: {
    maxRetained?: number;
  } = {},
) {
  if (segmentId <= 0n) {
    return [...retainedSegmentIds];
  }
  const nextRetainedSegmentIds = retainedSegmentIds.filter(
    (candidateSegmentId) => candidateSegmentId !== segmentId,
  );
  nextRetainedSegmentIds.push(segmentId);
  const maxRetained = Math.max(
    1,
    Math.round(options.maxRetained ?? DEFAULT_MAX_RETAINED_OVERLAY_SEGMENTS),
  );
  if (nextRetainedSegmentIds.length <= maxRetained) {
    return nextRetainedSegmentIds;
  }
  return nextRetainedSegmentIds.slice(
    nextRetainedSegmentIds.length - maxRetained,
  );
}
