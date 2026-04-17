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

export const DEFAULT_MAX_RETAINED_OVERLAY_SEGMENTS = 4;

function normalizeSegmentId(segmentId: number) {
  const normalizedSegmentId = Math.round(Number(segmentId));
  if (!Number.isSafeInteger(normalizedSegmentId) || normalizedSegmentId <= 0) {
    return undefined;
  }
  return normalizedSegmentId;
}

export function mergeSpatiallyIndexedSkeletonOverlaySegmentIds(
  activeSegmentIds: readonly number[],
  retainedSegmentIds: readonly number[],
) {
  const mergedSegmentIds = new Set<number>();
  for (const segmentId of [...activeSegmentIds, ...retainedSegmentIds]) {
    const normalizedSegmentId = normalizeSegmentId(segmentId);
    if (normalizedSegmentId === undefined) continue;
    mergedSegmentIds.add(normalizedSegmentId);
  }
  return [...mergedSegmentIds].sort((a, b) => a - b);
}

export function retainSpatiallyIndexedSkeletonOverlaySegment(
  retainedSegmentIds: readonly number[],
  segmentId: number,
  options: {
    maxRetained?: number;
  } = {},
) {
  const normalizedSegmentId = normalizeSegmentId(segmentId);
  if (normalizedSegmentId === undefined) {
    return [...retainedSegmentIds];
  }
  const nextRetainedSegmentIds = retainedSegmentIds.filter(
    (candidateSegmentId) => candidateSegmentId !== normalizedSegmentId,
  );
  nextRetainedSegmentIds.push(normalizedSegmentId);
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
