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

/**
 * @file Console/UI entry points for skeleton summary statistics.
 */

import type { SegmentationUserLayer } from "#src/layer/segmentation/index.js";
import { getSegmentIdFromLayerSelectionValue } from "#src/layer/segmentation/selection.js";
import type { SpatiallyIndexedSkeletonNode } from "#src/skeleton/api.js";
import type { SkeletonStatistics } from "#src/skeleton/skeleton_statistics.js";
import {
  computeSkeletonStatistics,
  formatSkeletonStatistics,
} from "#src/skeleton/skeleton_statistics.js";

/**
 * The first segmentation layer that exposes a spatially indexed skeleton.
 * Viewer states used with this tool carry at most one such layer.
 */
export function findSpatialSkeletonLayer(
  viewer: any,
): SegmentationUserLayer | undefined {
  for (const managedLayer of viewer?.layerManager?.managedLayers ?? []) {
    const userLayer = managedLayer.layer;
    if (
      userLayer?.spatialSkeletonState !== undefined &&
      typeof userLayer.getSpatiallyIndexedSkeletonLayer === "function"
    ) {
      return userLayer as SegmentationUserLayer;
    }
  }
  return undefined;
}

export function getSelectedSegmentIdForLayer(
  layer: SegmentationUserLayer,
): number | undefined {
  const layerSelectionState =
    layer.manager.root.selectionState.value?.layers.find(
      (entry) => entry.layer === layer,
    )?.state;
  return getSegmentIdFromLayerSelectionValue(layerSelectionState);
}

/** Cached nodes when present, otherwise a full-skeleton fetch. */
export async function getSkeletonNodesForStatistics(
  layer: SegmentationUserLayer,
  segmentId: number,
): Promise<SpatiallyIndexedSkeletonNode[]> {
  const { spatialSkeletonState } = layer;
  const cachedNodes = spatialSkeletonState.getCachedSegmentNodes(segmentId);
  if (cachedNodes !== undefined) return [...cachedNodes];
  const skeletonLayer = layer.getSpatiallyIndexedSkeletonLayer();
  if (skeletonLayer === undefined) {
    throw new Error("No spatially indexed skeleton layer is available.");
  }
  return await spatialSkeletonState.getFullSegmentNodes(
    skeletonLayer,
    segmentId,
  );
}

export async function computeSkeletonStatisticsForSegment(
  layer: SegmentationUserLayer,
  segmentId: number,
): Promise<SkeletonStatistics> {
  const nodes = await getSkeletonNodesForStatistics(layer, segmentId);
  return computeSkeletonStatistics(nodes);
}

/**
 * Prints statistics for `segmentId`, or for the currently selected segment
 * when it is omitted.  Returns the statistics object for further inspection.
 */
export async function printSkeletonStatistics(
  viewer: any,
  segmentId?: number | string,
): Promise<SkeletonStatistics | undefined> {
  const layer = findSpatialSkeletonLayer(viewer);
  if (layer === undefined) {
    console.warn("No segmentation layer with a spatial skeleton source found.");
    return undefined;
  }
  const resolvedSegmentId =
    segmentId === undefined
      ? getSelectedSegmentIdForLayer(layer)
      : Number(segmentId);
  if (
    resolvedSegmentId === undefined ||
    !Number.isSafeInteger(resolvedSegmentId) ||
    resolvedSegmentId <= 0
  ) {
    console.warn(
      "No skeleton id given and no segment is selected; pass an id explicitly.",
    );
    return undefined;
  }
  const statistics = await computeSkeletonStatisticsForSegment(
    layer,
    resolvedSegmentId,
  );
  console.log(
    formatSkeletonStatistics({
      ...statistics,
      segmentId: statistics.segmentId ?? resolvedSegmentId,
    }),
  );
  return statistics;
}

export function registerSkeletonStatisticsDebugApi(viewer: any) {
  viewer.printSkeletonStatistics = (segmentId?: number | string) =>
    printSkeletonStatistics(viewer, segmentId);
}
