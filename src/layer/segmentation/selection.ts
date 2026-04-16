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

import { parseUint64 } from "#src/util/json.js";

interface SpatialSkeletonSelectionStateLike {
  spatialSkeletonNodeId?: unknown;
  spatialSkeletonSegmentId?: unknown;
}

interface SpatialSkeletonViewerHoverMouseStateLike<TRenderLayer> {
  active: boolean;
  pickedRenderLayer: TRenderLayer | null | undefined;
  pickedSpatialSkeletonNodeId?: unknown;
}

interface SpatialSkeletonViewerHoverLayerLike<TRenderLayer> {
  renderLayers: readonly TRenderLayer[];
}

export enum SpatialSkeletonSelectionRecoveryStatus {
  PENDING = "pending",
  FAILED = "failed",
}

const MAX_SAFE_INTEGER_BIGINT = BigInt(Number.MAX_SAFE_INTEGER);

function parseSpatialSkeletonSelectionStateId(value: unknown) {
  if (typeof value !== "string") {
    return undefined;
  }
  try {
    const parsedValue = parseUint64(value);
    return parsedValue > 0n ? parsedValue : undefined;
  } catch {
    return undefined;
  }
}

function normalizeSpatialSkeletonSelectionStateId(value: unknown) {
  const parsedValue = parseSpatialSkeletonSelectionStateId(value);
  if (parsedValue === undefined || parsedValue > MAX_SAFE_INTEGER_BIGINT) {
    return undefined;
  }
  return Number(parsedValue);
}

function getSpatialSkeletonSelectionIdString(value: unknown) {
  return parseSpatialSkeletonSelectionStateId(value)?.toString();
}

function normalizeSpatialSkeletonViewerHoverNodeId(value: unknown) {
  return typeof value === "number" && Number.isSafeInteger(value) && value > 0
    ? value
    : undefined;
}

export function getSpatialSkeletonNodeIdFromLayerSelectionState(
  state: SpatialSkeletonSelectionStateLike | undefined,
) {
  return normalizeSpatialSkeletonSelectionStateId(state?.spatialSkeletonNodeId);
}

export function getSpatialSkeletonSegmentIdFromLayerSelectionState(
  state: SpatialSkeletonSelectionStateLike | undefined,
) {
  return normalizeSpatialSkeletonSelectionStateId(
    state?.spatialSkeletonSegmentId,
  );
}

export function getSpatialSkeletonSelectionRecoveryKey(
  state: SpatialSkeletonSelectionStateLike | undefined,
) {
  const nodeId = getSpatialSkeletonSelectionIdString(
    state?.spatialSkeletonNodeId,
  );
  const segmentId = getSpatialSkeletonSelectionIdString(
    state?.spatialSkeletonSegmentId,
  );
  if (nodeId === undefined || segmentId === undefined) {
    return undefined;
  }
  return `${nodeId}:${segmentId}`;
}

export function getSpatialSkeletonMissingSelectionDisplayState(
  state: SpatialSkeletonSelectionStateLike | undefined,
  options: {
    hasInspectableSource: boolean;
    hasCachedSegment: boolean;
    recoveryStatus: SpatialSkeletonSelectionRecoveryStatus | undefined;
  },
) {
  const recoveryKey = getSpatialSkeletonSelectionRecoveryKey(state);
  if (recoveryKey === undefined) {
    return {
      recoveryKey,
      recoveryStatus: undefined,
      shouldRequestRecovery: false,
      loading: false,
    };
  }
  const { hasInspectableSource, hasCachedSegment, recoveryStatus } = options;
  if (recoveryStatus === SpatialSkeletonSelectionRecoveryStatus.PENDING) {
    return {
      recoveryKey,
      recoveryStatus,
      shouldRequestRecovery: false,
      loading: true,
    };
  }
  const shouldRequestRecovery =
    !hasCachedSegment && hasInspectableSource && recoveryStatus === undefined;
  return {
    recoveryKey,
    recoveryStatus,
    shouldRequestRecovery,
    loading: shouldRequestRecovery,
  };
}

export function hasSpatialSkeletonNodeSelection(
  state: SpatialSkeletonSelectionStateLike | undefined,
) {
  return (
    getSpatialSkeletonSelectionIdString(state?.spatialSkeletonNodeId) !==
    undefined
  );
}

export function getSpatialSkeletonNodeIdFromViewerSelection<TLayer>(
  selection:
    | {
        layers: readonly {
          layer: TLayer;
          state: SpatialSkeletonSelectionStateLike;
        }[];
      }
    | undefined,
  layer: TLayer,
) {
  return getSpatialSkeletonNodeIdFromLayerSelectionState(
    selection?.layers.find((entry) => entry.layer === layer)?.state,
  );
}

export function getSpatialSkeletonNodeIdFromViewerHover<TRenderLayer>(
  mouseState: SpatialSkeletonViewerHoverMouseStateLike<TRenderLayer>,
  layer: SpatialSkeletonViewerHoverLayerLike<TRenderLayer>,
) {
  if (!mouseState.active) return undefined;
  const pickedRenderLayer = mouseState.pickedRenderLayer;
  if (pickedRenderLayer !== null) {
    if (
      pickedRenderLayer === undefined ||
      !layer.renderLayers.includes(pickedRenderLayer)
    ) {
      return undefined;
    }
  }
  // TODO (SKM): I think we can inline this function
  return normalizeSpatialSkeletonViewerHoverNodeId(
    mouseState.pickedSpatialSkeletonNodeId,
  );
}
