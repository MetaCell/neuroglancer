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

import { describe, expect, it } from "vitest";

import {
  getSpatialSkeletonMissingSelectionDisplayState,
  getSpatialSkeletonSegmentIdFromLayerSelectionState,
  getSpatialSkeletonNodeIdFromLayerSelectionState,
  getSpatialSkeletonNodeIdFromViewerHover,
  getSpatialSkeletonSelectionRecoveryKey,
  getSpatialSkeletonNodeIdFromViewerSelection,
  hasSpatialSkeletonNodeSelection,
} from "#src/layer/segmentation/selection.js";

describe("layer/segmentation/selection", () => {
  it("recognizes field-based spatial skeleton node selections", () => {
    expect(
      hasSpatialSkeletonNodeSelection({
        spatialSkeletonNodeId: "17",
        spatialSkeletonSegmentId: "9",
      }),
    ).toBe(true);
    expect(
      hasSpatialSkeletonNodeSelection({
        spatialSkeletonNodeId: "18446744073709551615",
      }),
    ).toBe(true);
    expect(
      hasSpatialSkeletonNodeSelection({
        spatialSkeletonNodeId: 17,
      }),
    ).toBe(false);
    expect(
      hasSpatialSkeletonNodeSelection({
        spatialSkeletonNodeId: 0,
      }),
    ).toBe(false);
    expect(hasSpatialSkeletonNodeSelection({})).toBe(false);
  });

  it("extracts node and segment ids from a layer selection state", () => {
    expect(
      getSpatialSkeletonNodeIdFromLayerSelectionState({
        spatialSkeletonNodeId: "23",
        spatialSkeletonSegmentId: "7",
      }),
    ).toBe(23);
    expect(
      getSpatialSkeletonSegmentIdFromLayerSelectionState({
        spatialSkeletonNodeId: "23",
        spatialSkeletonSegmentId: "7",
      }),
    ).toBe(7);
    expect(
      getSpatialSkeletonNodeIdFromLayerSelectionState({
        spatialSkeletonNodeId: -1,
      }),
    ).toBeUndefined();
    expect(
      getSpatialSkeletonSegmentIdFromLayerSelectionState({
        spatialSkeletonSegmentId: "9",
      }),
    ).toBe(9);
    expect(
      getSpatialSkeletonSelectionRecoveryKey({
        spatialSkeletonNodeId: "23",
        spatialSkeletonSegmentId: "7",
      }),
    ).toBe("23:7");
    expect(
      getSpatialSkeletonNodeIdFromLayerSelectionState({
        spatialSkeletonNodeId: "18446744073709551615",
      }),
    ).toBeUndefined();
    expect(
      getSpatialSkeletonSelectionRecoveryKey({
        spatialSkeletonNodeId: 23,
      }),
    ).toBeUndefined();
  });

  it("extracts the selected node id for the matching layer", () => {
    const layerA = {};
    const layerB = {};
    expect(
      getSpatialSkeletonNodeIdFromViewerSelection(
        {
          layers: [
            {
              layer: layerA,
              state: {},
            },
            {
              layer: layerB,
              state: {
                spatialSkeletonNodeId: "31",
                spatialSkeletonSegmentId: "8",
              },
            },
          ],
        },
        layerB,
      ),
    ).toBe(31);
    expect(
      getSpatialSkeletonNodeIdFromViewerSelection(
        {
          layers: [
            {
              layer: layerA,
              state: {
                spatialSkeletonNodeId: 4,
              },
            },
          ],
        },
        layerB,
      ),
    ).toBeUndefined();
  });

  it("extracts the hovered node id only for matching render layers", () => {
    const renderLayerA = {};
    const renderLayerB = {};
    const layer = {
      renderLayers: [renderLayerA],
    };
    expect(
      getSpatialSkeletonNodeIdFromViewerHover(
        {
          active: true,
          pickedRenderLayer: renderLayerA,
          pickedSpatialSkeletonNodeId: 31,
        },
        layer,
      ),
    ).toBe(31);
    expect(
      getSpatialSkeletonNodeIdFromViewerHover(
        {
          active: true,
          pickedRenderLayer: renderLayerB,
          pickedSpatialSkeletonNodeId: 31,
        },
        layer,
      ),
    ).toBeUndefined();
    expect(
      getSpatialSkeletonNodeIdFromViewerHover(
        {
          active: false,
          pickedRenderLayer: renderLayerA,
          pickedSpatialSkeletonNodeId: 31,
        },
        layer,
      ),
    ).toBeUndefined();
    expect(
      getSpatialSkeletonNodeIdFromViewerHover(
        {
          active: true,
          pickedRenderLayer: renderLayerA,
          pickedSpatialSkeletonNodeId: -1,
        },
        layer,
      ),
    ).toBeUndefined();
  });

  it("requests selection recovery only when a full-segment fetch can help", () => {
    expect(
      getSpatialSkeletonMissingSelectionDisplayState(
        {
          spatialSkeletonNodeId: "31",
          spatialSkeletonSegmentId: "8",
        },
        {
          hasInspectableSource: true,
          hasCachedSegment: false,
          recoveryStatus: undefined,
        },
      ),
    ).toEqual({
      recoveryKey: "31:8",
      recoveryStatus: undefined,
      shouldRequestRecovery: true,
      loading: true,
    });
    expect(
      getSpatialSkeletonMissingSelectionDisplayState(
        {
          spatialSkeletonNodeId: "31",
          spatialSkeletonSegmentId: "8",
        },
        {
          hasInspectableSource: true,
          hasCachedSegment: false,
          recoveryStatus: "pending",
        },
      ),
    ).toEqual({
      recoveryKey: "31:8",
      recoveryStatus: "pending",
      shouldRequestRecovery: false,
      loading: true,
    });
    expect(
      getSpatialSkeletonMissingSelectionDisplayState(
        {
          spatialSkeletonNodeId: "31",
          spatialSkeletonSegmentId: "8",
        },
        {
          hasInspectableSource: true,
          hasCachedSegment: true,
          recoveryStatus: undefined,
        },
      ),
    ).toEqual({
      recoveryKey: "31:8",
      recoveryStatus: undefined,
      shouldRequestRecovery: false,
      loading: false,
    });
    expect(
      getSpatialSkeletonMissingSelectionDisplayState(
        {
          spatialSkeletonNodeId: "31",
          spatialSkeletonSegmentId: "8",
        },
        {
          hasInspectableSource: true,
          hasCachedSegment: false,
          recoveryStatus: "failed",
        },
      ),
    ).toEqual({
      recoveryKey: "31:8",
      recoveryStatus: "failed",
      shouldRequestRecovery: false,
      loading: false,
    });
    expect(
      getSpatialSkeletonMissingSelectionDisplayState(
        {
          spatialSkeletonNodeId: 31,
        },
        {
          hasInspectableSource: true,
          hasCachedSegment: false,
          recoveryStatus: undefined,
        },
      ),
    ).toEqual({
      recoveryKey: undefined,
      recoveryStatus: undefined,
      shouldRequestRecovery: false,
      loading: false,
    });
  });
});
