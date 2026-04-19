import { describe, expect, it, vi } from "vitest";

import type { RenderLayerTransform } from "#src/render_coordinate_transform.js";
import { SpatialSkeletonActions } from "#src/skeleton/actions.js";
import { WatchableValue } from "#src/trackable_value.js";

if (!("WebGL2RenderingContext" in globalThis)) {
  Object.defineProperty(globalThis, "WebGL2RenderingContext", {
    value: new Proxy(class WebGL2RenderingContext {} as any, {
      get(target, property, receiver) {
        if (Reflect.has(target, property)) {
          return Reflect.get(target, property, receiver);
        }
        return 0;
      },
    }),
    configurable: true,
  });
}

const { SegmentationUserLayer } = await import(
  "#src/layer/segmentation/index.js"
);

const {
  PerspectiveViewSpatiallyIndexedSkeletonLayer,
  SliceViewPanelSpatiallyIndexedSkeletonLayer,
  SliceViewSpatiallyIndexedSkeletonLayer,
  MultiscaleSliceViewSpatiallyIndexedSkeletonLayer,
} = await import("#src/skeleton/frontend.js");

const { SegmentSelectionState } = await import(
  "#src/segmentation_display_state/frontend.js"
);

function makeEditableSpatialSkeletonSource(options: {
  rerootSkeleton?: (() => Promise<void>) | undefined;
} = {}) {
  return {
    listSkeletons: async () => [],
    getSkeleton: async () => [],
    fetchNodes: async () => [],
    getSpatialIndexMetadata: async () => null,
    addNode: async () => ({ treenodeId: 1, skeletonId: 1 }),
    insertNode: async () => ({ treenodeId: 1, skeletonId: 1 }),
    moveNode: async () => ({}),
    deleteNode: async () => ({}),
    updateDescription: async () => ({}),
    setTrueEnd: async () => ({}),
    removeTrueEnd: async () => ({}),
    updateRadius: async () => ({}),
    updateConfidence: async () => ({}),
    mergeSkeletons: async () => ({
      resultSkeletonId: 1,
      deletedSkeletonId: 2,
      stableAnnotationSwap: false,
    }),
    splitSkeleton: async () => ({
      existingSkeletonId: 1,
      newSkeletonId: 2,
    }),
    ...(options.rerootSkeleton === undefined
      ? {}
      : { rerootSkeleton: options.rerootSkeleton }),
  };
}

function makeSpatialSkeletonLayerWithSource(source: unknown) {
  return {
    source,
  };
}

describe("layer/segmentation spatial skeleton chunk stats", () => {
  it("tracks combined chunk load state from the loading render layers only", () => {
    const perspectiveLayer = Object.assign(
      Object.create(PerspectiveViewSpatiallyIndexedSkeletonLayer.prototype),
      {
        layerChunkProgressInfo: {
          numVisibleChunksNeeded: 5,
          numVisibleChunksAvailable: 3,
        },
      },
    );
    const sliceLayer = Object.assign(
      Object.create(SliceViewSpatiallyIndexedSkeletonLayer.prototype),
      {
        layerChunkProgressInfo: {
          numVisibleChunksNeeded: 4,
          numVisibleChunksAvailable: 2,
        },
      },
    );
    const multiscaleSliceLayer = Object.assign(
      Object.create(MultiscaleSliceViewSpatiallyIndexedSkeletonLayer.prototype),
      {
        layerChunkProgressInfo: {
          numVisibleChunksNeeded: 6,
          numVisibleChunksAvailable: 5,
        },
      },
    );
    const slicePanelLayer = Object.assign(
      Object.create(SliceViewPanelSpatiallyIndexedSkeletonLayer.prototype),
      {
        layerChunkProgressInfo: {
          numVisibleChunksNeeded: 100,
          numVisibleChunksAvailable: 100,
        },
      },
    );

    const layer = Object.assign(
      Object.create(SegmentationUserLayer.prototype),
      {
        renderLayers: [
          perspectiveLayer,
          sliceLayer,
          multiscaleSliceLayer,
          slicePanelLayer,
        ],
        spatialSkeletonVisibleChunksNeeded: new WatchableValue(0),
        spatialSkeletonVisibleChunksAvailable: new WatchableValue(0),
        spatialSkeletonVisibleChunksLoaded: new WatchableValue(false),
        displayState: {
          spatialSkeletonGridChunkStats2d: new WatchableValue({
            presentCount: 0,
            totalCount: 0,
          }),
          spatialSkeletonGridChunkStats3d: new WatchableValue({
            presentCount: 0,
            totalCount: 0,
          }),
        },
        updateSpatialSkeletonSourceState: vi.fn(),
      },
    );

    layer.updateSpatialSkeletonChunkLoadState();

    expect(layer.spatialSkeletonVisibleChunksNeeded.value).toBe(15);
    expect(layer.spatialSkeletonVisibleChunksAvailable.value).toBe(10);
    expect(layer.spatialSkeletonVisibleChunksLoaded.value).toBe(false);
  });
});

describe("layer/segmentation spatial skeleton action gating", () => {
  it("does not require max lod for skeleton actions", () => {
    const layer = Object.assign(
      Object.create(SegmentationUserLayer.prototype),
      {
        getSpatiallyIndexedSkeletonLayer: () =>
          makeSpatialSkeletonLayerWithSource(
            makeEditableSpatialSkeletonSource({
              rerootSkeleton: async () => {},
            }),
          ),
        spatialSkeletonVisibleChunksLoaded: new WatchableValue(true),
        spatialSkeletonVisibleChunksNeeded: new WatchableValue(0),
        spatialSkeletonVisibleChunksAvailable: new WatchableValue(0),
      },
    );

    expect(
      layer.getSpatialSkeletonActionsDisabledReason(
        SpatialSkeletonActions.mergeSkeletons,
      ),
    ).toBeUndefined();
    expect(
      layer.getSpatialSkeletonActionsDisabledReason(
        SpatialSkeletonActions.reroot,
        {
          requireVisibleChunks: false,
        },
      ),
    ).toBeUndefined();
    expect(
      layer.getSpatialSkeletonActionsDisabledReason([
        SpatialSkeletonActions.addNodes,
        SpatialSkeletonActions.moveNodes,
      ]),
    ).toBeUndefined();
  });

  it("still reports visible chunk loading when requested", () => {
    const layer = Object.assign(
      Object.create(SegmentationUserLayer.prototype),
      {
        getSpatiallyIndexedSkeletonLayer: () =>
          makeSpatialSkeletonLayerWithSource(makeEditableSpatialSkeletonSource()),
        spatialSkeletonVisibleChunksLoaded: new WatchableValue(false),
        spatialSkeletonVisibleChunksNeeded: new WatchableValue(3),
        spatialSkeletonVisibleChunksAvailable: new WatchableValue(1),
      },
    );

    expect(
      layer.getSpatialSkeletonActionsDisabledReason(
        SpatialSkeletonActions.splitSkeletons,
        {
          requireVisibleChunks: true,
        },
      ),
    ).toBe("Wait for visible skeleton chunks to load (1/3).");
  });

  it("reports missing reroot support explicitly", () => {
    const layer = Object.assign(
      Object.create(SegmentationUserLayer.prototype),
      {
        getSpatiallyIndexedSkeletonLayer: () =>
          makeSpatialSkeletonLayerWithSource(makeEditableSpatialSkeletonSource()),
        spatialSkeletonVisibleChunksLoaded: new WatchableValue(true),
        spatialSkeletonVisibleChunksNeeded: new WatchableValue(0),
        spatialSkeletonVisibleChunksAvailable: new WatchableValue(0),
      },
    );

    expect(
      layer.getSpatialSkeletonActionsDisabledReason(
        SpatialSkeletonActions.reroot,
        {
          requireVisibleChunks: false,
        },
      ),
    ).toBe(
      "The active spatial skeleton source does not support skeleton rerooting.",
    );
  });
});

describe("layer/segmentation spatial skeleton selection serialization", () => {
  it("accepts bigint segment selections for runtime spatial skeleton state", () => {
    const selectionState = new SegmentSelectionState();

    selectionState.set(7n);

    expect(selectionState.value).toBe(7n);
    expect(selectionState.baseValue).toBe(7n);
  });

  it("round-trips spatial skeleton selection ids as canonical strings", () => {
    const layer = Object.create(SegmentationUserLayer.prototype);
    Object.defineProperty(layer, "localCoordinateSpace", {
      value: { value: { rank: 0 } },
      configurable: true,
    });
    const state: any = {};
    layer.initializeSelectionState(state);

    layer.selectionStateFromJson(state, {
      spatialSkeletonNodeId: "23",
      spatialSkeletonSegmentId: "7",
      value: "7",
    });

    expect(state.spatialSkeletonNodeId).toBe("23");
    expect(state.spatialSkeletonSegmentId).toBe("7");
    expect(state.value).toBe(7n);
    expect(layer.selectionStateToJson(state, false)).toEqual({
      spatialSkeletonNodeId: "23",
      spatialSkeletonSegmentId: "7",
      value: "7",
    });
  });
});

describe("layer/segmentation spatial skeleton node navigation helpers", () => {
  it("maps model-space node positions through non-identity transforms before updating view state", () => {
    const dispatchGlobalPositionChanged = vi.fn();
    const dispatchLocalPositionChanged = vi.fn();
    const transform: RenderLayerTransform = {
      rank: 3,
      unpaddedRank: 3,
      localToRenderLayerDimensions: [1, -1, 2],
      globalToRenderLayerDimensions: [2, 0, 1, -1],
      channelToRenderLayerDimensions: [],
      channelToModelDimensions: [],
      channelSpaceShape: new Uint32Array(0),
      modelToRenderLayerTransform: new Float32Array([
        2, 0, 0, 0,
        0, 0, 1, 0,
        0, 3, 0, 0,
        10, -5, 1, 1,
      ]),
      modelDimensionNames: ["x", "y", "z"],
      layerDimensionNames: ["a", "b", "c"],
    };
    const layer = Object.create(SegmentationUserLayer.prototype);
    Object.assign(layer, {
      getSpatiallyIndexedSkeletonLayer: () => ({
        displayState: {
          transform: {
            value: transform,
          },
        },
      }),
    });
    Object.defineProperty(layer, "manager", {
      value: {
        root: {
          globalPosition: {
            value: new Float32Array([100, 101, 102, 103]),
            changed: {
              dispatch: dispatchGlobalPositionChanged,
            },
          },
        },
      },
      configurable: true,
    });
    Object.defineProperty(layer, "localPosition", {
      value: {
        value: new Float32Array([200, 201, 202]),
        changed: {
          dispatch: dispatchLocalPositionChanged,
        },
      },
      configurable: true,
    });

    layer.moveViewToSpatialSkeletonNodePosition([4, 5, 6]);

    expect(Array.from(layer.manager.root.globalPosition.value)).toEqual([
      6, 18, 13, 103,
    ]);
    expect(Array.from(layer.localPosition.value)).toEqual([13, 201, 6]);
    expect(dispatchLocalPositionChanged).toHaveBeenCalledTimes(1);
    expect(dispatchGlobalPositionChanged).toHaveBeenCalledTimes(1);
  });

  it("selects and moves to the provided node, or clears selection when absent", () => {
    const selectSpatialSkeletonNode = vi.fn();
    const moveViewToSpatialSkeletonNodePosition = vi.fn();
    const clearSpatialSkeletonNodeSelection = vi.fn();
    const layer = Object.assign(
      Object.create(SegmentationUserLayer.prototype),
      {
        selectSpatialSkeletonNode,
        moveViewToSpatialSkeletonNodePosition,
        clearSpatialSkeletonNodeSelection,
      },
    );
    Object.defineProperty(layer, "manager", {
      value: {
        root: {
          selectionState: {
            pin: {
              value: true,
            },
          },
        },
      },
      configurable: true,
    });
    const node = {
      nodeId: 31,
      segmentId: 9,
      position: new Float32Array([4, 5, 6]),
    };

    expect(layer.selectAndMoveToSpatialSkeletonNode(node)).toBe(true);
    expect(selectSpatialSkeletonNode).toHaveBeenCalledWith(31, true, {
      segmentId: 9,
      position: new Float32Array([4, 5, 6]),
    });
    expect(moveViewToSpatialSkeletonNodePosition).toHaveBeenCalledWith(
      node.position,
    );
    expect(clearSpatialSkeletonNodeSelection).not.toHaveBeenCalled();

    expect(layer.selectAndMoveToSpatialSkeletonNode(undefined, false)).toBe(
      false,
    );
    expect(clearSpatialSkeletonNodeSelection).toHaveBeenCalledWith(false);
  });
});
