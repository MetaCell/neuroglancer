import { describe, expect, it, vi } from "vitest";

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

const {
  SegmentationUserLayer,
} = await import("#src/layer/segmentation/index.js");

const {
  PerspectiveViewSpatiallyIndexedSkeletonLayer,
  SliceViewPanelSpatiallyIndexedSkeletonLayer,
  SliceViewSpatiallyIndexedSkeletonLayer,
  MultiscaleSliceViewSpatiallyIndexedSkeletonLayer,
} = await import("#src/skeleton/frontend.js");

const {
  SegmentSelectionState,
} = await import("#src/segmentation_display_state/frontend.js");

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
      Object.create(
        MultiscaleSliceViewSpatiallyIndexedSkeletonLayer.prototype,
      ),
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

    const layer = Object.assign(Object.create(SegmentationUserLayer.prototype), {
      renderLayers: [
        perspectiveLayer,
        sliceLayer,
        multiscaleSliceLayer,
        slicePanelLayer,
      ],
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
      spatialSkeletonState: {
        updateChunkLoadState: vi.fn(),
      },
      updateSpatialSkeletonSourceState: vi.fn(),
    });

    layer.updateSpatialSkeletonChunkLoadState();

    expect(layer.spatialSkeletonState.updateChunkLoadState).toHaveBeenCalledWith(
      15,
      10,
    );
  });
});

describe("layer/segmentation spatial skeleton action gating", () => {
  it("does not require max lod for skeleton actions", () => {
    const layer = Object.assign(Object.create(SegmentationUserLayer.prototype), {
      spatialSkeletonSourceCapabilities: new WatchableValue({
        inspectSkeletons: true,
        addNodes: true,
        moveNodes: true,
        deleteNodes: true,
        rerootSkeletons: true,
        editNodeLabels: true,
        editNodeProperties: true,
        mergeSkeletons: true,
        splitSkeletons: true,
      }),
      spatialSkeletonEditModeAllowed: new WatchableValue(false),
      spatialSkeletonVisibleChunksLoaded: new WatchableValue(true),
      spatialSkeletonVisibleChunksNeeded: new WatchableValue(0),
      spatialSkeletonVisibleChunksAvailable: new WatchableValue(0),
    });

    expect(
      layer.getSpatialSkeletonActionsDisabledReason("mergeSkeletons"),
    ).toBeUndefined();
    expect(
      layer.getSpatialSkeletonActionsDisabledReason("rerootSkeletons", {
        requireVisibleChunks: false,
      }),
    ).toBeUndefined();
    expect(
      layer.getSpatialSkeletonActionsDisabledReason(["addNodes", "moveNodes"]),
    ).toBeUndefined();
  });

  it("still reports visible chunk loading when requested", () => {
    const layer = Object.assign(Object.create(SegmentationUserLayer.prototype), {
      spatialSkeletonSourceCapabilities: new WatchableValue({
        inspectSkeletons: true,
        addNodes: false,
        moveNodes: false,
        deleteNodes: false,
        rerootSkeletons: false,
        editNodeLabels: false,
        editNodeProperties: false,
        mergeSkeletons: false,
        splitSkeletons: true,
      }),
      spatialSkeletonVisibleChunksLoaded: new WatchableValue(false),
      spatialSkeletonVisibleChunksNeeded: new WatchableValue(3),
      spatialSkeletonVisibleChunksAvailable: new WatchableValue(1),
    });

    expect(
      layer.getSpatialSkeletonActionsDisabledReason("splitSkeletons", {
        requireVisibleChunks: true,
      }),
    ).toBe("Wait for visible skeleton chunks to load (1/3).");
  });

  it("reports missing reroot support explicitly", () => {
    const layer = Object.assign(Object.create(SegmentationUserLayer.prototype), {
      spatialSkeletonSourceCapabilities: new WatchableValue({
        inspectSkeletons: true,
        addNodes: true,
        moveNodes: true,
        deleteNodes: true,
        rerootSkeletons: false,
        editNodeLabels: true,
        editNodeProperties: true,
        mergeSkeletons: true,
        splitSkeletons: true,
      }),
      spatialSkeletonVisibleChunksLoaded: new WatchableValue(true),
      spatialSkeletonVisibleChunksNeeded: new WatchableValue(0),
      spatialSkeletonVisibleChunksAvailable: new WatchableValue(0),
    });

    expect(
      layer.getSpatialSkeletonActionsDisabledReason("rerootSkeletons", {
        requireVisibleChunks: false,
      }),
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
  it("selects and moves to the provided node, or clears selection when absent", () => {
    const selectSpatialSkeletonNode = vi.fn();
    const moveViewToSpatialSkeletonNodePosition = vi.fn();
    const clearSpatialSkeletonNodeSelection = vi.fn();
    const layer = Object.assign(Object.create(SegmentationUserLayer.prototype), {
      selectSpatialSkeletonNode,
      moveViewToSpatialSkeletonNodePosition,
      clearSpatialSkeletonNodeSelection,
    });
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
