import { describe, expect, it, vi } from "vitest";

import { setSpatialSkeletonMode3dToLinesAndPoints } from "#src/skeleton/edit_mode_rendering.js";
import { SkeletonRenderMode } from "#src/skeleton/render_mode.js";

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

const { SpatialSkeletonEditModeTool } = await import(
  "#src/ui/spatial_skeleton_edit_tool.js"
);

describe("spatial_skeleton_edit_tool", () => {
  it("switches 3d skeleton rendering to lines and points", () => {
    const layer = {
      displayState: {
        skeletonRenderingOptions: {
          params2d: { mode: { value: SkeletonRenderMode.LINES } },
          params3d: { mode: { value: SkeletonRenderMode.LINES } },
        },
      },
    } as any;

    setSpatialSkeletonMode3dToLinesAndPoints(layer);

    expect(
      layer.displayState.skeletonRenderingOptions.params3d.mode.value,
    ).toBe(SkeletonRenderMode.LINES_AND_POINTS);
    expect(
      layer.displayState.skeletonRenderingOptions.params2d.mode.value,
    ).toBe(SkeletonRenderMode.LINES);
  });

  it("keeps parented add-node commits overlay-first without refetching chunks", () => {
    const applyCommittedAddNode = (
      SpatialSkeletonEditModeTool.prototype as any
    ).applyCommittedAddNode as (
      this: any,
      skeletonLayer: any,
      committedNode: { treenodeId: number; skeletonId: number },
      parentNodeId: number | undefined,
      position: Float32Array,
    ) => void;
    const upsertCachedNode = vi.fn();
    const selectSpatialSkeletonNode = vi.fn();
    const markSpatialSkeletonNodeDataChanged = vi.fn();
    const ensureSegmentVisibleByNumber = vi.fn();
    const pinSegmentByNumber = vi.fn();
    const skeletonLayer = {
      retainOverlaySegment: vi.fn(),
      invalidateSourceCaches: vi.fn(),
    };
    const tool = {
      ensureSegmentVisibleByNumber,
      pinSegmentByNumber,
      layer: {
        spatialSkeletonState: {
          upsertCachedNode,
        },
        selectSpatialSkeletonNode,
        markSpatialSkeletonNodeDataChanged,
        manager: {
          root: {
            selectionState: {
              pin: {
                value: true,
              },
            },
          },
        },
      },
    };
    const position = new Float32Array([1, 2, 3]);

    applyCommittedAddNode.call(
      tool,
      skeletonLayer,
      { treenodeId: 17, skeletonId: 11 },
      5,
      position,
    );

    expect(upsertCachedNode).toHaveBeenCalledWith(
      {
        nodeId: 17,
        segmentId: 11,
        position: new Float32Array([1, 2, 3]),
        parentNodeId: 5,
      },
      { allowUncachedSegment: false },
    );
    expect(ensureSegmentVisibleByNumber).toHaveBeenCalledWith(11);
    expect(pinSegmentByNumber).toHaveBeenCalledWith(11);
    expect(selectSpatialSkeletonNode).toHaveBeenCalledWith(17, true, {
      segmentId: 11,
      position: new Float32Array([1, 2, 3]),
    });
    expect(skeletonLayer.retainOverlaySegment).toHaveBeenCalledWith(11);
    expect(markSpatialSkeletonNodeDataChanged).toHaveBeenCalledWith({
      invalidateFullSkeletonCache: false,
    });
    expect(skeletonLayer.invalidateSourceCaches).not.toHaveBeenCalled();
  });

  it("seeds root add-node commits locally without overlay retention or refetching chunks", () => {
    const applyCommittedAddNode = (
      SpatialSkeletonEditModeTool.prototype as any
    ).applyCommittedAddNode as (
      this: any,
      skeletonLayer: any,
      committedNode: { treenodeId: number; skeletonId: number },
      parentNodeId: number | undefined,
      position: Float32Array,
    ) => void;
    const upsertCachedNode = vi.fn();
    const selectSpatialSkeletonNode = vi.fn();
    const markSpatialSkeletonNodeDataChanged = vi.fn();
    const ensureSegmentVisibleByNumber = vi.fn();
    const pinSegmentByNumber = vi.fn();
    const skeletonLayer = {
      retainOverlaySegment: vi.fn(),
      invalidateSourceCaches: vi.fn(),
    };
    const tool = {
      ensureSegmentVisibleByNumber,
      pinSegmentByNumber,
      layer: {
        spatialSkeletonState: {
          upsertCachedNode,
        },
        selectSpatialSkeletonNode,
        markSpatialSkeletonNodeDataChanged,
        manager: {
          root: {
            selectionState: {
              pin: {
                value: false,
              },
            },
          },
        },
      },
    };
    const position = new Float32Array([4, 5, 6]);

    applyCommittedAddNode.call(
      tool,
      skeletonLayer,
      { treenodeId: 29, skeletonId: 13 },
      undefined,
      position,
    );

    expect(upsertCachedNode).toHaveBeenCalledWith(
      {
        nodeId: 29,
        segmentId: 13,
        position: new Float32Array([4, 5, 6]),
        parentNodeId: undefined,
      },
      { allowUncachedSegment: true },
    );
    expect(ensureSegmentVisibleByNumber).toHaveBeenCalledWith(13);
    expect(pinSegmentByNumber).toHaveBeenCalledWith(13);
    expect(selectSpatialSkeletonNode).toHaveBeenCalledWith(29, false, {
      segmentId: 13,
      position: new Float32Array([4, 5, 6]),
    });
    expect(skeletonLayer.retainOverlaySegment).not.toHaveBeenCalled();
    expect(markSpatialSkeletonNodeDataChanged).toHaveBeenCalledWith({
      invalidateFullSkeletonCache: false,
    });
    expect(skeletonLayer.invalidateSourceCaches).not.toHaveBeenCalled();
  });

  it("suppresses the deleted merge segment while keeping the surviving result selected", () => {
    const applyCommittedMerge = (SpatialSkeletonEditModeTool.prototype as any)
      .applyCommittedMerge as (
      this: any,
      skeletonLayer: any,
      firstNode: { nodeId: number; segmentId?: number },
      secondNode: { nodeId: number; segmentId?: number },
      result: {
        resultSkeletonId?: number;
        deletedSkeletonId?: number;
        stableAnnotationSwap: boolean;
      },
    ) => {
      resultSkeletonId: number | undefined;
      deletedSkeletonId: number | undefined;
    };
    const updateVisibleSkeletonSegments = vi.fn();
    const mergeCachedSegments = vi.fn();
    const selectSpatialSkeletonNode = vi.fn();
    const markSpatialSkeletonNodeDataChanged = vi.fn();
    const clearSpatialSkeletonMergeAnchor = vi.fn();
    const deleteSegmentColor = vi.fn();
    const skeletonLayer = {
      suppressBrowseSegment: vi.fn(),
      invalidateSourceCaches: vi.fn(),
    };
    const tool = {
      updateVisibleSkeletonSegments,
      layer: {
        displayState: {
          segmentStatedColors: {
            value: {
              delete: deleteSegmentColor,
            },
          },
        },
        spatialSkeletonState: {
          mergeCachedSegments,
        },
        selectSpatialSkeletonNode,
        markSpatialSkeletonNodeDataChanged,
        clearSpatialSkeletonMergeAnchor,
        manager: {
          root: {
            selectionState: {
              pin: {
                value: true,
              },
            },
          },
        },
      },
    };

    const result = applyCommittedMerge.call(
      tool,
      skeletonLayer,
      { nodeId: 101, segmentId: 11 },
      { nodeId: 202, segmentId: 17 },
      {
        resultSkeletonId: 17,
        deletedSkeletonId: 11,
        stableAnnotationSwap: true,
      },
    );

    expect(result).toEqual({
      resultSkeletonId: 17,
      deletedSkeletonId: 11,
    });
    expect(updateVisibleSkeletonSegments).toHaveBeenCalledWith(17, 11);
    expect(mergeCachedSegments).toHaveBeenCalledWith({
      resultSegmentId: 17,
      mergedSegmentId: 11,
      childNodeId: 101,
      parentNodeId: 202,
    });
    expect(selectSpatialSkeletonNode).toHaveBeenCalledWith(101, true, {
      segmentId: 17,
    });
    expect(deleteSegmentColor).toHaveBeenCalledWith(11n);
    expect(skeletonLayer.suppressBrowseSegment).toHaveBeenCalledWith(11);
    expect(markSpatialSkeletonNodeDataChanged).toHaveBeenCalledWith({
      invalidateFullSkeletonCache: false,
    });
    expect(skeletonLayer.invalidateSourceCaches).toHaveBeenCalledTimes(1);
    expect(clearSpatialSkeletonMergeAnchor).toHaveBeenCalledTimes(1);
  });
});
