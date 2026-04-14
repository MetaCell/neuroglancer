import { afterEach, describe, expect, it, vi } from "vitest";

import {
  executeSpatialSkeletonSplit,
  undoSpatialSkeletonCommand,
} from "#src/layer/segmentation/spatial_skeleton_commands.js";
import { SpatialSkeletonCommandHistory } from "#src/skeleton/command_history.js";
import type { SpatiallyIndexedSkeletonNodeInfo } from "#src/skeleton/frontend.js";
import { StatusMessage } from "#src/status.js";

function cloneNode(
  node: SpatiallyIndexedSkeletonNodeInfo,
): SpatiallyIndexedSkeletonNodeInfo {
  return {
    ...node,
    position: new Float32Array(node.position),
    labels: node.labels === undefined ? undefined : [...node.labels],
  };
}

function cloneNodes(
  nodes: readonly SpatiallyIndexedSkeletonNodeInfo[] | undefined,
): SpatiallyIndexedSkeletonNodeInfo[] {
  return (nodes ?? []).map((node) => cloneNode(node));
}

function setSegmentNodes(
  cacheBySegment: Map<number, SpatiallyIndexedSkeletonNodeInfo[]>,
  cacheByNode: Map<number, SpatiallyIndexedSkeletonNodeInfo>,
  segmentId: number,
  nodes: readonly SpatiallyIndexedSkeletonNodeInfo[],
) {
  if (nodes.length === 0) {
    cacheBySegment.delete(segmentId);
  } else {
    cacheBySegment.set(segmentId, cloneNodes(nodes));
  }
  cacheByNode.clear();
  for (const segmentNodes of cacheBySegment.values()) {
    for (const node of segmentNodes) {
      cacheByNode.set(node.nodeId, node);
    }
  }
}

describe("spatial_skeleton_commands", () => {
  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("suppresses and clears the deleted segment when undoing a split", async () => {
    vi.spyOn(StatusMessage, "showTemporaryMessage").mockImplementation((() => ({
      dispose() {},
    })) as typeof StatusMessage.showTemporaryMessage);

    const originalSegmentId = 2973964;
    const splitSegmentId = 2973946;
    const formerParentNode: SpatiallyIndexedSkeletonNodeInfo = {
      nodeId: 21893039,
      segmentId: originalSegmentId,
      position: new Float32Array([10, 20, 30]),
      revisionToken: "parent-before",
    };
    const splitNodeBefore: SpatiallyIndexedSkeletonNodeInfo = {
      nodeId: 21893038,
      segmentId: originalSegmentId,
      parentNodeId: formerParentNode.nodeId,
      position: new Float32Array([11, 21, 31]),
      revisionToken: "split-before",
    };
    const splitNodeAfter: SpatiallyIndexedSkeletonNodeInfo = {
      ...splitNodeBefore,
      segmentId: splitSegmentId,
      parentNodeId: undefined,
      revisionToken: "split-after",
    };
    const splitNodeMergedBack: SpatiallyIndexedSkeletonNodeInfo = {
      ...splitNodeBefore,
      revisionToken: "split-merged-back",
    };

    const serverSegments = new Map<
      number,
      SpatiallyIndexedSkeletonNodeInfo[]
    >();
    const cacheBySegment = new Map<
      number,
      SpatiallyIndexedSkeletonNodeInfo[]
    >();
    const cacheByNode = new Map<number, SpatiallyIndexedSkeletonNodeInfo>();

    const syncCacheFromServer = (segmentId: number) => {
      setSegmentNodes(
        cacheBySegment,
        cacheByNode,
        segmentId,
        serverSegments.get(segmentId) ?? [],
      );
      return cacheBySegment.get(segmentId) ?? [];
    };

    serverSegments.set(originalSegmentId, [
      cloneNode(formerParentNode),
      cloneNode(splitNodeBefore),
    ]);
    syncCacheFromServer(originalSegmentId);

    const skeletonSource = {
      getSkeleton: vi.fn(),
      addNode: vi.fn(),
      insertNode: vi.fn(),
      moveNode: vi.fn(),
      deleteNode: vi.fn(),
      rerootSkeleton: vi.fn(),
      updateDescription: vi.fn(),
      setTrueEnd: vi.fn(),
      removeTrueEnd: vi.fn(),
      updateRadius: vi.fn(),
      updateConfidence: vi.fn(),
      splitSkeleton: vi.fn(async () => {
        serverSegments.set(originalSegmentId, [cloneNode(formerParentNode)]);
        serverSegments.set(splitSegmentId, [cloneNode(splitNodeAfter)]);
        return {
          existingSkeletonId: originalSegmentId,
          newSkeletonId: splitSegmentId,
        };
      }),
      mergeSkeletons: vi.fn(async () => {
        serverSegments.set(originalSegmentId, [
          cloneNode(formerParentNode),
          cloneNode(splitNodeMergedBack),
        ]);
        serverSegments.delete(splitSegmentId);
        return {
          resultSkeletonId: originalSegmentId,
          deletedSkeletonId: splitSegmentId,
          stableAnnotationSwap: false,
        };
      }),
    };

    const deleteSegmentColor = vi.fn();
    const invalidateCachedSegments = vi.fn((segmentIds: Iterable<number>) => {
      for (const segmentId of segmentIds) {
        setSegmentNodes(cacheBySegment, cacheByNode, segmentId, []);
      }
    });
    const getFullSegmentNodes = vi.fn(
      async (_skeletonLayer: unknown, segmentId: number) =>
        syncCacheFromServer(segmentId),
    );
    const skeletonLayer = {
      source: skeletonSource,
      getNode: vi.fn((nodeId: number) => cacheByNode.get(nodeId)),
      invalidateSourceCaches: vi.fn(),
      suppressBrowseSegment: vi.fn(),
    };
    const layer = {
      displayState: {
        segmentationGroupState: {
          value: {
            visibleSegments: new Set<bigint>([BigInt(originalSegmentId)]),
            selectedSegments: new Set<bigint>(),
            segmentEquivalences: {},
            temporaryVisibleSegments: new Set<bigint>(),
            temporarySegmentEquivalences: {},
            useTemporaryVisibleSegments: { value: false },
            useTemporarySegmentEquivalences: { value: false },
          },
        },
        segmentStatedColors: {
          value: {
            delete: deleteSegmentColor,
          },
        },
      },
      manager: {
        root: {
          selectionState: {
            pin: {
              value: true,
            },
          },
        },
      },
      spatialSkeletonState: {
        commandHistory: new SpatialSkeletonCommandHistory(),
        getCachedNode: (nodeId: number) => cacheByNode.get(nodeId),
        getCachedSegmentNodes: (segmentId: number) =>
          cacheBySegment.get(segmentId),
        getFullSegmentNodes,
        invalidateCachedSegments,
      },
      getSpatiallyIndexedSkeletonLayer: () => skeletonLayer,
      getCachedSpatialSkeletonSegmentNodesForEdit: (segmentId: number) =>
        cacheBySegment.get(segmentId) ?? [],
      selectSegment: vi.fn(),
      selectSpatialSkeletonNode: vi.fn(),
      markSpatialSkeletonNodeDataChanged: vi.fn(),
    };

    await executeSpatialSkeletonSplit(layer as any, {
      nodeId: splitNodeBefore.nodeId,
      segmentId: originalSegmentId,
    });

    skeletonLayer.suppressBrowseSegment.mockClear();
    deleteSegmentColor.mockClear();
    layer.selectSpatialSkeletonNode.mockClear();
    layer.markSpatialSkeletonNodeDataChanged.mockClear();
    skeletonLayer.invalidateSourceCaches.mockClear();
    invalidateCachedSegments.mockClear();
    getFullSegmentNodes.mockClear();

    await undoSpatialSkeletonCommand(layer as any);

    expect(skeletonSource.mergeSkeletons).toHaveBeenCalledWith(
      splitNodeBefore.nodeId,
      formerParentNode.nodeId,
      expect.any(Object),
    );
    expect(deleteSegmentColor).toHaveBeenCalledWith(BigInt(splitSegmentId));
    expect(skeletonLayer.suppressBrowseSegment).toHaveBeenCalledWith(
      splitSegmentId,
    );
    expect(layer.selectSpatialSkeletonNode).toHaveBeenCalledWith(
      splitNodeBefore.nodeId,
      true,
      { segmentId: originalSegmentId },
    );
    expect(invalidateCachedSegments).toHaveBeenCalledWith([
      originalSegmentId,
      splitSegmentId,
    ]);
    expect(getFullSegmentNodes).toHaveBeenCalledTimes(2);
    expect(
      layer.displayState.segmentationGroupState.value.visibleSegments.has(
        BigInt(originalSegmentId),
      ),
    ).toBe(true);
    expect(
      layer.displayState.segmentationGroupState.value.visibleSegments.has(
        BigInt(splitSegmentId),
      ),
    ).toBe(false);
    expect(cacheBySegment.get(splitSegmentId)).toBeUndefined();
    expect(
      cacheBySegment.get(originalSegmentId)?.map((node) => node.nodeId),
    ).toEqual([formerParentNode.nodeId, splitNodeBefore.nodeId]);
  });
});
