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

import { describe, expect, it, vi } from "vitest";

import { SpatialSkeletonActions } from "#src/skeleton/actions.js";
import {
  buildSpatiallyIndexedSkeletonNavigationGraph,
  getFlatListNodeIds,
  getSkeletonRootNode,
} from "#src/skeleton/navigation.js";
import {
  getEditableSpatiallyIndexedSkeletonSource,
  isSpatiallyIndexedSkeletonSourceReadOnly,
  SpatialSkeletonState,
} from "#src/skeleton/spatial_skeleton_manager.js";

function makeCommandFactory(action: string) {
  return {
    action,
    createCommand: vi.fn(),
  };
}

function makeEditableSourceCommands() {
  return {
    addNodesCommand: makeCommandFactory(SpatialSkeletonActions.addNodes),
    deleteNodesCommand: makeCommandFactory(SpatialSkeletonActions.deleteNodes),
    moveNodesCommand: makeCommandFactory(SpatialSkeletonActions.moveNodes),
    splitSkeletonsCommand: makeCommandFactory(
      SpatialSkeletonActions.splitSkeletons,
    ),
    mergeSkeletonsCommand: makeCommandFactory(
      SpatialSkeletonActions.mergeSkeletons,
    ),
  };
}

describe("skeleton/spatial_skeleton_manager", () => {
  it("returns an editable source when mandatory edit actions are present", () => {
    const source = {
      ...makeEditableSourceCommands(),
      readonly: false,
      listSkeletons: async () => [],
      getSkeleton: async () => [],
      fetchNodes: async () => [],
      getSpatialIndexMetadata: async () => null,
    };

    expect(getEditableSpatiallyIndexedSkeletonSource({ source })).toBe(source);
  });

  it("does not treat a source missing mandatory edit actions as editable", () => {
    const source = {
      ...makeEditableSourceCommands(),
      mergeSkeletonsCommand: undefined,
      readonly: false,
      listSkeletons: async () => [],
      getSkeleton: async () => [],
      fetchNodes: async () => [],
      getSpatialIndexMetadata: async () => null,
    };

    expect(
      getEditableSpatiallyIndexedSkeletonSource({ source }),
    ).toBeUndefined();
  });

  it("does not treat a command factory for the wrong action as editable", () => {
    const source = {
      ...makeEditableSourceCommands(),
      moveNodesCommand: makeCommandFactory(SpatialSkeletonActions.addNodes),
      readonly: false,
      listSkeletons: async () => [],
      getSkeleton: async () => [],
      fetchNodes: async () => [],
      getSpatialIndexMetadata: async () => null,
    };

    expect(
      getEditableSpatiallyIndexedSkeletonSource({ source }),
    ).toBeUndefined();
  });

  it("does not require optional edit actions for editable source validation", () => {
    const source = {
      ...makeEditableSourceCommands(),
      readonly: false,
      listSkeletons: async () => [],
      getSkeleton: async () => [],
      fetchNodes: async () => [],
      getSpatialIndexMetadata: async () => null,
    };

    expect(getEditableSpatiallyIndexedSkeletonSource({ source })).toBe(source);
  });

  it("validates optional confidence configuration for editable sources", () => {
    const source = {
      ...makeEditableSourceCommands(),
      editNodeConfidenceCommand: makeCommandFactory(
        SpatialSkeletonActions.editNodeConfidence,
      ),
      spatialSkeletonConfidenceConfiguration: {
        values: [0, 50, 100],
      },
      readonly: false,
      listSkeletons: async () => [],
      getSkeleton: async () => [],
      fetchNodes: async () => [],
      getSpatialIndexMetadata: async () => null,
    };

    expect(getEditableSpatiallyIndexedSkeletonSource({ source })).toBe(source);

    expect(
      getEditableSpatiallyIndexedSkeletonSource({
        source: {
          ...source,
          spatialSkeletonConfidenceConfiguration: {
            values: [0, Number.NaN, 100],
          },
        },
      }),
    ).toBeUndefined();
  });

  it("does not treat a read-only source with edit commands as editable", () => {
    const source = {
      ...makeEditableSourceCommands(),
      readonly: true,
      listSkeletons: async () => [],
      getSkeleton: async () => [],
      fetchNodes: async () => [],
      getSpatialIndexMetadata: async () => null,
    };

    expect(
      getEditableSpatiallyIndexedSkeletonSource({ source }),
    ).toBeUndefined();
  });

  it("treats missing or invalid spatial skeleton sources as read-only", () => {
    expect(isSpatiallyIndexedSkeletonSourceReadOnly(undefined)).toBe(true);
    expect(
      isSpatiallyIndexedSkeletonSourceReadOnly({ source: undefined }),
    ).toBe(true);
    expect(
      isSpatiallyIndexedSkeletonSourceReadOnly({
        source: {
          readonly: false,
        },
      }),
    ).toBe(true);
  });

  it("reads spatial skeleton source read-only state", () => {
    const source = {
      readonly: false,
      listSkeletons: async () => [],
      getSkeleton: async () => [],
      fetchNodes: async () => [],
      getSpatialIndexMetadata: async () => null,
    };

    expect(isSpatiallyIndexedSkeletonSourceReadOnly({ source })).toBe(false);
    expect(
      isSpatiallyIndexedSkeletonSourceReadOnly({
        source: {
          ...source,
          readonly: true,
        },
      }),
    ).toBe(true);
  });

  it("clears the full skeleton cache before notifying node data listeners", () => {
    const state = new SpatialSkeletonState();
    const cachedSegmentId = 11;
    (state as any).fullSegmentNodeCache.set(cachedSegmentId, [
      {
        nodeId: 1n,
        segmentId: cachedSegmentId,
        position: new Float32Array([1, 2, 3]),
      },
    ]);

    let cachePresentDuringNotification: boolean | undefined;
    state.nodeDataVersion.changed.add(() => {
      cachePresentDuringNotification = (state as any).fullSegmentNodeCache.has(
        cachedSegmentId,
      );
    });

    state.markNodeDataChanged();

    expect(cachePresentDuringNotification).toBe(false);
    expect((state as any).fullSegmentNodeCache.has(cachedSegmentId)).toBe(
      false,
    );
  });

  it("clears inspected cache state and pending node positions together", () => {
    const state = new SpatialSkeletonState();
    (state as any).replaceCachedSegmentNodes(11n, [
      {
        nodeId: 5n,
        segmentId: 11n,
        position: new Float32Array([1, 2, 3]),
      },
    ]);
    state.setPendingNodePosition(5n, [4, 5, 6]);
    const nodeDataVersion = state.nodeDataVersion.value;
    const pendingNodePositionVersion = state.pendingNodePositionVersion.value;

    expect(state.clearInspectedSkeletonCache()).toBe(true);
    expect(state.getCachedSegmentNodes(11n)).toBeUndefined();
    expect(state.getCachedNode(5n)).toBeUndefined();
    expect(state.getPendingNodePosition(5n)).toBeUndefined();
    expect(state.nodeDataVersion.value).toBe(nodeDataVersion + 1);
    expect(state.pendingNodePositionVersion.value).toBe(
      pendingNodePositionVersion + 1,
    );
  });

  it("can seed a brand-new cached segment from a local node mutation", () => {
    const state = new SpatialSkeletonState();

    const changed = state.upsertCachedNode(
      {
        nodeId: 5n,
        segmentId: 11n,
        position: new Float32Array([1, 2, 3]),
        parentNodeId: undefined,
        isTrueEnd: false,
      },
      { allowUncachedSegment: true },
    );

    expect(changed).toBe(true);
    expect(state.getCachedSegmentNodes(11n)).toEqual([
      {
        nodeId: 5n,
        segmentId: 11n,
        position: new Float32Array([1, 2, 3]),
        parentNodeId: undefined,
        description: undefined,
        isTrueEnd: false,
      },
    ]);
    expect(state.getCachedNode(5n)).toEqual({
      nodeId: 5n,
      segmentId: 11n,
      position: new Float32Array([1, 2, 3]),
      parentNodeId: undefined,
      description: undefined,
      isTrueEnd: false,
    });
  });

  it("updates cached node lookup when a node moves between cached segments", () => {
    const state = new SpatialSkeletonState();
    (state as any).replaceCachedSegmentNodes(11n, [
      {
        nodeId: 5n,
        segmentId: 11n,
        position: new Float32Array([1, 2, 3]),
        parentNodeId: undefined,
        isTrueEnd: false,
      },
    ]);
    (state as any).replaceCachedSegmentNodes(13n, [
      {
        nodeId: 7n,
        segmentId: 13n,
        position: new Float32Array([4, 5, 6]),
        parentNodeId: undefined,
        isTrueEnd: false,
      },
    ]);

    expect(
      state.upsertCachedNode({
        nodeId: 5n,
        segmentId: 13n,
        position: new Float32Array([7, 8, 9]),
        parentNodeId: undefined,
        isTrueEnd: false,
      }),
    ).toBe(true);

    expect(state.getCachedSegmentNodes(11n)).toBeUndefined();
    expect(state.getCachedSegmentNodes(13n)?.map((node) => node.nodeId)).toEqual(
      [5n, 7n],
    );
    expect(state.getCachedNode(5n)).toEqual({
      nodeId: 5n,
      segmentId: 13n,
      position: new Float32Array([7, 8, 9]),
      parentNodeId: undefined,
      description: undefined,
      isTrueEnd: false,
    });
  });

  it("does not drop an existing cached node when upserting into an uncached segment without permission", () => {
    const state = new SpatialSkeletonState();
    (state as any).replaceCachedSegmentNodes(11n, [
      {
        nodeId: 5n,
        segmentId: 11n,
        position: new Float32Array([1, 2, 3]),
        parentNodeId: undefined,
        isTrueEnd: false,
      },
    ]);

    expect(
      state.upsertCachedNode({
        nodeId: 5n,
        segmentId: 13n,
        position: new Float32Array([7, 8, 9]),
        parentNodeId: undefined,
        isTrueEnd: false,
      }),
    ).toBe(false);

    expect(state.getCachedSegmentNodes(11n)).toEqual([
      {
        nodeId: 5n,
        segmentId: 11n,
        position: new Float32Array([1, 2, 3]),
        parentNodeId: undefined,
        description: undefined,
        isTrueEnd: false,
      },
    ]);
    expect(state.getCachedSegmentNodes(13n)).toBeUndefined();
    expect(state.getCachedNode(5n)).toEqual({
      nodeId: 5n,
      segmentId: 11n,
      position: new Float32Array([1, 2, 3]),
      parentNodeId: undefined,
      description: undefined,
      isTrueEnd: false,
    });
  });

  it("does not cache a full segment fetch that was evicted while pending", async () => {
    const state = new SpatialSkeletonState();
    let resolveFetch:
      | ((
          value: Array<{
            nodeId: bigint;
            parentNodeId?: bigint;
            position: Float32Array;
            segmentId: bigint;
            isTrueEnd: boolean;
          }>,
        ) => void)
      | undefined;
    const getSkeleton = vi.fn(
      () =>
        new Promise<
          Array<{
            nodeId: bigint;
            parentNodeId?: bigint;
            position: Float32Array;
            segmentId: bigint;
            isTrueEnd: boolean;
          }>
        >((resolve) => {
          resolveFetch = resolve as typeof resolveFetch;
        }),
    );
    const skeletonLayer = {
      source: {
        readonly: false,
        listSkeletons: async () => [],
        getSkeleton,
        fetchNodes: async () => [],
        getSpatialIndexMetadata: async () => null,
      },
    } as any;

    const pending = state.getFullSegmentNodes(skeletonLayer, 11n);

    state.evictInactiveSegmentNodes([]);
    resolveFetch?.([
      {
        nodeId: 5n,
        parentNodeId: undefined,
        position: new Float32Array([1, 2, 3]),
        segmentId: 11n,
        isTrueEnd: false,
      },
    ]);

    await expect(pending).resolves.toEqual([
      {
        nodeId: 5n,
        segmentId: 11n,
        position: new Float32Array([1, 2, 3]),
        parentNodeId: undefined,
        description: undefined,
        isTrueEnd: false,
      },
    ]);
    expect(state.getCachedSegmentNodes(11n)).toBeUndefined();
    expect(state.getCachedNode(5n)).toBeUndefined();
  });

  it("aborts pending full segment fetches when the cache generation is cleared", async () => {
    const state = new SpatialSkeletonState();
    let receivedSignal: AbortSignal | undefined;
    const getSkeleton = vi.fn(
      (_segmentId: bigint, options?: { signal?: AbortSignal }) =>
        new Promise<never>((_resolve, reject) => {
          receivedSignal = options?.signal;
          options?.signal?.addEventListener(
            "abort",
            () => reject(options.signal?.reason),
            { once: true },
          );
        }),
    );

    const pending = state.getFullSegmentNodes(
      {
        source: {
          readonly: false,
          listSkeletons: async () => [],
          getSkeleton,
          fetchNodes: async () => [],
          getSpatialIndexMetadata: async () => null,
        },
      } as any,
      11n,
    );

    expect(receivedSignal?.aborted).toBe(false);
    expect(state.clearInspectedSkeletonCache()).toBe(true);
    expect(receivedSignal?.aborted).toBe(true);
    await expect(pending).rejects.toMatchObject({ name: "AbortError" });
    expect(state.getCachedSegmentNodes(11n)).toBeUndefined();
    expect(state.getCachedNode(11n)).toBeUndefined();
  });

  it("aborts pending full segment fetches when a segment is invalidated", async () => {
    const state = new SpatialSkeletonState();
    let receivedSignal: AbortSignal | undefined;
    const getSkeleton = vi.fn(
      (_segmentId: bigint, options?: { signal?: AbortSignal }) =>
        new Promise<never>((_resolve, reject) => {
          receivedSignal = options?.signal;
          options?.signal?.addEventListener(
            "abort",
            () => reject(options.signal?.reason),
            { once: true },
          );
        }),
    );

    const pending = state.getFullSegmentNodes(
      {
        source: {
          readonly: false,
          listSkeletons: async () => [],
          getSkeleton,
          fetchNodes: async () => [],
          getSpatialIndexMetadata: async () => null,
        },
      } as any,
      11n,
    );

    expect(receivedSignal?.aborted).toBe(false);
    expect(state.invalidateCachedSegments([11])).toBe(false);
    expect(receivedSignal?.aborted).toBe(true);
    await expect(pending).rejects.toMatchObject({ name: "AbortError" });
    expect(state.getCachedSegmentNodes(11n)).toBeUndefined();
    expect(state.getCachedNode(11n)).toBeUndefined();
  });

  it("aborts pending full segment fetches when a segment is evicted", async () => {
    const state = new SpatialSkeletonState();
    let receivedSignal: AbortSignal | undefined;
    const getSkeleton = vi.fn(
      (_segmentId: bigint, options?: { signal?: AbortSignal }) =>
        new Promise<never>((_resolve, reject) => {
          receivedSignal = options?.signal;
          options?.signal?.addEventListener(
            "abort",
            () => reject(options.signal?.reason),
            { once: true },
          );
        }),
    );

    const pending = state.getFullSegmentNodes(
      {
        source: {
          readonly: false,
          listSkeletons: async () => [],
          getSkeleton,
          fetchNodes: async () => [],
          getSpatialIndexMetadata: async () => null,
        },
      } as any,
      11n,
    );

    expect(receivedSignal?.aborted).toBe(false);
    expect(state.evictInactiveSegmentNodes([])).toBe(false);
    expect(receivedSignal?.aborted).toBe(true);
    await expect(pending).rejects.toMatchObject({ name: "AbortError" });
    expect(state.getCachedSegmentNodes(11n)).toBeUndefined();
    expect(state.getCachedNode(11n)).toBeUndefined();
  });

  it("notifies node data listeners after caching a fetched full segment", async () => {
    const state = new SpatialSkeletonState();
    const getSkeleton = vi.fn(async () => [
      {
        nodeId: 5n,
        parentNodeId: undefined,
        position: new Float32Array([1, 2, 3]),
        segmentId: 11n,
        isTrueEnd: false,
      },
    ]);
    const skeletonLayer = {
      source: {
        readonly: false,
        listSkeletons: async () => [],
        getSkeleton,
        fetchNodes: async () => [],
        getSpatialIndexMetadata: async () => null,
      },
    } as any;
    let notifications = 0;
    state.nodeDataVersion.changed.add(() => {
      notifications += 1;
    });

    await expect(state.getFullSegmentNodes(skeletonLayer, 11n)).resolves.toEqual(
      [
        {
          nodeId: 5n,
          segmentId: 11n,
          position: new Float32Array([1, 2, 3]),
          parentNodeId: undefined,
          description: undefined,
          isTrueEnd: false,
        },
      ],
    );

    expect(notifications).toBe(1);
    expect(state.getCachedNode(5n)).toEqual({
      nodeId: 5n,
      segmentId: 11n,
      position: new Float32Array([1, 2, 3]),
      parentNodeId: undefined,
      description: undefined,
      isTrueEnd: false,
    });
  });

  it("caches inspected source state from full skeleton inspection", async () => {
    const state = new SpatialSkeletonState();
    const getSkeleton = vi.fn(async () => [
      {
        nodeId: 5n,
        parentNodeId: undefined,
        position: new Float32Array([1, 2, 3]),
        segmentId: 11n,
        isTrueEnd: false,
        sourceState: { revisionToken: "2026-03-29T12:30:00Z" },
      },
    ]);

    await expect(
      state.getFullSegmentNodes(
        {
          source: {
            readonly: false,
            listSkeletons: async () => [],
            getSkeleton,
            fetchNodes: async () => [],
            getSpatialIndexMetadata: async () => null,
          },
        } as any,
        11n,
      ),
    ).resolves.toEqual([
      {
        nodeId: 5n,
        segmentId: 11n,
        position: new Float32Array([1, 2, 3]),
        parentNodeId: undefined,
        description: undefined,
        isTrueEnd: false,
        sourceState: { revisionToken: "2026-03-29T12:30:00Z" },
      },
    ]);

    expect(getSkeleton).toHaveBeenCalledTimes(1);
    expect(state.getCachedNode(5n)).toEqual({
      nodeId: 5n,
      segmentId: 11n,
      position: new Float32Array([1, 2, 3]),
      parentNodeId: undefined,
      description: undefined,
      isTrueEnd: false,
      sourceState: { revisionToken: "2026-03-29T12:30:00Z" },
    });
  });

  it("stores merge anchor state only when the node id is valid", () => {
    const state = new SpatialSkeletonState();

    expect(state.setMergeAnchor(5n)).toBe(true);
    expect(state.mergeAnchorNodeId.value).toBe(5n);

    expect(state.setMergeAnchor(0n)).toBe(true);
    expect(state.mergeAnchorNodeId.value).toBeUndefined();
  });

  it("stores provided radius and confidence independently", () => {
    const state = new SpatialSkeletonState();
    (state as any).replaceCachedSegmentNodes(11n, [
      {
        nodeId: 1n,
        segmentId: 11n,
        position: new Float32Array([1, 2, 3]),
        parentNodeId: undefined,
        radius: 4,
        confidence: 50,
      },
    ]);

    expect(state.setNodeRadius(1n, 6)).toBe(true);
    expect(state.setNodeConfidence(1n, 63)).toBe(true);
    expect(state.getCachedNode(1n)).toMatchObject({
      radius: 6,
      confidence: 63,
    });
  });

  it("removes and reparents nodes within the affected cached segment only", () => {
    const state = new SpatialSkeletonState();
    (state as any).replaceCachedSegmentNodes(11n, [
      {
        nodeId: 1n,
        segmentId: 11n,
        position: new Float32Array([1, 1, 1]),
        parentNodeId: undefined,
        isTrueEnd: false,
      },
      {
        nodeId: 2n,
        segmentId: 11n,
        position: new Float32Array([2, 2, 2]),
        parentNodeId: 1n,
        isTrueEnd: false,
      },
      {
        nodeId: 3n,
        segmentId: 11n,
        position: new Float32Array([3, 3, 3]),
        parentNodeId: 1n,
        isTrueEnd: false,
      },
    ]);
    (state as any).replaceCachedSegmentNodes(12n, [
      {
        nodeId: 4n,
        segmentId: 12n,
        position: new Float32Array([4, 4, 4]),
        parentNodeId: undefined,
        isTrueEnd: false,
      },
    ]);

    expect(
      state.removeCachedNode(1n, {
        parentNodeId: undefined,
        childNodeIds: [2n, 3n],
      }),
    ).toBe(true);

    expect(state.getCachedSegmentNodes(11n)).toEqual([
      {
        nodeId: 2n,
        segmentId: 11n,
        position: new Float32Array([2, 2, 2]),
        parentNodeId: undefined,
        description: undefined,
        isTrueEnd: false,
      },
      {
        nodeId: 3n,
        segmentId: 11n,
        position: new Float32Array([3, 3, 3]),
        parentNodeId: undefined,
        description: undefined,
        isTrueEnd: false,
      },
    ]);
    expect(state.getCachedSegmentNodes(12n)).toEqual([
      {
        nodeId: 4n,
        segmentId: 12n,
        position: new Float32Array([4, 4, 4]),
        parentNodeId: undefined,
        description: undefined,
        isTrueEnd: false,
      },
    ]);
  });

  it("reroots cached segment topology, confidence, and derived ordering", () => {
    const state = new SpatialSkeletonState();
    (state as any).replaceCachedSegmentNodes(11n, [
      {
        nodeId: 1n,
        segmentId: 11n,
        position: new Float32Array([1, 1, 1]),
        parentNodeId: undefined,
        confidence: 80,
      },
      {
        nodeId: 2n,
        segmentId: 11n,
        position: new Float32Array([2, 2, 2]),
        parentNodeId: 1n,
        confidence: 20,
      },
      {
        nodeId: 3n,
        segmentId: 11n,
        position: new Float32Array([3, 3, 3]),
        parentNodeId: 2n,
        confidence: 10,
      },
      {
        nodeId: 4n,
        segmentId: 11n,
        position: new Float32Array([4, 4, 4]),
        parentNodeId: 2n,
        confidence: 40,
      },
      {
        nodeId: 5n,
        segmentId: 11n,
        position: new Float32Array([5, 5, 5]),
        parentNodeId: 1n,
        confidence: 50,
      },
    ]);

    expect(state.rerootCachedSegment(3n)).toEqual([3n, 2n, 1n]);

    const cachedNodes = state.getCachedSegmentNodes(11n)!;
    expect(cachedNodes.find((node) => node.nodeId === 3n)).toMatchObject({
      parentNodeId: undefined,
      confidence: 100,
    });
    expect(cachedNodes.find((node) => node.nodeId === 2n)).toMatchObject({
      parentNodeId: 3n,
      confidence: 10,
    });
    expect(cachedNodes.find((node) => node.nodeId === 1n)).toMatchObject({
      parentNodeId: 2n,
      confidence: 20,
    });
    expect(cachedNodes.find((node) => node.nodeId === 4n)).toMatchObject({
      parentNodeId: 2n,
      confidence: 40,
    });
    expect(cachedNodes.find((node) => node.nodeId === 5n)).toMatchObject({
      parentNodeId: 1n,
      confidence: 50,
    });

    const graph = buildSpatiallyIndexedSkeletonNavigationGraph(cachedNodes);
    expect(getSkeletonRootNode(graph).nodeId).toBe(3n);
    expect(getFlatListNodeIds(graph)).toEqual([3n, 2n, 4n, 1n, 5n]);
  });

  it("stores empty segments in the cache if nothing present for that segment in cache", () => {
    const state = new SpatialSkeletonState();
    (state as any).replaceCachedSegmentNodes(1n, []);
    expect(state.getCachedSegmentNodes(1n)?.length).toBe(0);
  });

  it("deletes segment from cache if the segment becomes empty", () => {
    const state = new SpatialSkeletonState();
    const node = {
      nodeId: 1n,
      segmentId: 1n,
      position: new Float32Array([1, 1, 1]),
    };
    (state as any).replaceCachedSegmentNodes(1n, [node]);
    expect(state.getCachedSegmentNodes(1n)).toStrictEqual([node]);
    expect(state.getCachedNode(1n)).toBe(node);
    (state as any).replaceCachedSegmentNodes(1n, []);
    expect(state.getCachedSegmentNodes(1n)).toBeUndefined();
    expect(state.getCachedNode(1n)).toBeUndefined();
  });
});
