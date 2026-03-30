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

import {
  buildSpatiallyIndexedSkeletonNavigationGraph,
  getFlatListNodeIds,
  getSkeletonRootNode,
} from "#src/skeleton/navigation.js";
import {
  getSpatiallyIndexedSkeletonSourceCapabilities,
  isEditableSpatiallyIndexedSkeletonSource,
  SpatialSkeletonState,
} from "#src/skeleton/state.js";

describe("skeleton/state", () => {
  it("detects reroot capability without making it required for editable sources", () => {
    const editableSource = {
      getSkeleton: async () => [],
      addNode: async () => ({ treenodeId: 1, skeletonId: 1 }),
      moveNode: async () => {},
      deleteNode: async () => {},
      updateDescription: async () => {},
      setTrueEnd: async () => {},
      removeTrueEnd: async () => {},
      updateRadius: async () => {},
      updateConfidence: async () => {},
      mergeSkeletons: async () => ({
        resultSkeletonId: 1,
        deletedSkeletonId: 2,
        stableAnnotationSwap: false,
      }),
      splitSkeleton: async () => ({
        existingSkeletonId: 1,
        newSkeletonId: 2,
      }),
    };

    expect(isEditableSpatiallyIndexedSkeletonSource(editableSource)).toBe(true);
    expect(
      getSpatiallyIndexedSkeletonSourceCapabilities(editableSource)
        .rerootSkeletons,
    ).toBe(false);
    expect(
      getSpatiallyIndexedSkeletonSourceCapabilities({
        ...editableSource,
        rerootSkeleton: async () => {},
      }).rerootSkeletons,
    ).toBe(true);
  });

  it("clears the full skeleton cache before notifying node data listeners", () => {
    const state = new SpatialSkeletonState();
    const cachedSegmentId = 11;
    (state as any).fullSegmentNodeCache.set(cachedSegmentId, [
      {
        nodeId: 1,
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

  it("can seed a brand-new cached segment from a local node mutation", () => {
    const state = new SpatialSkeletonState();

    const changed = state.upsertCachedNode(
      {
        nodeId: 5,
        segmentId: 11,
        position: new Float32Array([1, 2, 3]),
        parentNodeId: undefined,
      },
      { allowUncachedSegment: true },
    );

    expect(changed).toBe(true);
    expect(state.getCachedSegmentNodes(11)).toEqual([
      {
        nodeId: 5,
        segmentId: 11,
        position: new Float32Array([1, 2, 3]),
        parentNodeId: undefined,
        labels: undefined,
      },
    ]);
    expect(state.getCachedNode(5)).toEqual({
      nodeId: 5,
      segmentId: 11,
      position: new Float32Array([1, 2, 3]),
      parentNodeId: undefined,
      labels: undefined,
    });
  });

  it("does not cache a full segment fetch that was evicted while pending", async () => {
    const state = new SpatialSkeletonState();
    let resolveFetch: ((value: Array<{ id: number; parent_id: null; x: number; y: number; z: number; skeleton_id: number }>) => void) | undefined;
    const getSkeleton = vi.fn(
      () =>
        new Promise<
          Array<{
            id: number;
            parent_id: null;
            x: number;
            y: number;
            z: number;
            skeleton_id: number;
          }>
        >((resolve) => {
          resolveFetch = resolve;
        }),
    );
    const skeletonLayer = {
      source: { getSkeleton },
    } as any;

    const pending = state.getFullSegmentNodes(skeletonLayer, 11);

    state.evictInactiveSegmentNodes([]);
    resolveFetch?.([
      {
        id: 5,
        parent_id: null,
        x: 1,
        y: 2,
        z: 3,
        skeleton_id: 11,
      },
    ]);

    await expect(pending).resolves.toEqual([
      {
        nodeId: 5,
        segmentId: 11,
        position: new Float32Array([1, 2, 3]),
        parentNodeId: undefined,
        labels: undefined,
      },
    ]);
    expect(state.getCachedSegmentNodes(11)).toBeUndefined();
    expect(state.getCachedNode(5)).toBeUndefined();
  });

  it("notifies node data listeners after caching a fetched full segment", async () => {
    const state = new SpatialSkeletonState();
    const getSkeleton = vi.fn(async () => [
      {
        id: 5,
        parent_id: null,
        x: 1,
        y: 2,
        z: 3,
        skeleton_id: 11,
      },
    ]);
    const skeletonLayer = {
      source: { getSkeleton },
    } as any;
    let notifications = 0;
    state.nodeDataVersion.changed.add(() => {
      notifications += 1;
    });

    await expect(state.getFullSegmentNodes(skeletonLayer, 11)).resolves.toEqual([
      {
        nodeId: 5,
        segmentId: 11,
        position: new Float32Array([1, 2, 3]),
        parentNodeId: undefined,
        labels: undefined,
      },
    ]);

    expect(notifications).toBe(1);
    expect(state.getCachedNode(5)).toEqual({
      nodeId: 5,
      segmentId: 11,
      position: new Float32Array([1, 2, 3]),
      parentNodeId: undefined,
      labels: undefined,
    });
  });

  it("keeps inspection bounded to a primary and secondary segment", () => {
    const state = new SpatialSkeletonState();
    (state as any).fullSegmentNodeCache.set(11, [
      {
        nodeId: 1,
        segmentId: 11,
        position: new Float32Array([1, 2, 3]),
      },
    ]);
    (state as any).fullSegmentNodeCache.set(13, [
      {
        nodeId: 2,
        segmentId: 13,
        position: new Float32Array([4, 5, 6]),
      },
    ]);
    (state as any).fullSegmentNodeCache.set(17, [
      {
        nodeId: 3,
        segmentId: 17,
        position: new Float32Array([7, 8, 9]),
      },
    ]);

    expect(state.setInspectedSegments(11, 13)).toBe(true);
    expect(state.getInspectedSegmentIds()).toEqual([11, 13]);
    expect(state.getCachedSegmentNodes(17)).toBeUndefined();

    expect(state.inspectSegment(17)).toBe(true);
    expect(state.getInspectedSegmentIds()).toEqual([17]);
    expect(state.getCachedSegmentNodes(11)).toBeUndefined();
    expect(state.getCachedSegmentNodes(13)).toBeUndefined();
  });

  it("stores merge anchor state only when both node and segment ids are valid", () => {
    const state = new SpatialSkeletonState();

    expect(state.setMergeAnchor(5, 11)).toBe(true);
    expect(state.mergeAnchorNodeId.value).toBe(5);
    expect(state.mergeAnchorSegmentId.value).toBe(11);

    expect(state.setMergeAnchor(7, undefined)).toBe(true);
    expect(state.mergeAnchorNodeId.value).toBeUndefined();
    expect(state.mergeAnchorSegmentId.value).toBeUndefined();
  });

  it("keeps primary inspection stable when adding a secondary segment", () => {
    const state = new SpatialSkeletonState();

    expect(state.inspectSegment(11)).toBe(true);
    expect(state.inspectSegment(13, { secondary: true })).toBe(true);
    expect(state.getInspectedSegmentIds()).toEqual([11, 13]);

    expect(state.inspectSegment(11, { secondary: true })).toBe(true);
    expect(state.getInspectedSegmentIds()).toEqual([11]);
  });

  it("reroots cached segment topology, confidence, and derived ordering", () => {
    const state = new SpatialSkeletonState();
    (state as any).fullSegmentNodeCache.set(11, [
      {
        nodeId: 1,
        segmentId: 11,
        position: new Float32Array([1, 1, 1]),
        parentNodeId: undefined,
        confidence: 10,
      },
      {
        nodeId: 2,
        segmentId: 11,
        position: new Float32Array([2, 2, 2]),
        parentNodeId: 1,
        confidence: 20,
      },
      {
        nodeId: 3,
        segmentId: 11,
        position: new Float32Array([3, 3, 3]),
        parentNodeId: 2,
        confidence: 30,
      },
      {
        nodeId: 4,
        segmentId: 11,
        position: new Float32Array([4, 4, 4]),
        parentNodeId: 2,
        confidence: 40,
      },
      {
        nodeId: 5,
        segmentId: 11,
        position: new Float32Array([5, 5, 5]),
        parentNodeId: 1,
        confidence: 50,
      },
    ]);
    (state as any).nodePropertyOverrides.set(2, {
      radius: 7,
      confidence: 20,
    });
    (state as any).rebuildCachedNodesById();

    expect(state.rerootCachedSegment(3)).toBe(true);

    const cachedNodes = state.getCachedSegmentNodes(11)!;
    expect(cachedNodes.find((node) => node.nodeId === 3)).toMatchObject({
      parentNodeId: undefined,
      confidence: 100,
    });
    expect(cachedNodes.find((node) => node.nodeId === 2)).toMatchObject({
      parentNodeId: 3,
      confidence: 30,
    });
    expect(cachedNodes.find((node) => node.nodeId === 1)).toMatchObject({
      parentNodeId: 2,
      confidence: 20,
    });
    expect(cachedNodes.find((node) => node.nodeId === 4)).toMatchObject({
      parentNodeId: 2,
      confidence: 40,
    });
    expect(cachedNodes.find((node) => node.nodeId === 5)).toMatchObject({
      parentNodeId: 1,
      confidence: 50,
    });
    expect(state.getNodePropertyOverride(2)).toEqual({
      radius: 7,
      confidence: 30,
    });

    const graph = buildSpatiallyIndexedSkeletonNavigationGraph(cachedNodes);
    expect(getSkeletonRootNode(graph).nodeId).toBe(3);
    expect(getFlatListNodeIds(graph)).toEqual([3, 2, 4, 1, 5]);
  });
});
