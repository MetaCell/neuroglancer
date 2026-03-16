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

import type {
  EditableSpatiallyIndexedSkeletonSource,
  SpatiallyIndexedSkeletonNode,
} from "#src/skeleton/api.js";
import type {
  SpatiallyIndexedSkeletonLayer,
  SpatiallyIndexedSkeletonNodeInfo,
} from "#src/skeleton/frontend.js";
import { WatchableValue } from "#src/trackable_value.js";
import { RefCounted } from "#src/util/disposable.js";

interface SpatialSkeletonSourceAccess {
  source: unknown;
}

export interface SpatiallyIndexedSkeletonSourceCapabilities {
  inspectSkeletons: boolean;
  addNodes: boolean;
  moveNodes: boolean;
  deleteNodes: boolean;
  editNodeLabels: boolean;
  mergeSkeletons: boolean;
  splitSkeletons: boolean;
}

export type SpatiallyIndexedSkeletonSourceCapability =
  keyof SpatiallyIndexedSkeletonSourceCapabilities;

export interface SpatiallyIndexedSkeletonInspectionSource {
  getSkeleton(
    skeletonId: number,
  ): Promise<readonly SpatiallyIndexedSkeletonNode[]>;
}

export const NO_SPATIALLY_INDEXED_SKELETON_SOURCE_CAPABILITIES: SpatiallyIndexedSkeletonSourceCapabilities =
  {
    inspectSkeletons: false,
    addNodes: false,
    moveNodes: false,
    deleteNodes: false,
    editNodeLabels: false,
    mergeSkeletons: false,
    splitSkeletons: false,
  };

function hasFunction<T extends string>(
  value: unknown,
  property: T,
): value is Record<T, (...args: any[]) => unknown> {
  return (
    typeof value === "object" &&
    value !== null &&
    typeof (value as Record<string, unknown>)[property] === "function"
  );
}

export function isEditableSpatiallyIndexedSkeletonSource(
  value: unknown,
): value is EditableSpatiallyIndexedSkeletonSource {
  const capabilities = getSpatiallyIndexedSkeletonSourceCapabilities(value);
  return (
    capabilities.inspectSkeletons &&
    capabilities.addNodes &&
    capabilities.moveNodes &&
    capabilities.deleteNodes &&
    capabilities.editNodeLabels &&
    capabilities.mergeSkeletons &&
    capabilities.splitSkeletons
  );
}

export function getEditableSpatiallyIndexedSkeletonSource(
  value: SpatialSkeletonSourceAccess | undefined,
): EditableSpatiallyIndexedSkeletonSource | undefined {
  if (value === undefined) return undefined;
  return isEditableSpatiallyIndexedSkeletonSource(value.source)
    ? value.source
    : undefined;
}

export function getSpatiallyIndexedSkeletonInspectionSource(
  value: SpatialSkeletonSourceAccess | undefined,
): SpatiallyIndexedSkeletonInspectionSource | undefined {
  if (value === undefined) return undefined;
  return hasFunction(value.source, "getSkeleton")
    ? (value.source as SpatiallyIndexedSkeletonInspectionSource)
    : undefined;
}

export function getSpatiallyIndexedSkeletonSourceCapabilities(
  value: unknown,
): SpatiallyIndexedSkeletonSourceCapabilities {
  return {
    inspectSkeletons: hasFunction(value, "getSkeleton"),
    addNodes: hasFunction(value, "addNode"),
    moveNodes: hasFunction(value, "moveNode"),
    deleteNodes: hasFunction(value, "deleteNode"),
    editNodeLabels:
      hasFunction(value, "addNodeLabel") &&
      hasFunction(value, "removeNodeLabel"),
    mergeSkeletons: hasFunction(value, "mergeSkeletons"),
    splitSkeletons: hasFunction(value, "splitSkeleton"),
  };
}

export function hasSpatiallyIndexedSkeletonSourceCapability(
  capabilities: SpatiallyIndexedSkeletonSourceCapabilities,
  capability: SpatiallyIndexedSkeletonSourceCapability,
) {
  return capabilities[capability];
}

export function hasAnySpatiallyIndexedSkeletonEditingCapability(
  capabilities: SpatiallyIndexedSkeletonSourceCapabilities,
) {
  return (
    capabilities.addNodes ||
    capabilities.moveNodes ||
    capabilities.deleteNodes ||
    capabilities.editNodeLabels ||
    capabilities.mergeSkeletons ||
    capabilities.splitSkeletons
  );
}

function spatiallyIndexedSkeletonSourceCapabilitiesEqual(
  a: SpatiallyIndexedSkeletonSourceCapabilities,
  b: SpatiallyIndexedSkeletonSourceCapabilities,
) {
  return (
    a.inspectSkeletons === b.inspectSkeletons &&
    a.addNodes === b.addNodes &&
    a.moveNodes === b.moveNodes &&
    a.deleteNodes === b.deleteNodes &&
    a.editNodeLabels === b.editNodeLabels &&
    a.mergeSkeletons === b.mergeSkeletons &&
    a.splitSkeletons === b.splitSkeletons
  );
}

export function mapSpatiallyIndexedSkeletonNodeToNodeInfo(
  node: SpatiallyIndexedSkeletonNode,
  fallbackSegmentId: number,
): SpatiallyIndexedSkeletonNodeInfo | undefined {
  const nodeId = Number(node.id);
  const segmentIdValue = Number(node.skeleton_id);
  const x = Number(node.x);
  const y = Number(node.y);
  const z = Number(node.z);
  if (
    !Number.isFinite(nodeId) ||
    !Number.isFinite(segmentIdValue) ||
    !Number.isFinite(x) ||
    !Number.isFinite(y) ||
    !Number.isFinite(z)
  ) {
    return undefined;
  }
  const parentNodeId =
    node.parent_id === undefined ||
    node.parent_id === null ||
    !Number.isFinite(Number(node.parent_id))
      ? undefined
      : Math.round(Number(node.parent_id));
  return {
    nodeId: Math.round(nodeId),
    segmentId: Math.round(
      Number.isFinite(segmentIdValue) ? segmentIdValue : fallbackSegmentId,
    ),
    position: new Float32Array([x, y, z]),
    parentNodeId,
    labels: node.labels,
  };
}

function cloneSpatiallyIndexedSkeletonNodeInfo(
  node: SpatiallyIndexedSkeletonNodeInfo,
): SpatiallyIndexedSkeletonNodeInfo {
  return {
    ...node,
    position: new Float32Array(node.position),
    labels: node.labels === undefined ? undefined : [...node.labels],
  };
}

export class SpatialSkeletonState extends RefCounted {
  readonly sourceCapabilities = new WatchableValue(
    NO_SPATIALLY_INDEXED_SKELETON_SOURCE_CAPABILITIES,
  );
  readonly editMode = new WatchableValue(false);
  readonly mergeMode = new WatchableValue(false);
  readonly splitMode = new WatchableValue(false);
  readonly selectedNodeId = new WatchableValue<number | undefined>(undefined);
  readonly treeEndNodeId = new WatchableValue<number | undefined>(undefined);
  readonly visibleChunksNeeded = new WatchableValue(0);
  readonly visibleChunksAvailable = new WatchableValue(0);
  readonly visibleChunksLoaded = new WatchableValue(false);
  readonly nodeDataVersion = new WatchableValue(0);
  readonly pendingNodePositionVersion = new WatchableValue(0);

  private nodeDescriptions = new Map<number, string>();
  private pendingNodePositions = new Map<number, Float32Array>();
  private fullSkeletonCacheGeneration = 0;
  private fullSegmentNodeCache = new Map<
    number,
    SpatiallyIndexedSkeletonNodeInfo[]
  >();
  private pendingFullSegmentNodeFetches = new Map<
    number,
    Promise<SpatiallyIndexedSkeletonNodeInfo[]>
  >();
  private cachedNodesById = new Map<number, SpatiallyIndexedSkeletonNodeInfo>();

  setNodeDescription(nodeId: number, description: string) {
    const value = description.trim();
    if (value.length === 0) {
      this.nodeDescriptions.delete(nodeId);
      return;
    }
    this.nodeDescriptions.set(nodeId, value);
  }

  getNodeDescription(nodeId: number) {
    return this.nodeDescriptions.get(nodeId);
  }

  getPendingNodeIds() {
    return this.pendingNodePositions.keys();
  }

  getPendingNodePosition(nodeId: number) {
    return this.pendingNodePositions.get(nodeId);
  }

  setPendingNodePosition(nodeId: number, position: ArrayLike<number>) {
    const normalizedNodeId = Math.round(Number(nodeId));
    const x = Number(position[0]);
    const y = Number(position[1]);
    const z = Number(position[2]);
    if (
      !Number.isSafeInteger(normalizedNodeId) ||
      normalizedNodeId <= 0 ||
      !Number.isFinite(x) ||
      !Number.isFinite(y) ||
      !Number.isFinite(z)
    ) {
      return false;
    }
    const existing = this.pendingNodePositions.get(normalizedNodeId);
    if (
      existing !== undefined &&
      existing[0] === x &&
      existing[1] === y &&
      existing[2] === z
    ) {
      return false;
    }
    this.pendingNodePositions.set(
      normalizedNodeId,
      new Float32Array([x, y, z]),
    );
    this.pendingNodePositionVersion.value =
      this.pendingNodePositionVersion.value + 1;
    return true;
  }

  clearPendingNodePosition(nodeId: number) {
    const normalizedNodeId = Math.round(Number(nodeId));
    if (
      !Number.isSafeInteger(normalizedNodeId) ||
      normalizedNodeId <= 0 ||
      !this.pendingNodePositions.delete(normalizedNodeId)
    ) {
      return false;
    }
    this.pendingNodePositionVersion.value =
      this.pendingNodePositionVersion.value + 1;
    return true;
  }

  clearPendingNodePositions() {
    if (this.pendingNodePositions.size === 0) {
      return false;
    }
    this.pendingNodePositions.clear();
    this.pendingNodePositionVersion.value =
      this.pendingNodePositionVersion.value + 1;
    return true;
  }

  updateSourceCapabilities(
    capabilities: SpatiallyIndexedSkeletonSourceCapabilities,
  ) {
    if (
      spatiallyIndexedSkeletonSourceCapabilitiesEqual(
        this.sourceCapabilities.value,
        capabilities,
      )
    ) {
      return;
    }
    this.sourceCapabilities.value = capabilities;
  }

  markNodeDataChanged(options: { invalidateFullSkeletonCache?: boolean } = {}) {
    if (options.invalidateFullSkeletonCache ?? true) {
      this.clearFullSkeletonCache();
    }
    this.nodeDataVersion.value = this.nodeDataVersion.value + 1;
  }

  updateChunkLoadState(needed: number, available: number) {
    this.visibleChunksNeeded.value = needed;
    this.visibleChunksAvailable.value = available;
    this.visibleChunksLoaded.value = needed > 0 && available >= needed;
  }

  getCachedSegmentNodes(segmentId: number) {
    return this.fullSegmentNodeCache.get(segmentId);
  }

  getCachedNode(nodeId: number) {
    return this.cachedNodesById.get(nodeId);
  }

  updateCachedNode(
    nodeId: number,
    update: (
      node: SpatiallyIndexedSkeletonNodeInfo,
    ) => SpatiallyIndexedSkeletonNodeInfo,
  ) {
    let changed = false;
    for (const [segmentId, segmentNodes] of this.fullSegmentNodeCache) {
      let segmentChanged = false;
      const nextSegmentNodes = segmentNodes.map((candidate) => {
        if (candidate.nodeId !== nodeId) return candidate;
        const updatedNode = update(candidate);
        segmentChanged ||= updatedNode !== candidate;
        return updatedNode;
      });
      if (!segmentChanged) continue;
      this.fullSegmentNodeCache.set(segmentId, nextSegmentNodes);
      changed = true;
    }
    if (changed) {
      this.rebuildCachedNodesById();
    }
    return changed;
  }

  upsertCachedNode(
    node: SpatiallyIndexedSkeletonNodeInfo,
    options: { allowUncachedSegment?: boolean } = {},
  ) {
    const normalizedNode = cloneSpatiallyIndexedSkeletonNodeInfo(node);
    const targetSegmentCached = this.fullSegmentNodeCache.has(
      normalizedNode.segmentId,
    );
    if (
      !targetSegmentCached &&
      !(options.allowUncachedSegment ?? false) &&
      !this.cachedNodesById.has(normalizedNode.nodeId)
    ) {
      return false;
    }
    let changed = false;
    let foundInTargetSegment = false;
    for (const [segmentId, segmentNodes] of this.fullSegmentNodeCache) {
      const existingIndex = segmentNodes.findIndex(
        (candidate) => candidate.nodeId === normalizedNode.nodeId,
      );
      if (existingIndex === -1) {
        continue;
      }
      if (segmentId === normalizedNode.segmentId) {
        const nextSegmentNodes = segmentNodes.slice();
        nextSegmentNodes[existingIndex] = normalizedNode;
        this.fullSegmentNodeCache.set(segmentId, nextSegmentNodes);
        changed = true;
        foundInTargetSegment = true;
        continue;
      }
      const nextSegmentNodes = segmentNodes.filter(
        (candidate) => candidate.nodeId !== normalizedNode.nodeId,
      );
      this.fullSegmentNodeCache.set(segmentId, nextSegmentNodes);
      changed = true;
    }
    if (!targetSegmentCached && (options.allowUncachedSegment ?? false)) {
      this.pendingFullSegmentNodeFetches.delete(normalizedNode.segmentId);
      this.fullSegmentNodeCache.set(normalizedNode.segmentId, [normalizedNode]);
      changed = true;
    } else if (targetSegmentCached && !foundInTargetSegment) {
      const targetNodes = this.fullSegmentNodeCache.get(
        normalizedNode.segmentId,
      )!;
      this.fullSegmentNodeCache.set(
        normalizedNode.segmentId,
        [...targetNodes, normalizedNode].sort((a, b) => a.nodeId - b.nodeId),
      );
      changed = true;
    }
    if (changed) {
      this.rebuildCachedNodesById();
    }
    return changed;
  }

  moveCachedNode(nodeId: number, position: ArrayLike<number>) {
    const x = Number(position[0]);
    const y = Number(position[1]);
    const z = Number(position[2]);
    if (!Number.isFinite(x) || !Number.isFinite(y) || !Number.isFinite(z)) {
      return false;
    }
    return this.updateCachedNode(nodeId, (node) => {
      if (
        node.position[0] === x &&
        node.position[1] === y &&
        node.position[2] === z
      ) {
        return node;
      }
      return {
        ...node,
        position: new Float32Array([x, y, z]),
      };
    });
  }

  removeCachedNode(
    nodeId: number,
    options: {
      parentNodeId?: number;
      childNodeIds?: Iterable<number>;
    } = {},
  ) {
    const childNodeIds = options.childNodeIds
      ? new Set(options.childNodeIds)
      : undefined;
    let changed = false;
    for (const [segmentId, segmentNodes] of this.fullSegmentNodeCache) {
      let segmentChanged = false;
      const nextSegmentNodes: SpatiallyIndexedSkeletonNodeInfo[] = [];
      for (const candidate of segmentNodes) {
        if (candidate.nodeId === nodeId) {
          segmentChanged = true;
          continue;
        }
        if (childNodeIds?.has(candidate.nodeId)) {
          nextSegmentNodes.push({
            ...candidate,
            parentNodeId: options.parentNodeId,
          });
          segmentChanged = true;
          continue;
        }
        nextSegmentNodes.push(candidate);
      }
      if (!segmentChanged) continue;
      this.fullSegmentNodeCache.set(segmentId, nextSegmentNodes);
      changed = true;
    }
    if (changed) {
      this.rebuildCachedNodesById();
    }
    return changed;
  }

  setCachedNodeParent(nodeId: number, parentNodeId: number | undefined) {
    return this.updateCachedNode(nodeId, (node) => {
      if (node.parentNodeId === parentNodeId) {
        return node;
      }
      return {
        ...node,
        parentNodeId,
      };
    });
  }

  invalidateCachedSegments(segmentIds: Iterable<number>) {
    let changed = false;
    for (const segmentId of segmentIds) {
      const normalizedSegmentId = Math.round(Number(segmentId));
      if (
        !Number.isSafeInteger(normalizedSegmentId) ||
        normalizedSegmentId <= 0
      ) {
        continue;
      }
      changed =
        this.fullSegmentNodeCache.delete(normalizedSegmentId) || changed;
      this.pendingFullSegmentNodeFetches.delete(normalizedSegmentId);
    }
    if (changed) {
      this.rebuildCachedNodesById();
    }
    return changed;
  }

  mergeCachedSegments(options: {
    resultSegmentId: number;
    mergedSegmentId: number;
    childNodeId: number;
    parentNodeId: number;
  }) {
    const resultSegmentId = Math.round(Number(options.resultSegmentId));
    const mergedSegmentId = Math.round(Number(options.mergedSegmentId));
    const childNodeId = Math.round(Number(options.childNodeId));
    const parentNodeId = Math.round(Number(options.parentNodeId));
    if (
      !Number.isSafeInteger(resultSegmentId) ||
      resultSegmentId <= 0 ||
      !Number.isSafeInteger(mergedSegmentId) ||
      mergedSegmentId <= 0 ||
      !Number.isSafeInteger(childNodeId) ||
      childNodeId <= 0 ||
      !Number.isSafeInteger(parentNodeId) ||
      parentNodeId <= 0
    ) {
      return false;
    }
    const resultNodes = this.fullSegmentNodeCache.get(resultSegmentId);
    const mergedNodes = this.fullSegmentNodeCache.get(mergedSegmentId);
    if (resultNodes === undefined || mergedNodes === undefined) {
      return this.invalidateCachedSegments([resultSegmentId, mergedSegmentId]);
    }
    const nextResultNodes = [
      ...resultNodes.map(cloneSpatiallyIndexedSkeletonNodeInfo),
      ...mergedNodes.map((node) => {
        const nextNode = cloneSpatiallyIndexedSkeletonNodeInfo(node);
        nextNode.segmentId = resultSegmentId;
        if (nextNode.nodeId === childNodeId) {
          nextNode.parentNodeId = parentNodeId;
        }
        return nextNode;
      }),
    ].sort((a, b) => a.nodeId - b.nodeId);
    this.pendingFullSegmentNodeFetches.delete(resultSegmentId);
    this.pendingFullSegmentNodeFetches.delete(mergedSegmentId);
    this.fullSegmentNodeCache.set(resultSegmentId, nextResultNodes);
    this.fullSegmentNodeCache.delete(mergedSegmentId);
    this.rebuildCachedNodesById();
    return true;
  }

  splitCachedSegmentAtNode(options: {
    existingSegmentId: number;
    nodeId: number;
    newSegmentId: number;
  }) {
    const existingSegmentId = Math.round(Number(options.existingSegmentId));
    const nodeId = Math.round(Number(options.nodeId));
    const newSegmentId = Math.round(Number(options.newSegmentId));
    if (
      !Number.isSafeInteger(existingSegmentId) ||
      existingSegmentId <= 0 ||
      !Number.isSafeInteger(nodeId) ||
      nodeId <= 0 ||
      !Number.isSafeInteger(newSegmentId) ||
      newSegmentId <= 0
    ) {
      return false;
    }
    const existingNodes = this.fullSegmentNodeCache.get(existingSegmentId);
    if (existingNodes === undefined) {
      return this.invalidateCachedSegments([existingSegmentId, newSegmentId]);
    }
    const childrenByParent = new Map<number, number[]>();
    for (const node of existingNodes) {
      if (node.parentNodeId === undefined) continue;
      let children = childrenByParent.get(node.parentNodeId);
      if (children === undefined) {
        children = [];
        childrenByParent.set(node.parentNodeId, children);
      }
      children.push(node.nodeId);
    }
    const movedNodeIds = new Set<number>();
    const queue = [nodeId];
    for (let queueIndex = 0; queueIndex < queue.length; ++queueIndex) {
      const currentNodeId = queue[queueIndex];
      if (movedNodeIds.has(currentNodeId)) continue;
      movedNodeIds.add(currentNodeId);
      for (const childNodeId of childrenByParent.get(currentNodeId) ?? []) {
        queue.push(childNodeId);
      }
    }
    if (!movedNodeIds.has(nodeId)) {
      return this.invalidateCachedSegments([existingSegmentId, newSegmentId]);
    }
    const remainingNodes: SpatiallyIndexedSkeletonNodeInfo[] = [];
    const movedNodes: SpatiallyIndexedSkeletonNodeInfo[] = [];
    for (const node of existingNodes) {
      const nextNode = cloneSpatiallyIndexedSkeletonNodeInfo(node);
      if (movedNodeIds.has(node.nodeId)) {
        nextNode.segmentId = newSegmentId;
        if (nextNode.nodeId === nodeId) {
          nextNode.parentNodeId = undefined;
        }
        movedNodes.push(nextNode);
      } else {
        remainingNodes.push(nextNode);
      }
    }
    if (movedNodes.length === 0) {
      return false;
    }
    this.pendingFullSegmentNodeFetches.delete(existingSegmentId);
    this.pendingFullSegmentNodeFetches.delete(newSegmentId);
    this.fullSegmentNodeCache.set(existingSegmentId, remainingNodes);
    this.fullSegmentNodeCache.set(
      newSegmentId,
      movedNodes.sort((a, b) => a.nodeId - b.nodeId),
    );
    this.rebuildCachedNodesById();
    return true;
  }

  evictInactiveSegmentNodes(activeSegmentIds: Iterable<number>) {
    const activeSegmentIdSet = new Set(activeSegmentIds);
    let changed = false;
    for (const segmentId of this.fullSegmentNodeCache.keys()) {
      if (activeSegmentIdSet.has(segmentId)) continue;
      this.fullSegmentNodeCache.delete(segmentId);
      changed = true;
    }
    for (const segmentId of this.pendingFullSegmentNodeFetches.keys()) {
      if (activeSegmentIdSet.has(segmentId)) continue;
      this.pendingFullSegmentNodeFetches.delete(segmentId);
    }
    if (changed) {
      this.rebuildCachedNodesById();
    }
  }

  async getFullSegmentNodes(
    skeletonLayer: SpatiallyIndexedSkeletonLayer,
    segmentId: number,
  ): Promise<SpatiallyIndexedSkeletonNodeInfo[]> {
    const cached = this.fullSegmentNodeCache.get(segmentId);
    if (cached !== undefined) {
      return cached;
    }
    const pending = this.pendingFullSegmentNodeFetches.get(segmentId);
    if (pending !== undefined) {
      return pending;
    }
    const skeletonSource =
      getSpatiallyIndexedSkeletonInspectionSource(skeletonLayer);
    if (skeletonSource === undefined) {
      throw new Error(
        "The active spatial skeleton source does not expose full skeleton inspection.",
      );
    }
    const fetchVersion = this.fullSkeletonCacheGeneration;
    let fetchPromise!: Promise<SpatiallyIndexedSkeletonNodeInfo[]>;
    fetchPromise = (async () => {
      const fetchedNodes = await skeletonSource.getSkeleton(segmentId);
      const dedupedNodes = new Map<number, SpatiallyIndexedSkeletonNodeInfo>();
      for (const fetchedNode of fetchedNodes) {
        const mappedNode = mapSpatiallyIndexedSkeletonNodeToNodeInfo(
          fetchedNode,
          segmentId,
        );
        if (mappedNode === undefined) continue;
        if (!dedupedNodes.has(mappedNode.nodeId)) {
          dedupedNodes.set(mappedNode.nodeId, mappedNode);
        }
      }
      const normalizedNodes = [...dedupedNodes.values()].sort(
        (a, b) => a.nodeId - b.nodeId,
      );
      if (
        this.fullSkeletonCacheGeneration === fetchVersion &&
        this.pendingFullSegmentNodeFetches.get(segmentId) === fetchPromise
      ) {
        this.fullSegmentNodeCache.set(segmentId, normalizedNodes);
        this.rebuildCachedNodesById();
      }
      return normalizedNodes;
    })().finally(() => {
      if (this.pendingFullSegmentNodeFetches.get(segmentId) === fetchPromise) {
        this.pendingFullSegmentNodeFetches.delete(segmentId);
      }
    });
    this.pendingFullSegmentNodeFetches.set(segmentId, fetchPromise);
    return fetchPromise;
  }

  private clearFullSkeletonCache() {
    this.fullSkeletonCacheGeneration++;
    this.fullSegmentNodeCache.clear();
    this.pendingFullSegmentNodeFetches.clear();
    this.cachedNodesById.clear();
  }

  private rebuildCachedNodesById() {
    this.cachedNodesById.clear();
    for (const segmentNodes of this.fullSegmentNodeCache.values()) {
      for (const node of segmentNodes) {
        if (!this.cachedNodesById.has(node.nodeId)) {
          this.cachedNodesById.set(node.nodeId, node);
        }
      }
    }
  }
}
