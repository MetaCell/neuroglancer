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
  SpatiallyIndexedSkeletonEditContext,
  SpatiallyIndexedSkeletonPropertyEditingOptions,
  SpatiallyIndexedSkeletonNode,
  SpatiallyIndexedSkeletonNodeRevisionUpdate,
  SpatiallyIndexedSkeletonRevisionToken,
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
  rerootSkeletons: boolean;
  editNodeLabels: boolean;
  editNodeProperties: boolean;
  mergeSkeletons: boolean;
  splitSkeletons: boolean;
}

export type SpatiallyIndexedSkeletonSourceCapability =
  keyof SpatiallyIndexedSkeletonSourceCapabilities;

export interface SpatiallyIndexedSkeletonInspectionSource {
  getSkeleton(
    skeletonId: number,
    options?: { signal?: AbortSignal },
  ): Promise<readonly SpatiallyIndexedSkeletonNode[]>;
}

export interface SpatiallyIndexedSkeletonRerootSource {
  rerootSkeleton(
    nodeId: number,
    editContext?: SpatiallyIndexedSkeletonEditContext,
  ): Promise<void>;
}

export interface SpatiallyIndexedSkeletonNodeRevisionLookupSource {
  getNodeRevisionUpdates(
    nodeIds: readonly number[],
  ): Promise<readonly SpatiallyIndexedSkeletonNodeRevisionUpdate[]>;
}

export interface SpatiallyIndexedSkeletonPropertyEditingOptionsSource {
  getPropertyEditingOptions(): SpatiallyIndexedSkeletonPropertyEditingOptions;
}

export const NO_SPATIALLY_INDEXED_SKELETON_SOURCE_CAPABILITIES: SpatiallyIndexedSkeletonSourceCapabilities =
  {
    inspectSkeletons: false,
    addNodes: false,
    moveNodes: false,
    deleteNodes: false,
    rerootSkeletons: false,
    editNodeLabels: false,
    editNodeProperties: false,
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
    capabilities.editNodeProperties &&
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

export function getSpatiallyIndexedSkeletonRerootSource(
  value: SpatialSkeletonSourceAccess | undefined,
): SpatiallyIndexedSkeletonRerootSource | undefined {
  if (value === undefined) return undefined;
  return hasFunction(value.source, "rerootSkeleton")
    ? (value.source as SpatiallyIndexedSkeletonRerootSource)
    : undefined;
}

export function getSpatiallyIndexedSkeletonNodeRevisionLookupSource(
  value: SpatialSkeletonSourceAccess | undefined,
): SpatiallyIndexedSkeletonNodeRevisionLookupSource | undefined {
  if (value === undefined) return undefined;
  return hasFunction(value.source, "getNodeRevisionUpdates")
    ? (value.source as SpatiallyIndexedSkeletonNodeRevisionLookupSource)
    : undefined;
}

export function getSpatiallyIndexedSkeletonPropertyEditingOptions(
  value: SpatialSkeletonSourceAccess | undefined,
): SpatiallyIndexedSkeletonPropertyEditingOptions | undefined {
  if (value === undefined || !hasFunction(value.source, "getPropertyEditingOptions")) {
    return undefined;
  }
  return (
    value.source as SpatiallyIndexedSkeletonPropertyEditingOptionsSource
  ).getPropertyEditingOptions();
}

export function getSpatiallyIndexedSkeletonSourceCapabilities(
  value: unknown,
): SpatiallyIndexedSkeletonSourceCapabilities {
  return {
    inspectSkeletons: hasFunction(value, "getSkeleton"),
    addNodes: hasFunction(value, "addNode"),
    moveNodes: hasFunction(value, "moveNode"),
    deleteNodes: hasFunction(value, "deleteNode"),
    rerootSkeletons: hasFunction(value, "rerootSkeleton"),
    editNodeLabels:
      hasFunction(value, "updateDescription") &&
      hasFunction(value, "setTrueEnd") &&
      hasFunction(value, "removeTrueEnd"),
    editNodeProperties:
      hasFunction(value, "updateRadius") &&
      hasFunction(value, "updateConfidence"),
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
    capabilities.rerootSkeletons ||
    capabilities.editNodeLabels ||
    capabilities.editNodeProperties ||
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
    a.rerootSkeletons === b.rerootSkeletons &&
    a.editNodeLabels === b.editNodeLabels &&
    a.editNodeProperties === b.editNodeProperties &&
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
    radius:
      node.radius === undefined || !Number.isFinite(Number(node.radius))
        ? undefined
        : Number(node.radius),
    confidence:
      node.confidence === undefined || !Number.isFinite(Number(node.confidence))
        ? undefined
        : Number(node.confidence),
    labels: node.labels,
    ...(node.revisionToken === undefined
      ? {}
      : { revisionToken: node.revisionToken }),
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
  readonly mergeAnchorNodeId = new WatchableValue<number | undefined>(
    undefined,
  );
  readonly nodeDataVersion = new WatchableValue(0);
  readonly pendingNodePositionVersion = new WatchableValue(0);

  private pendingNodePositions = new Map<number, Float32Array>();
  private fullSkeletonCacheGeneration = 0;
  private fullSegmentNodeCache = new Map<
    number,
    SpatiallyIndexedSkeletonNodeInfo[]
  >();
  private pendingFullSegmentNodeFetches = new Map<
    number,
    {
      promise: Promise<SpatiallyIndexedSkeletonNodeInfo[]>;
      abortController: AbortController;
    }
  >();
  private cachedNodesById = new Map<number, SpatiallyIndexedSkeletonNodeInfo>();

  setNodeProperties(
    nodeId: number,
    properties: { radius: number; confidence: number },
  ) {
    const normalizedNodeId = this.normalizeNodeId(nodeId);
    const radius = Number(properties.radius);
    const confidence = Number(properties.confidence);
    if (
      normalizedNodeId === undefined ||
      !Number.isFinite(radius) ||
      !Number.isFinite(confidence)
    ) {
      return false;
    }
    return this.updateCachedNode(normalizedNodeId, (node) => {
      if (node.radius === radius && node.confidence === confidence) {
        return node;
      }
      return {
        ...node,
        radius,
        confidence,
      };
    });
  }

  getPendingNodeIds() {
    return this.pendingNodePositions.keys();
  }

  getPendingNodePosition(nodeId: number) {
    return this.pendingNodePositions.get(nodeId);
  }

  private normalizeNodeId(nodeId: number | undefined) {
    if (nodeId === undefined) return undefined;
    const normalizedNodeId = Math.round(Number(nodeId));
    if (!Number.isSafeInteger(normalizedNodeId) || normalizedNodeId <= 0) {
      return undefined;
    }
    return normalizedNodeId;
  }

  setMergeAnchor(nodeId: number | undefined) {
    const normalizedNodeId = this.normalizeNodeId(nodeId);
    if (this.mergeAnchorNodeId.value === normalizedNodeId) {
      return false;
    }
    this.mergeAnchorNodeId.value = normalizedNodeId;
    return true;
  }

  clearMergeAnchor() {
    return this.setMergeAnchor(undefined);
  }

  setPendingNodePosition(nodeId: number, position: ArrayLike<number>) {
    const normalizedNodeId = this.normalizeNodeId(nodeId);
    const x = Number(position[0]);
    const y = Number(position[1]);
    const z = Number(position[2]);
    if (
      normalizedNodeId === undefined ||
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
    const normalizedNodeId = this.normalizeNodeId(nodeId);
    if (
      normalizedNodeId === undefined ||
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

  clearInspectedSkeletonCache() {
    const cacheChanged =
      this.fullSegmentNodeCache.size !== 0 ||
      this.pendingFullSegmentNodeFetches.size !== 0 ||
      this.cachedNodesById.size !== 0;
    const pendingChanged = this.clearPendingNodePositions();
    if (!cacheChanged) {
      return pendingChanged;
    }
    this.clearFullSkeletonCache();
    this.nodeDataVersion.value = this.nodeDataVersion.value + 1;
    return true;
  }

  markNodeDataChanged(options: { invalidateFullSkeletonCache?: boolean } = {}) {
    if (options.invalidateFullSkeletonCache ?? true) {
      this.clearFullSkeletonCache();
    }
    this.nodeDataVersion.value = this.nodeDataVersion.value + 1;
  }

  getCachedSegmentNodes(segmentId: number) {
    return this.fullSegmentNodeCache.get(segmentId);
  }

  getCachedNode(nodeId: number) {
    return this.cachedNodesById.get(nodeId);
  }

  private replaceCachedSegmentNodes(
    segmentId: number,
    nextSegmentNodes: readonly SpatiallyIndexedSkeletonNodeInfo[],
  ) {
    const previousSegmentNodes = this.fullSegmentNodeCache.get(segmentId);
    if (previousSegmentNodes !== undefined) {
      for (const node of previousSegmentNodes) {
        if (this.cachedNodesById.get(node.nodeId) === node) {
          this.cachedNodesById.delete(node.nodeId);
        }
      }
    }
    if (nextSegmentNodes.length === 0) {
      if (previousSegmentNodes === undefined) {
        return false;
      }
      this.fullSegmentNodeCache.delete(segmentId);
      return true;
    }
    const normalizedSegmentNodes = [...nextSegmentNodes];
    this.fullSegmentNodeCache.set(segmentId, normalizedSegmentNodes);
    for (const node of normalizedSegmentNodes) {
      this.cachedNodesById.set(node.nodeId, node);
    }
    return true;
  }

  private deleteCachedSegment(segmentId: number) {
    return this.replaceCachedSegmentNodes(segmentId, []);
  }

  private abortPendingFullSegmentNodeFetch(
    segmentId: number,
    message: string,
  ) {
    const pendingEntry = this.pendingFullSegmentNodeFetches.get(segmentId);
    if (pendingEntry === undefined) {
      return false;
    }
    pendingEntry.abortController.abort(
      new DOMException(message, "AbortError"),
    );
    this.pendingFullSegmentNodeFetches.delete(segmentId);
    return true;
  }

  setCachedNodeRevision(
    nodeId: number,
    revisionToken: SpatiallyIndexedSkeletonRevisionToken | undefined,
  ) {
    if (revisionToken === undefined) {
      return false;
    }
    return this.updateCachedNode(nodeId, (node) => {
      if (node.revisionToken === revisionToken) {
        return node;
      }
      return {
        ...node,
        revisionToken,
      };
    });
  }

  setCachedNodeRevisions(
    revisionUpdates: readonly SpatiallyIndexedSkeletonNodeRevisionUpdate[],
  ) {
    let changed = false;
    for (const update of revisionUpdates) {
      changed =
        this.setCachedNodeRevision(update.nodeId, update.revisionToken) ||
        changed;
    }
    return changed;
  }

  private getCachedSegmentIdForNode(nodeId: number) {
    const normalizedNodeId = this.normalizeNodeId(nodeId);
    if (normalizedNodeId === undefined) {
      return undefined;
    }
    return this.cachedNodesById.get(normalizedNodeId)?.segmentId;
  }

  private updateCachedNodeInSegment(
    segmentId: number,
    nodeId: number,
    update: (
      node: SpatiallyIndexedSkeletonNodeInfo,
    ) => SpatiallyIndexedSkeletonNodeInfo,
  ) {
    const segmentNodes = this.fullSegmentNodeCache.get(segmentId);
    if (segmentNodes === undefined) {
      return false;
    }
    let segmentChanged = false;
    const nextSegmentNodes = segmentNodes.map((candidate) => {
      if (candidate.nodeId !== nodeId) return candidate;
      const updatedNode = update(candidate);
      segmentChanged ||= updatedNode !== candidate;
      return updatedNode;
    });
    if (!segmentChanged) {
      return false;
    }
    this.replaceCachedSegmentNodes(segmentId, nextSegmentNodes);
    return true;
  }

  private upsertCachedNodeInSegment(
    segmentId: number,
    node: SpatiallyIndexedSkeletonNodeInfo,
  ) {
    const segmentNodes = this.fullSegmentNodeCache.get(segmentId);
    if (segmentNodes === undefined) {
      return false;
    }
    const existingIndex = segmentNodes.findIndex(
      (candidate) => candidate.nodeId === node.nodeId,
    );
    if (existingIndex !== -1) {
      const nextSegmentNodes = segmentNodes.slice();
      nextSegmentNodes[existingIndex] = node;
      this.replaceCachedSegmentNodes(segmentId, nextSegmentNodes);
      return true;
    }
    const insertIndex = segmentNodes.findIndex(
      (candidate) => candidate.nodeId > node.nodeId,
    );
    const nextSegmentNodes = segmentNodes.slice();
    nextSegmentNodes.splice(
      insertIndex === -1 ? nextSegmentNodes.length : insertIndex,
      0,
      node,
    );
    this.replaceCachedSegmentNodes(segmentId, nextSegmentNodes);
    return true;
  }

  updateCachedNode(
    nodeId: number,
    update: (
      node: SpatiallyIndexedSkeletonNodeInfo,
    ) => SpatiallyIndexedSkeletonNodeInfo,
  ) {
    const segmentId = this.getCachedSegmentIdForNode(nodeId);
    if (segmentId === undefined) {
      return false;
    }
    return this.updateCachedNodeInSegment(segmentId, nodeId, update);
  }

  upsertCachedNode(
    node: SpatiallyIndexedSkeletonNodeInfo,
    options: { allowUncachedSegment?: boolean } = {},
  ) {
    const normalizedNode = cloneSpatiallyIndexedSkeletonNodeInfo(node);
    const targetSegmentCached = this.fullSegmentNodeCache.has(
      normalizedNode.segmentId,
    );
    const allowUncachedSegment = options.allowUncachedSegment ?? false;
    const existingSegmentId = this.getCachedSegmentIdForNode(
      normalizedNode.nodeId,
    );
    if (!targetSegmentCached && !allowUncachedSegment) {
      return false;
    }
    let changed = false;
    if (
      existingSegmentId !== undefined &&
      existingSegmentId !== normalizedNode.segmentId
    ) {
      const existingSegmentNodes =
        this.fullSegmentNodeCache.get(existingSegmentId);
      if (existingSegmentNodes !== undefined) {
        this.replaceCachedSegmentNodes(
          existingSegmentId,
          existingSegmentNodes.filter(
            (candidate) => candidate.nodeId !== normalizedNode.nodeId,
          ),
        );
        changed = true;
      }
    }
    if (!targetSegmentCached && allowUncachedSegment) {
      this.abortPendingFullSegmentNodeFetch(
        normalizedNode.segmentId,
        "spatial skeleton full-segment inspection request replaced by local segment cache update",
      );
      this.replaceCachedSegmentNodes(normalizedNode.segmentId, [
        normalizedNode,
      ]);
      return true;
    }
    return (
      this.upsertCachedNodeInSegment(normalizedNode.segmentId, normalizedNode) ||
      changed
    );
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
    const normalizedNodeId = this.normalizeNodeId(nodeId);
    if (normalizedNodeId === undefined) {
      return false;
    }
    const childNodeIds = options.childNodeIds
      ? new Set(
          [...options.childNodeIds]
            .map((value) => this.normalizeNodeId(Number(value)))
            .filter((value): value is number => value !== undefined),
        )
      : undefined;
    let segmentId = this.getCachedSegmentIdForNode(normalizedNodeId);
    if (segmentId === undefined && childNodeIds !== undefined) {
      for (const childNodeId of childNodeIds) {
        segmentId = this.getCachedSegmentIdForNode(childNodeId);
        if (segmentId !== undefined) {
          break;
        }
      }
    }
    if (segmentId === undefined) {
      return false;
    }
    const segmentNodes = this.fullSegmentNodeCache.get(segmentId);
    if (segmentNodes === undefined) {
      return false;
    }
    let segmentChanged = false;
    const nextSegmentNodes: SpatiallyIndexedSkeletonNodeInfo[] = [];
    for (const candidate of segmentNodes) {
      if (candidate.nodeId === normalizedNodeId) {
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
    if (!segmentChanged) {
      return false;
    }
    this.replaceCachedSegmentNodes(segmentId, nextSegmentNodes);
    return true;
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

  rerootCachedSegment(
    nodeId: number,
  ) {
    const normalizedNodeId = this.normalizeNodeId(nodeId);
    if (normalizedNodeId === undefined) {
      return undefined;
    }
    const targetNode = this.cachedNodesById.get(normalizedNodeId);
    if (targetNode === undefined) {
      return undefined;
    }
    const segmentNodes = this.fullSegmentNodeCache.get(targetNode.segmentId);
    if (segmentNodes === undefined) {
      return undefined;
    }

    const nodeById = new Map<number, SpatiallyIndexedSkeletonNodeInfo>();
    for (const node of segmentNodes) {
      nodeById.set(node.nodeId, node);
    }
    const startNode = nodeById.get(normalizedNodeId);
    if (startNode === undefined) {
      return undefined;
    }
    if (startNode.parentNodeId === undefined) {
      return [startNode.nodeId];
    }

    const pathNodeIds: number[] = [];
    const seen = new Set<number>();
    let currentNode: SpatiallyIndexedSkeletonNodeInfo | undefined = startNode;
    while (currentNode !== undefined) {
      if (seen.has(currentNode.nodeId)) {
        return undefined;
      }
      seen.add(currentNode.nodeId);
      pathNodeIds.push(currentNode.nodeId);
      const parentNodeId = currentNode.parentNodeId;
      if (parentNodeId === undefined) {
        break;
      }
      currentNode = nodeById.get(parentNodeId);
      if (currentNode === undefined) {
        return undefined;
      }
    }

    const nextParentByNodeId = new Map<number, number | undefined>();
    const nextConfidenceByNodeId = new Map<number, number | undefined>();
    nextParentByNodeId.set(startNode.nodeId, undefined);
    nextConfidenceByNodeId.set(startNode.nodeId, 100);

    let downstreamConfidence = startNode.confidence;
    for (let i = 1; i < pathNodeIds.length; ++i) {
      const upstreamNodeId = pathNodeIds[i];
      const upstreamNode = nodeById.get(upstreamNodeId)!;
      nextParentByNodeId.set(upstreamNodeId, pathNodeIds[i - 1]);
      nextConfidenceByNodeId.set(
        upstreamNodeId,
        downstreamConfidence ?? upstreamNode.confidence,
      );
      downstreamConfidence = upstreamNode.confidence;
    }

    let changed = false;
    const nextSegmentNodes = segmentNodes.map((candidate) => {
      if (!nextParentByNodeId.has(candidate.nodeId)) {
        return candidate;
      }
      const nextParentNodeId = nextParentByNodeId.get(candidate.nodeId);
      const nextConfidence = nextConfidenceByNodeId.get(candidate.nodeId);
      if (
        candidate.parentNodeId === nextParentNodeId &&
        candidate.confidence === nextConfidence
      ) {
        return candidate;
      }
      changed = true;
      return {
        ...candidate,
        parentNodeId: nextParentNodeId,
        confidence: nextConfidence,
      };
    });
    if (!changed) {
      return pathNodeIds;
    }
    this.replaceCachedSegmentNodes(targetNode.segmentId, nextSegmentNodes);
    return pathNodeIds;
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
      changed = this.deleteCachedSegment(normalizedSegmentId) || changed;
      this.abortPendingFullSegmentNodeFetch(
        normalizedSegmentId,
        "spatial skeleton full-segment inspection request invalidated for segment",
      );
    }
    return changed;
  }

  evictInactiveSegmentNodes(activeSegmentIds: Iterable<number>) {
    const activeSegmentIdSet = new Set(activeSegmentIds);
    let changed = false;
    for (const segmentId of this.fullSegmentNodeCache.keys()) {
      if (activeSegmentIdSet.has(segmentId)) continue;
      changed = this.deleteCachedSegment(segmentId) || changed;
    }
    for (const segmentId of [...this.pendingFullSegmentNodeFetches.keys()]) {
      if (activeSegmentIdSet.has(segmentId)) continue;
      this.abortPendingFullSegmentNodeFetch(
        segmentId,
        "spatial skeleton full-segment inspection request evicted for inactive segment",
      );
    }
    return changed;
  }

  async getFullSegmentNodes(
    skeletonLayer: SpatiallyIndexedSkeletonLayer,
    segmentId: number,
  ): Promise<SpatiallyIndexedSkeletonNodeInfo[]> {
    const cached = this.fullSegmentNodeCache.get(segmentId);
    if (cached !== undefined) {
      return cached;
    }
    const pendingEntry = this.pendingFullSegmentNodeFetches.get(segmentId);
    if (pendingEntry !== undefined) {
      return pendingEntry.promise;
    }
    const skeletonSource =
      getSpatiallyIndexedSkeletonInspectionSource(skeletonLayer);
    if (skeletonSource === undefined) {
      throw new Error(
        "The active spatial skeleton source does not expose full skeleton inspection.",
      );
    }
    const fetchVersion = this.fullSkeletonCacheGeneration;
    const abortController = new AbortController();
    const pendingFetch: {
      promise?: Promise<SpatiallyIndexedSkeletonNodeInfo[]>;
    } = {};
    const fetchPromise = (async () => {
      const fetchedNodes = await skeletonSource.getSkeleton(segmentId, {
        signal: abortController.signal,
      });
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
        pendingFetch.promise !== undefined &&
        this.pendingFullSegmentNodeFetches.get(segmentId)?.promise ===
          pendingFetch.promise
      ) {
        this.replaceCachedSegmentNodes(segmentId, normalizedNodes);
        this.markNodeDataChanged({ invalidateFullSkeletonCache: false });
      }
      return normalizedNodes;
    })().finally(() => {
      if (
        this.pendingFullSegmentNodeFetches.get(segmentId)?.promise ===
        pendingFetch.promise
      ) {
        this.pendingFullSegmentNodeFetches.delete(segmentId);
      }
    });
    pendingFetch.promise = fetchPromise;
    this.pendingFullSegmentNodeFetches.set(segmentId, {
      promise: fetchPromise,
      abortController,
    });
    return fetchPromise;
  }

  private clearFullSkeletonCache() {
    this.fullSkeletonCacheGeneration++;
    for (const segmentId of [...this.pendingFullSegmentNodeFetches.keys()]) {
      this.abortPendingFullSegmentNodeFetch(
        segmentId,
        "stale spatial skeleton full-segment inspection request",
      );
    }
    this.fullSegmentNodeCache.clear();
    this.cachedNodesById.clear();
  }
}
