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
  SpatialSkeletonConfidenceConfiguration,
  SpatiallyIndexedSkeletonNode,
  SpatialSkeletonSourceState,
  SpatiallyIndexedSkeletonSource,
} from "#src/skeleton/api.js";
import {
  getSpatialSkeletonEditCommandFactoryFromSource,
  getSpatialSkeletonEditCommandMetadata,
  isSpatialSkeletonEditCommandFactory,
  SPATIAL_SKELETON_EDIT_COMMAND_METADATA,
} from "#src/skeleton/command_factories.js";
import { SpatialSkeletonCommandHistory } from "#src/skeleton/command_history.js";
import type { SpatialSkeletonAction } from "#src/skeleton/command_protocol.js";
import type { SpatiallyIndexedSkeletonLayer } from "#src/skeleton/frontend.js";
import { WatchableValue } from "#src/trackable_value.js";
import { RefCounted } from "#src/util/disposable.js";
import { PromiseConcurrencyLimiter } from "#src/util/promise_concurrency_limiter.js";

interface SpatialSkeletonSourceAccess {
  source: unknown;
}

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

function getProperty<T extends string>(value: unknown, property: T): unknown {
  return typeof value === "object" && value !== null
    ? (value as Record<T, unknown>)[property]
    : undefined;
}

function hasValidCommandFactory(
  value: unknown,
  metadata: (typeof SPATIAL_SKELETON_EDIT_COMMAND_METADATA)[number],
) {
  const commandFactory = getProperty(value, metadata.commandProperty);
  return (
    (commandFactory === undefined && !metadata.required) ||
    isSpatialSkeletonEditCommandFactory(commandFactory, metadata.action)
  );
}

function isFiniteNumberArray(value: unknown): value is readonly number[] {
  return (
    Array.isArray(value) &&
    value.every((entry) => typeof entry === "number" && Number.isFinite(entry))
  );
}

function isSpatialSkeletonConfidenceConfiguration(
  value: unknown,
): value is SpatialSkeletonConfidenceConfiguration {
  return (
    typeof value === "object" &&
    value !== null &&
    isFiniteNumberArray(getProperty(value, "values"))
  );
}

function hasOptionalConfidenceConfiguration(value: unknown) {
  const configuration = getProperty(
    value,
    "spatialSkeletonConfidenceConfiguration",
  );
  return (
    configuration === undefined ||
    isSpatialSkeletonConfidenceConfiguration(configuration)
  );
}

export function isSpatiallyIndexedSkeletonSource(
  value: unknown,
): value is SpatiallyIndexedSkeletonSource {
  return (
    typeof getProperty(value, "readonly") === "boolean" &&
    hasFunction(value, "listSkeletons") &&
    hasFunction(value, "getSkeleton") &&
    hasFunction(value, "getSpatialIndexMetadata") &&
    hasFunction(value, "fetchNodes")
  );
}

export function isEditableSpatiallyIndexedSkeletonSource(
  value: unknown,
): value is EditableSpatiallyIndexedSkeletonSource {
  return (
    isSpatiallyIndexedSkeletonSource(value) &&
    !value.readonly &&
    SPATIAL_SKELETON_EDIT_COMMAND_METADATA.every((metadata) =>
      hasValidCommandFactory(value, metadata),
    ) &&
    hasOptionalConfidenceConfiguration(value)
  );
}

export function getSpatiallyIndexedSkeletonSource(
  value: SpatialSkeletonSourceAccess | undefined,
): SpatiallyIndexedSkeletonSource | undefined {
  if (value === undefined) return undefined;
  return isSpatiallyIndexedSkeletonSource(value.source)
    ? value.source
    : undefined;
}

export function isSpatiallyIndexedSkeletonSourceReadOnly(
  value: SpatialSkeletonSourceAccess | undefined,
): boolean {
  return getSpatiallyIndexedSkeletonSource(value)?.readonly ?? true;
}

export function getEditableSpatiallyIndexedSkeletonSource(
  value: SpatialSkeletonSourceAccess | undefined,
): EditableSpatiallyIndexedSkeletonSource | undefined {
  if (value === undefined) return undefined;
  return isEditableSpatiallyIndexedSkeletonSource(value.source)
    ? value.source
    : undefined;
}

export function getSpatialSkeletonEditCommandFactoryForAction(
  source: EditableSpatiallyIndexedSkeletonSource,
  action: SpatialSkeletonAction,
) {
  return getSpatialSkeletonEditCommandFactoryFromSource(source, action);
}

export function editableSpatiallyIndexedSkeletonSourceSupportsAction(
  source: EditableSpatiallyIndexedSkeletonSource,
  action: SpatialSkeletonAction,
) {
  const commandFactory = getSpatialSkeletonEditCommandFactoryForAction(
    source,
    action,
  );
  if (commandFactory === undefined) return false;
  const metadata = getSpatialSkeletonEditCommandMetadata(action);
  return (
    metadata?.requiresConfidenceConfiguration !== true ||
    source.spatialSkeletonConfidenceConfiguration !== undefined
  );
}

export function normalizeSpatiallyIndexedSkeletonNode(
  node: SpatiallyIndexedSkeletonNode,
  fallbackSegmentId: number,
): SpatiallyIndexedSkeletonNode | undefined {
  const nodeId = Number(node.nodeId);
  const segmentIdValue = Number(node.segmentId);
  const x = Number(node.position[0]);
  const y = Number(node.position[1]);
  const z = Number(node.position[2]);
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
    node.parentNodeId === undefined ||
    !Number.isFinite(Number(node.parentNodeId))
      ? undefined
      : Math.round(Number(node.parentNodeId));
  return {
    ...node,
    nodeId: Math.round(nodeId),
    segmentId: Math.round(
      Number.isFinite(segmentIdValue) ? segmentIdValue : fallbackSegmentId,
    ),
    position: new Float32Array([x, y, z]),
    parentNodeId,
    description:
      typeof node.description === "string" && node.description.length > 0
        ? node.description
        : undefined,
    isTrueEnd: node.isTrueEnd ?? false,
    ...((node.radius !== undefined && Number.isFinite(Number(node.radius))) ||
    (node.confidence !== undefined && Number.isFinite(Number(node.confidence)))
      ? {
          ...(node.radius !== undefined && Number.isFinite(Number(node.radius))
            ? { radius: Number(node.radius) }
            : {}),
          ...(node.confidence !== undefined &&
          Number.isFinite(Number(node.confidence))
            ? { confidence: Number(node.confidence) }
            : {}),
        }
      : {}),
    ...(node.sourceState === undefined
      ? {}
      : { sourceState: node.sourceState }),
  };
}

function cloneSpatiallyIndexedSkeletonNode(
  node: SpatiallyIndexedSkeletonNode,
): SpatiallyIndexedSkeletonNode {
  return {
    ...node,
    position: new Float32Array(node.position),
  };
}

/**
 * Full-segment skeleton fetches bypass the chunk queue manager, so they are
 * capped separately at min(this, the concurrentDownloads viewer setting).
 */
const MAX_CONCURRENT_FULL_SEGMENT_NODE_FETCHES = 8;

interface FullSegmentNodeFetch {
  promise: Promise<SpatiallyIndexedSkeletonNode[]>;
  abortController: AbortController;
}

interface FullSkeletonCacheScope {
  readonly skeletonLayer: SpatiallyIndexedSkeletonLayer | undefined;
  readonly source: SpatiallyIndexedSkeletonSource | undefined;
  generation: number;
  readonly segmentNodes: Map<number, SpatiallyIndexedSkeletonNode[]>;
  readonly pendingSegmentNodeFetches: Map<number, FullSegmentNodeFetch>;
  readonly nodesById: Map<number, SpatiallyIndexedSkeletonNode>;
}

function makeFullSkeletonCacheScope(
  skeletonLayer?: SpatiallyIndexedSkeletonLayer,
  source?: SpatiallyIndexedSkeletonSource,
): FullSkeletonCacheScope {
  return {
    skeletonLayer,
    source,
    generation: 0,
    segmentNodes: new Map(),
    pendingSegmentNodeFetches: new Map(),
    nodesById: new Map(),
  };
}

export class SpatialSkeletonState extends RefCounted {
  readonly commandHistory = this.registerDisposer(
    new SpatialSkeletonCommandHistory(),
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
  private readonly legacyFullSkeletonCacheScope = makeFullSkeletonCacheScope();
  private readonly fullSkeletonCacheScopes = new Set<FullSkeletonCacheScope>([
    this.legacyFullSkeletonCacheScope,
  ]);
  private readonly fullSkeletonCacheScopeByLayer = new WeakMap<
    SpatiallyIndexedSkeletonLayer,
    FullSkeletonCacheScope
  >();
  private fullSegmentNodeFetchLimitLayer:
    | SpatiallyIndexedSkeletonLayer
    | undefined;
  private fullSegmentNodeFetchLimiter = new PromiseConcurrencyLimiter(() => {
    const itemLimit =
      this.fullSegmentNodeFetchLimitLayer?.chunkManager?.chunkQueueManager
        ?.capacities?.download?.itemLimit?.value;
    return Math.min(
      MAX_CONCURRENT_FULL_SEGMENT_NODE_FETCHES,
      itemLimit ?? Number.POSITIVE_INFINITY,
    );
  });

  setNodeRadius(
    nodeId: number,
    radius: number,
    skeletonLayer?: SpatiallyIndexedSkeletonLayer,
  ) {
    const normalizedNodeId = this.normalizeNodeId(nodeId);
    radius = Number(radius);
    if (normalizedNodeId === undefined || !Number.isFinite(radius)) {
      return false;
    }
    return this.updateCachedNode(
      normalizedNodeId,
      (node) => {
        if (node.radius === radius) {
          return node;
        }
        return {
          ...node,
          radius,
        };
      },
      skeletonLayer,
    );
  }

  setNodeConfidence(
    nodeId: number,
    confidence: number,
    skeletonLayer?: SpatiallyIndexedSkeletonLayer,
  ) {
    const normalizedNodeId = this.normalizeNodeId(nodeId);
    confidence = Number(confidence);
    if (normalizedNodeId === undefined || !Number.isFinite(confidence)) {
      return false;
    }
    return this.updateCachedNode(
      normalizedNodeId,
      (node) => {
        if (node.confidence === confidence) {
          return node;
        }
        return {
          ...node,
          confidence,
        };
      },
      skeletonLayer,
    );
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

  updateCommandHistorySource(source: unknown) {
    return this.commandHistory.setSource(source);
  }

  private fullSkeletonCacheScopeHasState(scope: FullSkeletonCacheScope) {
    return (
      scope.segmentNodes.size !== 0 ||
      scope.pendingSegmentNodeFetches.size !== 0 ||
      scope.nodesById.size !== 0
    );
  }

  private clearFullSkeletonCacheScope(
    scope: FullSkeletonCacheScope,
    message: string,
  ) {
    const changed = this.fullSkeletonCacheScopeHasState(scope);
    scope.generation++;
    for (const segmentId of scope.pendingSegmentNodeFetches.keys()) {
      this.abortPendingFullSegmentNodeFetch(scope, segmentId, message);
    }
    scope.segmentNodes.clear();
    scope.nodesById.clear();
    return changed;
  }

  private removeFullSkeletonCacheScope(
    scope: FullSkeletonCacheScope,
    message: string,
  ) {
    this.fullSkeletonCacheScopes.delete(scope);
    if (scope.skeletonLayer !== undefined) {
      this.fullSkeletonCacheScopeByLayer.delete(scope.skeletonLayer);
    }
    return this.clearFullSkeletonCacheScope(scope, message);
  }

  private getFullSkeletonCacheScopeForLayer(
    skeletonLayer: SpatiallyIndexedSkeletonLayer,
    create: boolean,
  ) {
    const source = getSpatiallyIndexedSkeletonSource(skeletonLayer);
    const existing = this.fullSkeletonCacheScopeByLayer.get(skeletonLayer);
    if (existing !== undefined && existing.source !== source) {
      if (
        this.removeFullSkeletonCacheScope(
          existing,
          "spatial skeleton source replaced during full-segment inspection",
        )
      ) {
        this.nodeDataVersion.value = this.nodeDataVersion.value + 1;
      }
    } else if (existing !== undefined) {
      return existing;
    }
    if (source === undefined || !create) {
      return undefined;
    }
    const scope = makeFullSkeletonCacheScope(skeletonLayer, source);
    this.fullSkeletonCacheScopes.add(scope);
    this.fullSkeletonCacheScopeByLayer.set(skeletonLayer, scope);
    return scope;
  }

  private getUniqueCacheScope(
    predicate: (scope: FullSkeletonCacheScope) => boolean,
  ) {
    let match: FullSkeletonCacheScope | undefined;
    for (const scope of this.fullSkeletonCacheScopes) {
      if (!predicate(scope)) continue;
      if (match !== undefined) return undefined;
      match = scope;
    }
    return match;
  }

  private getCacheScopeForSegment(
    segmentId: number,
    skeletonLayer?: SpatiallyIndexedSkeletonLayer,
  ) {
    if (skeletonLayer !== undefined) {
      return this.getFullSkeletonCacheScopeForLayer(skeletonLayer, false);
    }
    return this.getUniqueCacheScope(
      (scope) =>
        scope.segmentNodes.has(segmentId) ||
        scope.pendingSegmentNodeFetches.has(segmentId),
    );
  }

  private getCacheScopeForNode(
    nodeId: number,
    skeletonLayer?: SpatiallyIndexedSkeletonLayer,
  ) {
    if (skeletonLayer !== undefined) {
      return this.getFullSkeletonCacheScopeForLayer(skeletonLayer, false);
    }
    return this.getUniqueCacheScope((scope) => scope.nodesById.has(nodeId));
  }

  clearInspectedSkeletonCache() {
    const cacheChanged = [...this.fullSkeletonCacheScopes].some((scope) =>
      this.fullSkeletonCacheScopeHasState(scope),
    );
    const pendingChanged = this.clearPendingNodePositions();
    if (!cacheChanged) {
      return pendingChanged;
    }
    this.clearFullSkeletonCache();
    this.nodeDataVersion.value = this.nodeDataVersion.value + 1;
    return true;
  }

  releaseSkeletonLayerCache(skeletonLayer: SpatiallyIndexedSkeletonLayer) {
    const scope = this.fullSkeletonCacheScopeByLayer.get(skeletonLayer);
    if (scope === undefined) return false;
    return this.removeFullSkeletonCacheScope(
      scope,
      "spatial skeleton layer disposed during full-segment inspection",
    );
  }

  registerSkeletonLayerCache(skeletonLayer: SpatiallyIndexedSkeletonLayer) {
    return (
      this.getFullSkeletonCacheScopeForLayer(skeletonLayer, true) !== undefined
    );
  }

  markNodeDataChanged(options: { invalidateFullSkeletonCache?: boolean } = {}) {
    if (options.invalidateFullSkeletonCache ?? true) {
      this.clearFullSkeletonCache();
    }
    this.nodeDataVersion.value = this.nodeDataVersion.value + 1;
  }

  getCachedSegmentNodes(
    segmentId: number,
    skeletonLayer?: SpatiallyIndexedSkeletonLayer,
  ) {
    return this.getCacheScopeForSegment(
      segmentId,
      skeletonLayer,
    )?.segmentNodes.get(segmentId);
  }

  getCachedNode(nodeId: number, skeletonLayer?: SpatiallyIndexedSkeletonLayer) {
    return this.getCacheScopeForNode(nodeId, skeletonLayer)?.nodesById.get(
      nodeId,
    );
  }

  private replaceCachedSegmentNodesInScope(
    scope: FullSkeletonCacheScope,
    segmentId: number,
    nextSegmentNodes: readonly SpatiallyIndexedSkeletonNode[],
  ) {
    const previousSegmentNodes = scope.segmentNodes.get(segmentId);
    if (previousSegmentNodes !== undefined) {
      for (const node of previousSegmentNodes) {
        if (scope.nodesById.get(node.nodeId) === node) {
          scope.nodesById.delete(node.nodeId);
        }
      }
    }
    if (nextSegmentNodes.length === 0) {
      if (previousSegmentNodes === undefined) {
        // No previous entry, assume this is an empty segment
        scope.segmentNodes.set(segmentId, []);
      } else {
        // Previous entry exists, this is a segment being cleared
        scope.segmentNodes.delete(segmentId);
      }
      return true;
    }
    const normalizedSegmentNodes = [...nextSegmentNodes];
    scope.segmentNodes.set(segmentId, normalizedSegmentNodes);
    for (const node of normalizedSegmentNodes) {
      scope.nodesById.set(node.nodeId, node);
    }
    return true;
  }

  // Retained for legacy cache mutation paths and focused cache tests. New
  // source-aware callers should supply a skeleton layer to the public APIs.
  private replaceCachedSegmentNodes(
    segmentId: number,
    nextSegmentNodes: readonly SpatiallyIndexedSkeletonNode[],
  ) {
    const scope =
      this.getCacheScopeForSegment(segmentId) ??
      this.legacyFullSkeletonCacheScope;
    return this.replaceCachedSegmentNodesInScope(
      scope,
      segmentId,
      nextSegmentNodes,
    );
  }

  private deleteCachedSegment(
    scope: FullSkeletonCacheScope,
    segmentId: number,
  ) {
    const previousSegmentNodes = scope.segmentNodes.get(segmentId);
    if (previousSegmentNodes === undefined) return false;
    for (const node of previousSegmentNodes) {
      if (scope.nodesById.get(node.nodeId) === node) {
        scope.nodesById.delete(node.nodeId);
      }
    }

    scope.segmentNodes.delete(segmentId);
    return true;
  }

  private abortPendingFullSegmentNodeFetch(
    scope: FullSkeletonCacheScope,
    segmentId: number,
    message: string,
  ) {
    const pendingEntry = scope.pendingSegmentNodeFetches.get(segmentId);
    if (pendingEntry === undefined) {
      return false;
    }
    scope.pendingSegmentNodeFetches.delete(segmentId);
    pendingEntry.abortController.abort(new DOMException(message, "AbortError"));
    return true;
  }

  setCachedNodeSourceState(
    nodeId: number,
    sourceState: SpatialSkeletonSourceState | undefined,
    skeletonLayer?: SpatiallyIndexedSkeletonLayer,
  ) {
    if (sourceState === undefined) {
      return false;
    }
    return this.updateCachedNode(
      nodeId,
      (node) => {
        if (node.sourceState === sourceState) {
          return node;
        }
        return {
          ...node,
          sourceState,
        };
      },
      skeletonLayer,
    );
  }

  setCachedNodeSourceStates(
    sourceStateUpdates: readonly {
      nodeId: number;
      sourceState: SpatialSkeletonSourceState;
    }[],
    skeletonLayer?: SpatiallyIndexedSkeletonLayer,
  ) {
    let changed = false;
    for (const update of sourceStateUpdates) {
      changed =
        this.setCachedNodeSourceState(
          update.nodeId,
          update.sourceState,
          skeletonLayer,
        ) || changed;
    }
    return changed;
  }

  private updateCachedNodeInSegment(
    scope: FullSkeletonCacheScope,
    segmentId: number,
    nodeId: number,
    update: (
      node: SpatiallyIndexedSkeletonNode,
    ) => SpatiallyIndexedSkeletonNode,
  ) {
    const segmentNodes = scope.segmentNodes.get(segmentId);
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
    this.replaceCachedSegmentNodesInScope(scope, segmentId, nextSegmentNodes);
    return true;
  }

  private upsertCachedNodeInSegment(
    scope: FullSkeletonCacheScope,
    segmentId: number,
    node: SpatiallyIndexedSkeletonNode,
  ) {
    const segmentNodes = scope.segmentNodes.get(segmentId);
    if (segmentNodes === undefined) {
      return false;
    }
    const existingIndex = segmentNodes.findIndex(
      (candidate) => candidate.nodeId === node.nodeId,
    );
    if (existingIndex !== -1) {
      const nextSegmentNodes = segmentNodes.slice();
      nextSegmentNodes[existingIndex] = node;
      this.replaceCachedSegmentNodesInScope(scope, segmentId, nextSegmentNodes);
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
    this.replaceCachedSegmentNodesInScope(scope, segmentId, nextSegmentNodes);
    return true;
  }

  updateCachedNode(
    nodeId: number,
    update: (
      node: SpatiallyIndexedSkeletonNode,
    ) => SpatiallyIndexedSkeletonNode,
    skeletonLayer?: SpatiallyIndexedSkeletonLayer,
  ) {
    const normalizedNodeId = this.normalizeNodeId(nodeId);
    if (normalizedNodeId === undefined) {
      return false;
    }
    const scope = this.getCacheScopeForNode(normalizedNodeId, skeletonLayer);
    const segmentId = scope?.nodesById.get(normalizedNodeId)?.segmentId;
    if (scope === undefined || segmentId === undefined) {
      return false;
    }
    return this.updateCachedNodeInSegment(
      scope,
      segmentId,
      normalizedNodeId,
      update,
    );
  }

  upsertCachedNode(
    node: SpatiallyIndexedSkeletonNode,
    options: {
      allowUncachedSegment?: boolean;
      skeletonLayer?: SpatiallyIndexedSkeletonLayer;
    } = {},
  ) {
    const normalizedNode = cloneSpatiallyIndexedSkeletonNode(node);
    const allowUncachedSegment = options.allowUncachedSegment ?? false;
    const existingScope = this.getCacheScopeForNode(
      normalizedNode.nodeId,
      options.skeletonLayer,
    );
    let targetScope = this.getCacheScopeForSegment(
      normalizedNode.segmentId,
      options.skeletonLayer,
    );
    targetScope ??= existingScope;
    if (
      targetScope === undefined &&
      allowUncachedSegment &&
      options.skeletonLayer !== undefined
    ) {
      targetScope = this.getFullSkeletonCacheScopeForLayer(
        options.skeletonLayer,
        true,
      );
    }
    if (
      targetScope === undefined &&
      allowUncachedSegment &&
      options.skeletonLayer === undefined
    ) {
      const hasConcreteScope = [...this.fullSkeletonCacheScopes].some(
        (scope) => scope.skeletonLayer !== undefined,
      );
      targetScope = this.getUniqueCacheScope(
        (scope) => scope.skeletonLayer !== undefined,
      );
      if (targetScope === undefined && hasConcreteScope) {
        return false;
      }
    }
    targetScope ??= allowUncachedSegment
      ? this.legacyFullSkeletonCacheScope
      : undefined;
    if (targetScope === undefined) {
      return false;
    }
    const targetSegmentCached = targetScope.segmentNodes.has(
      normalizedNode.segmentId,
    );
    const existingSegmentId = targetScope.nodesById.get(
      normalizedNode.nodeId,
    )?.segmentId;
    if (!targetSegmentCached && !allowUncachedSegment) {
      return false;
    }
    let changed = false;
    if (
      existingSegmentId !== undefined &&
      existingSegmentId !== normalizedNode.segmentId
    ) {
      const existingSegmentNodes =
        targetScope.segmentNodes.get(existingSegmentId);
      if (existingSegmentNodes !== undefined) {
        this.replaceCachedSegmentNodesInScope(
          targetScope,
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
        targetScope,
        normalizedNode.segmentId,
        "spatial skeleton full-segment inspection request replaced by local segment cache update",
      );
      if (targetScope === this.legacyFullSkeletonCacheScope) {
        this.replaceCachedSegmentNodes(normalizedNode.segmentId, [
          normalizedNode,
        ]);
      } else {
        this.replaceCachedSegmentNodesInScope(
          targetScope,
          normalizedNode.segmentId,
          [normalizedNode],
        );
      }
      return true;
    }
    return (
      this.upsertCachedNodeInSegment(
        targetScope,
        normalizedNode.segmentId,
        normalizedNode,
      ) || changed
    );
  }

  moveCachedNode(
    nodeId: number,
    position: ArrayLike<number>,
    skeletonLayer?: SpatiallyIndexedSkeletonLayer,
  ) {
    const x = Number(position[0]);
    const y = Number(position[1]);
    const z = Number(position[2]);
    if (!Number.isFinite(x) || !Number.isFinite(y) || !Number.isFinite(z)) {
      return false;
    }
    return this.updateCachedNode(
      nodeId,
      (node) => {
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
      },
      skeletonLayer,
    );
  }

  removeCachedNode(
    nodeId: number,
    options: {
      parentNodeId?: number;
      childNodeIds?: Iterable<number>;
      skeletonLayer?: SpatiallyIndexedSkeletonLayer;
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
    let scope = this.getCacheScopeForNode(
      normalizedNodeId,
      options.skeletonLayer,
    );
    let segmentId = scope?.nodesById.get(normalizedNodeId)?.segmentId;
    if (segmentId === undefined && childNodeIds !== undefined) {
      for (const childNodeId of childNodeIds) {
        scope = this.getCacheScopeForNode(childNodeId, options.skeletonLayer);
        segmentId = scope?.nodesById.get(childNodeId)?.segmentId;
        if (scope !== undefined && segmentId !== undefined) {
          break;
        }
      }
    }
    if (scope === undefined || segmentId === undefined) {
      return false;
    }
    const segmentNodes = scope.segmentNodes.get(segmentId);
    if (segmentNodes === undefined) {
      return false;
    }
    let segmentChanged = false;
    const nextSegmentNodes: SpatiallyIndexedSkeletonNode[] = [];
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
    this.replaceCachedSegmentNodesInScope(scope, segmentId, nextSegmentNodes);
    return true;
  }

  setCachedNodeParent(
    nodeId: number,
    parentNodeId: number | undefined,
    skeletonLayer?: SpatiallyIndexedSkeletonLayer,
  ) {
    return this.updateCachedNode(
      nodeId,
      (node) => {
        if (node.parentNodeId === parentNodeId) {
          return node;
        }
        return {
          ...node,
          parentNodeId,
        };
      },
      skeletonLayer,
    );
  }

  rerootCachedSegment(
    nodeId: number,
    skeletonLayer?: SpatiallyIndexedSkeletonLayer,
  ) {
    const normalizedNodeId = this.normalizeNodeId(nodeId);
    if (normalizedNodeId === undefined) {
      return undefined;
    }
    const scope = this.getCacheScopeForNode(normalizedNodeId, skeletonLayer);
    const targetNode = scope?.nodesById.get(normalizedNodeId);
    if (scope === undefined || targetNode === undefined) {
      return undefined;
    }
    const segmentNodes = scope.segmentNodes.get(targetNode.segmentId);
    if (segmentNodes === undefined) {
      return undefined;
    }

    const nodeById = new Map<number, SpatiallyIndexedSkeletonNode>();
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
    let currentNode: SpatiallyIndexedSkeletonNode | undefined = startNode;
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
    this.replaceCachedSegmentNodesInScope(
      scope,
      targetNode.segmentId,
      nextSegmentNodes,
    );
    return pathNodeIds;
  }

  invalidateCachedSegments(
    segmentIds: Iterable<number>,
    skeletonLayer?: SpatiallyIndexedSkeletonLayer,
  ) {
    let changed = false;
    for (const segmentId of segmentIds) {
      const normalizedSegmentId = Math.round(Number(segmentId));
      if (
        !Number.isSafeInteger(normalizedSegmentId) ||
        normalizedSegmentId <= 0
      ) {
        continue;
      }
      const scopes = [
        this.getCacheScopeForSegment(normalizedSegmentId, skeletonLayer),
      ].filter((scope): scope is FullSkeletonCacheScope => scope !== undefined);
      for (const scope of scopes) {
        changed =
          this.deleteCachedSegment(scope, normalizedSegmentId) || changed;
        this.abortPendingFullSegmentNodeFetch(
          scope,
          normalizedSegmentId,
          "spatial skeleton full-segment inspection request invalidated for segment",
        );
      }
    }
    return changed;
  }

  evictInactiveSegmentNodes(
    activeSegmentIds: Iterable<number>,
    skeletonLayer?: SpatiallyIndexedSkeletonLayer,
  ) {
    const activeSegmentIdSet = new Set(activeSegmentIds);
    const scopes = [
      skeletonLayer === undefined
        ? this.getUniqueCacheScope((scope) =>
            this.fullSkeletonCacheScopeHasState(scope),
          )
        : this.getFullSkeletonCacheScopeForLayer(skeletonLayer, false),
    ].filter((scope): scope is FullSkeletonCacheScope => scope !== undefined);
    let changed = false;
    for (const scope of scopes) {
      for (const segmentId of scope.segmentNodes.keys()) {
        if (activeSegmentIdSet.has(segmentId)) continue;
        changed = this.deleteCachedSegment(scope, segmentId) || changed;
      }
      for (const segmentId of scope.pendingSegmentNodeFetches.keys()) {
        if (activeSegmentIdSet.has(segmentId)) continue;
        this.abortPendingFullSegmentNodeFetch(
          scope,
          segmentId,
          "spatial skeleton full-segment inspection request evicted for inactive segment",
        );
      }
    }
    return changed;
  }

  async getFullSegmentNodes(
    skeletonLayer: SpatiallyIndexedSkeletonLayer,
    segmentId: number,
  ): Promise<SpatiallyIndexedSkeletonNode[]> {
    const scope = this.getFullSkeletonCacheScopeForLayer(skeletonLayer, true);
    const skeletonSource = scope?.source;
    if (scope === undefined || skeletonSource === undefined) {
      throw new Error(
        "The active spatial skeleton source does not expose full skeleton inspection.",
      );
    }
    const cached = scope.segmentNodes.get(segmentId);
    if (cached !== undefined) {
      return cached;
    }
    const pendingEntry = scope.pendingSegmentNodeFetches.get(segmentId);
    if (pendingEntry !== undefined) {
      return pendingEntry.promise;
    }
    const fetchVersion = scope.generation;
    const abortController = new AbortController();
    const pendingFetch: {
      promise?: Promise<SpatiallyIndexedSkeletonNode[]>;
    } = {};
    this.fullSegmentNodeFetchLimitLayer = skeletonLayer;
    const fetchPromise = this.fullSegmentNodeFetchLimiter
      .run(
        async () => {
          const fetchedNodes = await skeletonSource.getSkeleton(segmentId, {
            signal: abortController.signal,
          });
          const normalizedNodes: SpatiallyIndexedSkeletonNode[] = [];
          for (const fetchedNode of fetchedNodes) {
            const mappedNode = normalizeSpatiallyIndexedSkeletonNode(
              fetchedNode,
              segmentId,
            );
            if (mappedNode === undefined) continue;
            normalizedNodes.push(mappedNode);
          }
          normalizedNodes.sort((a, b) => a.nodeId - b.nodeId);
          if (
            scope.generation === fetchVersion &&
            this.fullSkeletonCacheScopeByLayer.get(skeletonLayer) === scope &&
            pendingFetch.promise !== undefined &&
            scope.pendingSegmentNodeFetches.get(segmentId)?.promise ===
              pendingFetch.promise
          ) {
            this.replaceCachedSegmentNodesInScope(
              scope,
              segmentId,
              normalizedNodes,
            );
            this.markNodeDataChanged({ invalidateFullSkeletonCache: false });
          }
          return normalizedNodes;
        },
        { signal: abortController.signal },
      )
      .finally(() => {
        if (
          scope.pendingSegmentNodeFetches.get(segmentId)?.promise ===
          pendingFetch.promise
        ) {
          scope.pendingSegmentNodeFetches.delete(segmentId);
        }
      });
    pendingFetch.promise = fetchPromise;
    scope.pendingSegmentNodeFetches.set(segmentId, {
      promise: fetchPromise,
      abortController,
    });
    return fetchPromise;
  }

  private clearFullSkeletonCache() {
    for (const scope of this.fullSkeletonCacheScopes) {
      this.clearFullSkeletonCacheScope(
        scope,
        "stale spatial skeleton full-segment inspection request",
      );
    }
  }
}

export interface SpatialSkeletonLayerContext {
  getSpatiallyIndexedSkeletonLayer(): SpatiallyIndexedSkeletonLayer | undefined;
  readonly spatialSkeletonState: SpatialSkeletonState;
}
