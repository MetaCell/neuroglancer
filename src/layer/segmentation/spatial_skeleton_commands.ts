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

import type { SegmentationUserLayer } from "#src/layer/segmentation/index.js";
import {
  addSegmentToVisibleSets,
  removeSegmentFromVisibleSets,
} from "#src/segmentation_display_state/base.js";
import type {
  EditableSpatiallyIndexedSkeletonSource,
  SpatiallyIndexedSkeletonAddNodeResult,
  SpatiallyIndexedSkeletonEditContext,
  SpatiallyIndexedSkeletonInsertNodeResult,
  SpatiallyIndexedSkeletonMergeResult,
  SpatiallyIndexedSkeletonNodeRevisionUpdate,
  SpatiallyIndexedSkeletonSplitResult,
} from "#src/skeleton/api.js";
import {
  SpatialSkeletonCommand,
  type SpatialSkeletonCommandContext,
} from "#src/skeleton/command_history.js";
import {
  buildSpatiallyIndexedSkeletonMultiNodeEditContext,
  buildSpatiallyIndexedSkeletonNeighborhoodEditContext,
  buildSpatiallyIndexedSkeletonNodeEditContext,
  findSpatiallyIndexedSkeletonNodeInfo,
  getSpatiallyIndexedSkeletonDirectChildren,
} from "#src/skeleton/edit_state.js";
import type {
  SpatiallyIndexedSkeletonLayer,
  SpatiallyIndexedSkeletonNodeInfo,
} from "#src/skeleton/frontend.js";
import {
  hasSpatialSkeletonTrueEndLabel,
  isSpatialSkeletonClosedEndLabel,
  updateSpatialSkeletonTrueEndLabels,
} from "#src/skeleton/node_types.js";
import {
  getEditableSpatiallyIndexedSkeletonSource,
  getSpatiallyIndexedSkeletonNodeRevisionLookupSource,
} from "#src/skeleton/state.js";
import { StatusMessage } from "#src/status.js";

type CachedNodeTarget = Pick<
  SpatiallyIndexedSkeletonNodeInfo,
  "nodeId" | "segmentId" | "position"
> & {
  parentNodeId?: number;
  radius?: number;
  confidence?: number;
  labels?: readonly string[];
  revisionToken?: string | number;
};

interface DeleteOperationContext {
  node: SpatiallyIndexedSkeletonNodeInfo;
  parentNode: SpatiallyIndexedSkeletonNodeInfo | undefined;
  childNodes: readonly SpatiallyIndexedSkeletonNodeInfo[];
  editContext: SpatiallyIndexedSkeletonEditContext;
}

interface SpatialSkeletonResolvedEditNode {
  skeletonLayer: SpatiallyIndexedSkeletonLayer;
  skeletonSource: EditableSpatiallyIndexedSkeletonSource;
  segmentNodes: readonly SpatiallyIndexedSkeletonNodeInfo[];
  node: SpatiallyIndexedSkeletonNodeInfo;
}

function normalizeSpatialSkeletonLabel(label: string) {
  return label.trim().toLowerCase();
}

function getSpatialSkeletonDescriptionLabels(
  labels: readonly string[] | undefined,
) {
  const result: string[] = [];
  const seen = new Set<string>();
  for (const label of labels ?? []) {
    const trimmed = label.trim();
    if (trimmed.length === 0 || isSpatialSkeletonClosedEndLabel(trimmed)) {
      continue;
    }
    const key = normalizeSpatialSkeletonLabel(trimmed);
    if (seen.has(key)) continue;
    seen.add(key);
    result.push(trimmed);
  }
  return result;
}

function mergeSpatialSkeletonNodeLabels(
  labels: readonly string[] | undefined,
  descriptionLabels: readonly string[],
) {
  const result: string[] = [];
  const seen = new Set<string>();
  for (const label of labels ?? []) {
    const trimmed = label.trim();
    if (trimmed.length === 0 || !isSpatialSkeletonClosedEndLabel(trimmed)) {
      continue;
    }
    const key = normalizeSpatialSkeletonLabel(trimmed);
    if (seen.has(key)) continue;
    seen.add(key);
    result.push(trimmed);
  }
  for (const label of descriptionLabels) {
    const trimmed = label.trim();
    if (trimmed.length === 0) continue;
    const key = normalizeSpatialSkeletonLabel(trimmed);
    if (seen.has(key)) continue;
    seen.add(key);
    result.push(trimmed);
  }
  return result.length === 0 ? undefined : result;
}

function spatialSkeletonLabelListsEqual(
  a: readonly string[] | undefined,
  b: readonly string[] | undefined,
) {
  if (a === b) return true;
  if (a === undefined || b === undefined) {
    return a === undefined && b === undefined;
  }
  if (a.length !== b.length) return false;
  for (let i = 0; i < a.length; ++i) {
    if (a[i] !== b[i]) return false;
  }
  return true;
}

function cloneNodeSnapshot(
  node: Pick<
    SpatiallyIndexedSkeletonNodeInfo,
    | "nodeId"
    | "segmentId"
    | "position"
    | "parentNodeId"
    | "radius"
    | "confidence"
    | "labels"
    | "revisionToken"
  >,
): CachedNodeTarget {
  return {
    nodeId: node.nodeId,
    segmentId: node.segmentId,
    position: new Float32Array(node.position),
    parentNodeId: node.parentNodeId,
    radius: node.radius,
    confidence: node.confidence,
    labels: node.labels === undefined ? undefined : [...node.labels],
    revisionToken: node.revisionToken,
  };
}

function normalizeStableNodeId(
  layer: SegmentationUserLayer,
  nodeId: number | undefined,
) {
  if (nodeId === undefined) {
    return undefined;
  }
  return (
    layer.spatialSkeletonState.commandHistory.mappings.getStableNodeId(nodeId) ??
    nodeId
  );
}

function normalizeStableSegmentId(
  layer: SegmentationUserLayer,
  segmentId: number | undefined,
) {
  if (segmentId === undefined) {
    return undefined;
  }
  return (
    layer.spatialSkeletonState.commandHistory.mappings.getStableSegmentId(
      segmentId,
    ) ?? segmentId
  );
}

function getEditableSkeletonSourceForLayer(
  layer: SegmentationUserLayer,
): {
  skeletonLayer: SpatiallyIndexedSkeletonLayer;
  skeletonSource: EditableSpatiallyIndexedSkeletonSource;
} {
  const skeletonLayer = layer.getSpatiallyIndexedSkeletonLayer();
  if (skeletonLayer === undefined) {
    throw new Error(
      "No spatially indexed skeleton source is currently loaded.",
    );
  }
  const skeletonSource = getEditableSpatiallyIndexedSkeletonSource(skeletonLayer);
  if (skeletonSource === undefined) {
    throw new Error(
      "Unable to resolve editable skeleton source for the active layer.",
    );
  }
  return { skeletonLayer, skeletonSource };
}

function ensureVisibleSegment(
  layer: SegmentationUserLayer,
  segmentId: number | undefined,
) {
  if (
    segmentId === undefined ||
    !Number.isSafeInteger(Math.round(Number(segmentId))) ||
    Math.round(Number(segmentId)) <= 0
  ) {
    return;
  }
  addSegmentToVisibleSets(
    layer.displayState.segmentationGroupState.value,
    BigInt(Math.round(Number(segmentId))),
  );
}

function selectSegment(
  layer: SegmentationUserLayer,
  segmentId: number | undefined,
  pin: boolean,
) {
  if (
    segmentId === undefined ||
    !Number.isSafeInteger(Math.round(Number(segmentId))) ||
    Math.round(Number(segmentId)) <= 0
  ) {
    return;
  }
  layer.selectSegment(BigInt(Math.round(Number(segmentId))), pin);
}

function removeVisibleSegment(
  layer: SegmentationUserLayer,
  segmentId: number | undefined,
  options: {
    deselect?: boolean;
  } = {},
) {
  if (
    segmentId === undefined ||
    !Number.isSafeInteger(Math.round(Number(segmentId))) ||
    Math.round(Number(segmentId)) <= 0
  ) {
    return;
  }
  removeSegmentFromVisibleSets(
    layer.displayState.segmentationGroupState.value,
    BigInt(Math.round(Number(segmentId))),
    options,
  );
}

function findRootNode(
  segmentNodes: readonly SpatiallyIndexedSkeletonNodeInfo[],
) {
  return segmentNodes.find((candidate) => candidate.parentNodeId === undefined);
}

async function getResolvedNodeForEdit(
  layer: SegmentationUserLayer,
  stableNodeId: number,
  stableSegmentId: number | undefined,
): Promise<SpatialSkeletonResolvedEditNode> {
  const currentNodeId =
    layer.spatialSkeletonState.commandHistory.mappings.resolveNodeId(
      stableNodeId,
    );
  if (currentNodeId === undefined) {
    throw new Error(`Unable to resolve current node ${stableNodeId}.`);
  }
  const { skeletonLayer, skeletonSource } = getEditableSkeletonSourceForLayer(
    layer,
  );
  const cachedNode =
    layer.spatialSkeletonState.getCachedNode(currentNodeId) ??
    skeletonLayer.getNode(currentNodeId);
  const candidateSegmentId =
    cachedNode?.segmentId ??
    layer.spatialSkeletonState.commandHistory.mappings.resolveSegmentId(
      stableSegmentId,
    );
  if (candidateSegmentId === undefined) {
    throw new Error(
      `Unable to resolve the current segment for node ${stableNodeId}.`,
    );
  }
  let segmentNodes =
    layer.spatialSkeletonState.getCachedSegmentNodes(candidateSegmentId);
  if (segmentNodes === undefined) {
    segmentNodes = await layer.spatialSkeletonState.getFullSegmentNodes(
      skeletonLayer,
      candidateSegmentId,
    );
  }
  const node = findSpatiallyIndexedSkeletonNodeInfo(segmentNodes, currentNodeId);
  if (node === undefined) {
    throw new Error(
      `Node ${currentNodeId} is not available in the inspected skeleton cache.`,
    );
  }
  return {
    skeletonLayer,
    skeletonSource,
    segmentNodes,
    node,
  };
}

function buildInsertEditContext(
  parentNode: SpatiallyIndexedSkeletonNodeInfo,
  childNodes: readonly SpatiallyIndexedSkeletonNodeInfo[],
): SpatiallyIndexedSkeletonEditContext {
  return {
    node: buildSpatiallyIndexedSkeletonNodeEditContext(parentNode).node,
    children: childNodes.map((child) => ({
      nodeId: child.nodeId,
      revisionToken:
        child.revisionToken ??
        (() => {
          throw new Error(
            `Inspected child node ${child.nodeId} is missing revision metadata.`,
          );
        })(),
    })),
  };
}

async function refreshTopologySegments(
  layer: SegmentationUserLayer,
  segmentIds: readonly number[],
) {
  const normalizedSegmentIds = [
    ...new Set(
      segmentIds.filter((value) => Number.isSafeInteger(Math.round(value))),
    ),
  ].map((value) => Math.round(value));
  if (normalizedSegmentIds.length === 0) {
    return;
  }
  const { skeletonLayer } = getEditableSkeletonSourceForLayer(layer);
  layer.spatialSkeletonState.invalidateCachedSegments(normalizedSegmentIds);
  layer.markSpatialSkeletonNodeDataChanged({
    invalidateFullSkeletonCache: false,
  });
  skeletonLayer.invalidateSourceCaches();
  await Promise.allSettled(
    normalizedSegmentIds.map((segmentId) =>
      layer.spatialSkeletonState.getFullSegmentNodes(skeletonLayer, segmentId),
    ),
  );
}

function applyAddNodeToCache(
  layer: SegmentationUserLayer,
  skeletonLayer: SpatiallyIndexedSkeletonLayer,
  committedNode: SpatiallyIndexedSkeletonAddNodeResult,
  parentNodeId: number | undefined,
  position: Float32Array,
  options: {
    focusSelection: boolean;
    moveView: boolean;
    pinSegment: boolean;
  },
) {
  const newNode: CachedNodeTarget = {
    nodeId: committedNode.treenodeId,
    segmentId: committedNode.skeletonId,
    position: new Float32Array(position),
    parentNodeId,
    ...(committedNode.revisionToken === undefined
      ? {}
      : { revisionToken: committedNode.revisionToken }),
  };
  layer.spatialSkeletonState.upsertCachedNode(newNode, {
    allowUncachedSegment: parentNodeId === undefined,
  });
  if (
    parentNodeId !== undefined &&
    committedNode.parentRevisionToken !== undefined
  ) {
    layer.spatialSkeletonState.setCachedNodeRevision(
      parentNodeId,
      committedNode.parentRevisionToken,
    );
  }
  ensureVisibleSegment(layer, newNode.segmentId);
  selectSegment(layer, newNode.segmentId, options.pinSegment);
  if (options.focusSelection) {
    layer.selectSpatialSkeletonNode(
      newNode.nodeId,
      layer.manager.root.selectionState.pin.value,
      {
        segmentId: newNode.segmentId,
        position: newNode.position,
      },
    );
    if (options.moveView) {
      layer.moveViewToSpatialSkeletonNodePosition(newNode.position);
    }
  }
  if (parentNodeId !== undefined) {
    skeletonLayer.retainOverlaySegment(newNode.segmentId);
  }
  layer.markSpatialSkeletonNodeDataChanged({
    invalidateFullSkeletonCache: false,
  });
}

function applyDeleteNodeToCache(
  layer: SegmentationUserLayer,
  deleteContext: DeleteOperationContext,
  options: {
    moveView: boolean;
  },
  childRevisionUpdates: readonly SpatiallyIndexedSkeletonNodeRevisionUpdate[] = [],
) {
  const { node, parentNode, childNodes } = deleteContext;
  const directChildIds = childNodes.map((child) => child.nodeId);
  layer.spatialSkeletonState.removeCachedNode(node.nodeId, {
    parentNodeId: node.parentNodeId,
    childNodeIds: directChildIds,
  });
  if (childRevisionUpdates.length > 0) {
    layer.spatialSkeletonState.setCachedNodeRevisions(childRevisionUpdates);
  }
  if (parentNode !== undefined) {
    if (options.moveView) {
      layer.selectAndMoveToSpatialSkeletonNode(
        parentNode,
        layer.manager.root.selectionState.pin.value,
      );
    } else {
      layer.selectSpatialSkeletonNode(
        parentNode.nodeId,
        layer.manager.root.selectionState.pin.value,
        {
          segmentId: parentNode.segmentId,
          position: parentNode.position,
        },
      );
    }
  } else {
    layer.clearSpatialSkeletonNodeSelection(
      layer.manager.root.selectionState.pin.value,
    );
  }
  const remainingSegmentNodes =
    layer.spatialSkeletonState.getCachedSegmentNodes(node.segmentId) ?? [];
  if (remainingSegmentNodes.length === 0) {
    removeVisibleSegment(layer, node.segmentId, { deselect: true });
  }
  layer.markSpatialSkeletonNodeDataChanged({
    invalidateFullSkeletonCache: false,
  });
}

async function restoreNodeMetadata(
  layer: SegmentationUserLayer,
  skeletonSource: EditableSpatiallyIndexedSkeletonSource,
  createdNode: SpatiallyIndexedSkeletonNodeInfo,
  snapshot: CachedNodeTarget,
) {
  let nextNode = cloneNodeSnapshot(createdNode);
  if (snapshot.radius !== undefined && snapshot.radius !== nextNode.radius) {
    const radiusResult = await skeletonSource.updateRadius(
      createdNode.nodeId,
      snapshot.radius,
      buildSpatiallyIndexedSkeletonNodeEditContext(nextNode),
    );
    nextNode = {
      ...nextNode,
      radius: snapshot.radius,
      revisionToken: radiusResult.revisionToken ?? nextNode.revisionToken,
    };
  }
  if (
    snapshot.confidence !== undefined &&
    snapshot.confidence !== nextNode.confidence
  ) {
    if (nextNode.revisionToken === undefined) {
      throw new Error(
        `Node ${createdNode.nodeId} is missing revision metadata required to restore confidence.`,
      );
    }
    const confidenceResult = await skeletonSource.updateConfidence(
      createdNode.nodeId,
      snapshot.confidence,
      buildSpatiallyIndexedSkeletonNodeEditContext(nextNode),
    );
    nextNode = {
      ...nextNode,
      confidence: snapshot.confidence,
      revisionToken: confidenceResult.revisionToken ?? nextNode.revisionToken,
    };
  }
  const nextDescriptionLabels = getSpatialSkeletonDescriptionLabels(snapshot.labels);
  const nextLabels = mergeSpatialSkeletonNodeLabels(
    snapshot.labels,
    nextDescriptionLabels,
  );
  const restoredLabels = updateSpatialSkeletonTrueEndLabels(
    nextLabels,
    hasSpatialSkeletonTrueEndLabel(snapshot.labels),
  );
  if (!spatialSkeletonLabelListsEqual(nextNode.labels, restoredLabels)) {
    const descriptionResult = await skeletonSource.updateDescription(
      createdNode.nodeId,
      nextDescriptionLabels.join("\n"),
      {
        trueEnd: hasSpatialSkeletonTrueEndLabel(snapshot.labels),
      },
    );
    nextNode = {
      ...nextNode,
      labels: restoredLabels,
      revisionToken: descriptionResult.revisionToken ?? nextNode.revisionToken,
    };
  }
  layer.spatialSkeletonState.upsertCachedNode(nextNode);
  return nextNode;
}

class AddNodeCommand implements SpatialSkeletonCommand {
  readonly label = "Add node";
  private stableNodeId: number | undefined;
  private stableSegmentId: number | undefined;

  constructor(
    private layer: SegmentationUserLayer,
    private stableParentNodeId: number | undefined,
    private targetSkeletonId: number,
    private position: Float32Array,
  ) {}

  private async addNode(
    _context: SpatialSkeletonCommandContext,
    options: {
      moveView: boolean;
      pinSegment: boolean;
      statusPrefix: string;
    },
  ) {
    const { skeletonLayer, skeletonSource } = getEditableSkeletonSourceForLayer(
      this.layer,
    );
    const currentParentNodeId =
      this.stableParentNodeId === undefined
        ? undefined
        : this.layer.spatialSkeletonState.commandHistory.mappings.resolveNodeId(
            this.stableParentNodeId,
          );
    let resolvedEditContext: SpatiallyIndexedSkeletonEditContext | undefined;
    let resolvedSkeletonId = this.targetSkeletonId;
    if (currentParentNodeId !== undefined) {
      const parentNode = (
        await getResolvedNodeForEdit(
          this.layer,
          this.stableParentNodeId!,
          normalizeStableSegmentId(this.layer, this.targetSkeletonId),
        )
      ).node;
      resolvedSkeletonId = parentNode.segmentId;
      resolvedEditContext = buildSpatiallyIndexedSkeletonNodeEditContext(parentNode);
    }
    const result = await skeletonSource.addNode(
      resolvedSkeletonId,
      Number(this.position[0]),
      Number(this.position[1]),
      Number(this.position[2]),
      currentParentNodeId,
      resolvedEditContext,
    );
    if (this.stableNodeId === undefined) {
      this.stableNodeId = result.treenodeId;
    } else {
      this.layer.spatialSkeletonState.commandHistory.mappings.remapNodeId(
        this.stableNodeId,
        result.treenodeId,
      );
    }
    if (this.stableSegmentId === undefined) {
      this.stableSegmentId = result.skeletonId;
    } else {
      this.layer.spatialSkeletonState.commandHistory.mappings.remapSegmentId(
        this.stableSegmentId,
        result.skeletonId,
      );
    }
    applyAddNodeToCache(
      this.layer,
      skeletonLayer,
      result,
      currentParentNodeId,
      this.position,
      {
        focusSelection: true,
        moveView: options.moveView,
        pinSegment: options.pinSegment,
      },
    );
    StatusMessage.showTemporaryMessage(
      `${options.statusPrefix} node ${result.treenodeId} on segment ${result.skeletonId}.`,
    );
  }

  async execute(context: SpatialSkeletonCommandContext) {
    await this.addNode(context, {
      moveView: true,
      pinSegment: true,
      statusPrefix: "Added",
    });
  }

  async undo(_context: SpatialSkeletonCommandContext) {
    if (this.stableNodeId === undefined) {
      throw new Error("Add-node undo is missing the created node id.");
    }
    const resolvedNode = await getResolvedNodeForEdit(
      this.layer,
      this.stableNodeId,
      this.stableSegmentId,
    );
    const deleteContext =
      await this.layer.getSpatialSkeletonDeleteOperationContext(resolvedNode.node);
    const result = await resolvedNode.skeletonSource.deleteNode(
      resolvedNode.node.nodeId,
      {
        childNodeIds: [],
        editContext: deleteContext.editContext,
      },
    );
    applyDeleteNodeToCache(this.layer, deleteContext, { moveView: false }, result.childRevisionUpdates);
    StatusMessage.showTemporaryMessage(
      `Undid add node ${resolvedNode.node.nodeId}.`,
    );
  }

  async redo(context: SpatialSkeletonCommandContext) {
    await this.addNode(context, {
      moveView: false,
      pinSegment: false,
      statusPrefix: "Redid add of",
    });
  }
}

class MoveNodeCommand implements SpatialSkeletonCommand {
  readonly label = "Move node";

  constructor(
    private layer: SegmentationUserLayer,
    private stableNodeId: number,
    private stableSegmentId: number | undefined,
    private beforePosition: Float32Array,
    private afterPosition: Float32Array,
  ) {}

  private async moveTo(
    position: Float32Array,
    statusPrefix: string,
  ) {
    const { node, skeletonLayer, skeletonSource } = await getResolvedNodeForEdit(
      this.layer,
      this.stableNodeId,
      this.stableSegmentId,
    );
    const result = await skeletonSource.moveNode(
      node.nodeId,
      Number(position[0]),
      Number(position[1]),
      Number(position[2]),
      buildSpatiallyIndexedSkeletonNodeEditContext(node),
    );
    skeletonLayer.retainOverlaySegment(node.segmentId);
    this.layer.spatialSkeletonState.moveCachedNode(node.nodeId, position);
    if (result.revisionToken !== undefined) {
      this.layer.spatialSkeletonState.setCachedNodeRevision(
        node.nodeId,
        result.revisionToken,
      );
    }
    this.layer.markSpatialSkeletonNodeDataChanged({
      invalidateFullSkeletonCache: false,
    });
    StatusMessage.showTemporaryMessage(
      `${statusPrefix} node ${node.nodeId} to (${Math.round(position[0])}, ${Math.round(position[1])}, ${Math.round(position[2])}).`,
    );
  }

  execute() {
    return this.moveTo(this.afterPosition, "Moved");
  }

  undo() {
    return this.moveTo(this.beforePosition, "Undid move of");
  }

  redo() {
    return this.moveTo(this.afterPosition, "Redid move of");
  }
}

class DeleteNodeCommand implements SpatialSkeletonCommand {
  readonly label = "Delete node";
  private stableDeletedNodeId: number;
  private stableSegmentId: number | undefined;
  private stableParentNodeId: number | undefined;
  private stableChildNodeIds: number[];
  private deletedSnapshot: CachedNodeTarget;

  constructor(
    private layer: SegmentationUserLayer,
    node: SpatiallyIndexedSkeletonNodeInfo,
    childNodes: readonly SpatiallyIndexedSkeletonNodeInfo[],
  ) {
    this.stableDeletedNodeId = normalizeStableNodeId(layer, node.nodeId)!;
    this.stableSegmentId = normalizeStableSegmentId(layer, node.segmentId);
    this.stableParentNodeId = normalizeStableNodeId(layer, node.parentNodeId);
    this.stableChildNodeIds = childNodes.map(
      (child) => normalizeStableNodeId(layer, child.nodeId)!,
    );
    this.deletedSnapshot = cloneNodeSnapshot(node);
  }

  private async deleteNode(options: {
    moveView: boolean;
    statusPrefix: string;
  }) {
    const resolvedNode = await getResolvedNodeForEdit(
      this.layer,
      this.stableDeletedNodeId,
      this.stableSegmentId,
    );
    const deleteContext =
      await this.layer.getSpatialSkeletonDeleteOperationContext(resolvedNode.node);
    const result = await resolvedNode.skeletonSource.deleteNode(
      resolvedNode.node.nodeId,
      {
        childNodeIds: deleteContext.childNodes.map((child) => child.nodeId),
        editContext: deleteContext.editContext,
      },
    );
    applyDeleteNodeToCache(
      this.layer,
      deleteContext,
      { moveView: options.moveView },
      result.childRevisionUpdates,
    );
    resolvedNode.skeletonLayer.invalidateSourceCaches();
    StatusMessage.showTemporaryMessage(
      `${options.statusPrefix} node ${resolvedNode.node.nodeId}.`,
    );
  }

  private async restoreDeletedNode(statusPrefix: string) {
    const { skeletonSource } = getEditableSkeletonSourceForLayer(this.layer);
    const currentParentNode =
      this.stableParentNodeId === undefined
        ? undefined
        : (
            await getResolvedNodeForEdit(
              this.layer,
              this.stableParentNodeId,
              this.stableSegmentId,
            )
          ).node;
    const currentChildNodes = await Promise.all(
      this.stableChildNodeIds.map((stableChildNodeId) =>
        getResolvedNodeForEdit(
          this.layer,
          stableChildNodeId,
          this.stableSegmentId,
        ).then((result) => result.node),
      ),
    );
    const createResult:
      | SpatiallyIndexedSkeletonAddNodeResult
      | SpatiallyIndexedSkeletonInsertNodeResult =
      currentChildNodes.length === 0
        ? await skeletonSource.addNode(
            currentParentNode?.segmentId ?? 0,
            Number(this.deletedSnapshot.position[0]),
            Number(this.deletedSnapshot.position[1]),
            Number(this.deletedSnapshot.position[2]),
            currentParentNode?.nodeId,
            currentParentNode === undefined
              ? undefined
              : buildSpatiallyIndexedSkeletonNodeEditContext(currentParentNode),
          )
        : await skeletonSource.insertNode(
            currentParentNode?.segmentId ?? this.deletedSnapshot.segmentId,
            Number(this.deletedSnapshot.position[0]),
            Number(this.deletedSnapshot.position[1]),
            Number(this.deletedSnapshot.position[2]),
            currentParentNode?.nodeId ??
              (() => {
                throw new Error(
                  "Delete-node undo is missing the parent node needed for insertion.",
                );
              })(),
            currentChildNodes.map((child) => child.nodeId),
            buildInsertEditContext(currentParentNode!, currentChildNodes),
          );
    this.layer.spatialSkeletonState.commandHistory.mappings.remapNodeId(
      this.stableDeletedNodeId,
      createResult.treenodeId,
    );
    if (this.stableSegmentId === undefined) {
      this.stableSegmentId = createResult.skeletonId;
    } else {
      this.layer.spatialSkeletonState.commandHistory.mappings.remapSegmentId(
        this.stableSegmentId,
        createResult.skeletonId,
      );
    }
    const restoredNode: SpatiallyIndexedSkeletonNodeInfo = {
      nodeId: createResult.treenodeId,
      segmentId: createResult.skeletonId,
      position: new Float32Array(this.deletedSnapshot.position),
      parentNodeId: currentParentNode?.nodeId,
      revisionToken: createResult.revisionToken,
      radius: undefined,
      confidence: undefined,
      labels: undefined,
    };
    this.layer.spatialSkeletonState.upsertCachedNode(restoredNode, {
      allowUncachedSegment: currentParentNode === undefined,
    });
    if (createResult.parentRevisionToken !== undefined && currentParentNode) {
      this.layer.spatialSkeletonState.setCachedNodeRevision(
        currentParentNode.nodeId,
        createResult.parentRevisionToken,
      );
    }
    if (
      "childRevisionUpdates" in createResult &&
      createResult.childRevisionUpdates?.length
    ) {
      this.layer.spatialSkeletonState.setCachedNodeRevisions(
        createResult.childRevisionUpdates,
      );
    }
    const restoredMetadataNode = await restoreNodeMetadata(
      this.layer,
      skeletonSource,
      restoredNode,
      this.deletedSnapshot,
    );
    ensureVisibleSegment(this.layer, restoredMetadataNode.segmentId);
    this.layer.selectSpatialSkeletonNode(
      restoredMetadataNode.nodeId,
      this.layer.manager.root.selectionState.pin.value,
      {
        segmentId: restoredMetadataNode.segmentId,
        position: restoredMetadataNode.position,
      },
    );
    this.layer.markSpatialSkeletonNodeDataChanged({
      invalidateFullSkeletonCache: false,
    });
    StatusMessage.showTemporaryMessage(
      `${statusPrefix} node ${restoredMetadataNode.nodeId}.`,
    );
  }

  execute() {
    return this.deleteNode({
      moveView: true,
      statusPrefix: "Deleted",
    });
  }

  undo() {
    return this.restoreDeletedNode("Restored");
  }

  redo() {
    return this.deleteNode({
      moveView: false,
      statusPrefix: "Redid deletion of",
    });
  }
}

class NodeLabelsCommand implements SpatialSkeletonCommand {
  readonly label = "Edit node labels";

  constructor(
    private layer: SegmentationUserLayer,
    private stableNodeId: number,
    private stableSegmentId: number | undefined,
    private beforeLabels: readonly string[] | undefined,
    private afterLabels: readonly string[] | undefined,
  ) {}

  private async applyLabels(
    labels: readonly string[] | undefined,
    statusPrefix: string,
  ) {
    const { node, skeletonSource } = await getResolvedNodeForEdit(
      this.layer,
      this.stableNodeId,
      this.stableSegmentId,
    );
    const descriptionLabels = getSpatialSkeletonDescriptionLabels(labels);
    const nextLabels = mergeSpatialSkeletonNodeLabels(
      node.labels,
      descriptionLabels,
    );
    const restoredLabels = updateSpatialSkeletonTrueEndLabels(
      nextLabels,
      hasSpatialSkeletonTrueEndLabel(labels),
    );
    const result = await skeletonSource.updateDescription(
      node.nodeId,
      descriptionLabels.join("\n"),
      {
        trueEnd: hasSpatialSkeletonTrueEndLabel(labels),
      },
    );
    this.layer.spatialSkeletonState.updateCachedNode(node.nodeId, (candidate) => {
      if (spatialSkeletonLabelListsEqual(candidate.labels, restoredLabels)) {
        return candidate;
      }
      return {
        ...candidate,
        labels: restoredLabels,
      };
    });
    if (result.revisionToken !== undefined) {
      this.layer.spatialSkeletonState.setCachedNodeRevision(
        node.nodeId,
        result.revisionToken,
      );
    }
    this.layer.markSpatialSkeletonNodeDataChanged({
      invalidateFullSkeletonCache: false,
    });
    StatusMessage.showTemporaryMessage(`${statusPrefix} node ${node.nodeId} labels.`);
  }

  execute() {
    return this.applyLabels(this.afterLabels, "Updated");
  }

  undo() {
    return this.applyLabels(this.beforeLabels, "Undid label update for");
  }

  redo() {
    return this.applyLabels(this.afterLabels, "Redid label update for");
  }
}

class NodePropertiesCommand implements SpatialSkeletonCommand {
  readonly label = "Edit node properties";

  constructor(
    private layer: SegmentationUserLayer,
    private stableNodeId: number,
    private stableSegmentId: number | undefined,
    private before: { radius: number; confidence: number },
    private after: { radius: number; confidence: number },
  ) {}

  private async applyProperties(
    next: { radius: number; confidence: number },
    statusPrefix: string,
  ) {
    const { node, skeletonSource } = await getResolvedNodeForEdit(
      this.layer,
      this.stableNodeId,
      this.stableSegmentId,
    );
    let currentNode = cloneNodeSnapshot(node);
    if (currentNode.radius !== next.radius) {
      const radiusResult = await skeletonSource.updateRadius(
        node.nodeId,
        next.radius,
        buildSpatiallyIndexedSkeletonNodeEditContext(currentNode),
      );
      currentNode = {
        ...currentNode,
        radius: next.radius,
        revisionToken: radiusResult.revisionToken ?? currentNode.revisionToken,
      };
    }
    if (currentNode.confidence !== next.confidence) {
      if (currentNode.revisionToken === undefined) {
        throw new Error(
          `Node ${node.nodeId} is missing revision metadata required to update confidence.`,
        );
      }
      const confidenceResult = await skeletonSource.updateConfidence(
        node.nodeId,
        next.confidence,
        buildSpatiallyIndexedSkeletonNodeEditContext(currentNode),
      );
      currentNode = {
        ...currentNode,
        confidence: next.confidence,
        revisionToken:
          confidenceResult.revisionToken ?? currentNode.revisionToken,
      };
    }
    this.layer.spatialSkeletonState.setNodeProperties(node.nodeId, next);
    if (currentNode.revisionToken !== undefined) {
      this.layer.spatialSkeletonState.setCachedNodeRevision(
        node.nodeId,
        currentNode.revisionToken,
      );
    }
    this.layer.markSpatialSkeletonNodeDataChanged({
      invalidateFullSkeletonCache: false,
    });
    StatusMessage.showTemporaryMessage(
      `${statusPrefix} node ${node.nodeId} properties.`,
    );
  }

  execute() {
    return this.applyProperties(this.after, "Updated");
  }

  undo() {
    return this.applyProperties(this.before, "Undid property update for");
  }

  redo() {
    return this.applyProperties(this.after, "Redid property update for");
  }
}

class RerootCommand implements SpatialSkeletonCommand {
  readonly label = "Reroot skeleton";

  constructor(
    private layer: SegmentationUserLayer,
    private stableNodeId: number,
    private stableSegmentId: number | undefined,
    private stablePreviousRootNodeId: number,
  ) {}

  private async rerootAt(
    stableTargetNodeId: number,
    statusPrefix: string,
  ) {
    const resolvedNode = await getResolvedNodeForEdit(
      this.layer,
      stableTargetNodeId,
      this.stableSegmentId,
    );
    if (resolvedNode.node.parentNodeId === undefined) {
      return;
    }
    if (resolvedNode.skeletonSource.rerootSkeleton === undefined) {
      throw new Error(
        "Unable to resolve a reroot-capable skeleton source for the active layer.",
      );
    }
    await resolvedNode.skeletonSource.rerootSkeleton(
      resolvedNode.node.nodeId,
      buildSpatiallyIndexedSkeletonNeighborhoodEditContext(
        resolvedNode.node,
        resolvedNode.segmentNodes,
      ),
    );
    const rerootedPathNodeIds =
      this.layer.spatialSkeletonState.rerootCachedSegment(
        resolvedNode.node.nodeId,
      ) ?? [];
    if (rerootedPathNodeIds.length > 0) {
      try {
        const revisionLookupSource =
          getSpatiallyIndexedSkeletonNodeRevisionLookupSource(
            resolvedNode.skeletonLayer,
          );
        const revisionUpdates =
          await revisionLookupSource?.getNodeRevisionUpdates(rerootedPathNodeIds);
        if (revisionUpdates !== undefined && revisionUpdates.length > 0) {
          this.layer.spatialSkeletonState.setCachedNodeRevisions(revisionUpdates);
        }
      } catch {
        // Ignore follow-up revision refresh failures. The reroot itself succeeded.
      }
    }
    this.layer.selectSpatialSkeletonNode(
      resolvedNode.node.nodeId,
      this.layer.manager.root.selectionState.pin.value,
      {
        segmentId: resolvedNode.node.segmentId,
        position: resolvedNode.node.position,
      },
    );
    this.layer.markSpatialSkeletonNodeDataChanged({
      invalidateFullSkeletonCache: false,
    });
    StatusMessage.showTemporaryMessage(
      `${statusPrefix} node ${resolvedNode.node.nodeId} as root.`,
    );
  }

  execute() {
    return this.rerootAt(this.stableNodeId, "Set");
  }

  undo() {
    return this.rerootAt(this.stablePreviousRootNodeId, "Undid reroot for");
  }

  redo() {
    return this.rerootAt(this.stableNodeId, "Redid reroot for");
  }
}

class SplitCommand implements SpatialSkeletonCommand {
  readonly label = "Split skeleton";
  private stableNewSegmentId: number | undefined;

  constructor(
    private layer: SegmentationUserLayer,
    private stableNodeId: number,
    private stableSegmentId: number | undefined,
    private stableFormerParentNodeId: number | undefined,
  ) {}

  private async split(statusPrefix: string) {
    const resolvedNode = await getResolvedNodeForEdit(
      this.layer,
      this.stableNodeId,
      this.stableSegmentId,
    );
    let result: SpatiallyIndexedSkeletonSplitResult;
    try {
      result = await resolvedNode.skeletonSource.splitSkeleton(
        resolvedNode.node.nodeId,
        buildSpatiallyIndexedSkeletonNeighborhoodEditContext(
          resolvedNode.node,
          resolvedNode.segmentNodes,
        ),
      );
    } catch (error) {
      await refreshTopologySegments(this.layer, [resolvedNode.node.segmentId]);
      throw error;
    }
    const newSkeletonId = result.newSkeletonId;
    const existingSkeletonId =
      result.existingSkeletonId ?? resolvedNode.node.segmentId;
    if (newSkeletonId === undefined) {
      throw new Error(
        "The active skeleton source did not return a new skeleton id for the split.",
      );
    }
    if (this.stableNewSegmentId === undefined) {
      this.stableNewSegmentId = newSkeletonId;
    } else {
      this.layer.spatialSkeletonState.commandHistory.mappings.remapSegmentId(
        this.stableNewSegmentId,
        newSkeletonId,
      );
    }
    if (this.stableSegmentId !== undefined) {
      this.layer.spatialSkeletonState.commandHistory.mappings.remapSegmentId(
        this.stableSegmentId,
        existingSkeletonId,
      );
    }
    ensureVisibleSegment(this.layer, existingSkeletonId);
    ensureVisibleSegment(this.layer, newSkeletonId);
    selectSegment(this.layer, newSkeletonId, true);
    this.layer.selectSpatialSkeletonNode(
      resolvedNode.node.nodeId,
      this.layer.manager.root.selectionState.pin.value,
      {
        segmentId: newSkeletonId,
      },
    );
    await refreshTopologySegments(this.layer, [existingSkeletonId, newSkeletonId]);
    StatusMessage.showTemporaryMessage(
      `${statusPrefix} skeleton ${existingSkeletonId}. New skeleton: ${newSkeletonId}.`,
    );
  }

  private async mergeBack(statusPrefix: string) {
    if (this.stableFormerParentNodeId === undefined) {
      throw new Error("Split-node undo is missing the former parent node.");
    }
    const splitNode = await getResolvedNodeForEdit(
      this.layer,
      this.stableNodeId,
      this.stableNewSegmentId ?? this.stableSegmentId,
    );
    const formerParent = await getResolvedNodeForEdit(
      this.layer,
      this.stableFormerParentNodeId,
      this.stableSegmentId,
    );
    let result: SpatiallyIndexedSkeletonMergeResult;
    try {
      result = await splitNode.skeletonSource.mergeSkeletons(
        splitNode.node.nodeId,
        formerParent.node.nodeId,
        buildSpatiallyIndexedSkeletonMultiNodeEditContext(
          splitNode.node,
          formerParent.node,
        ),
      );
    } catch (error) {
      await refreshTopologySegments(this.layer, [
        splitNode.node.segmentId,
        formerParent.node.segmentId,
      ]);
      throw error;
    }
    const resultSkeletonId =
      result.resultSkeletonId ?? formerParent.node.segmentId;
    const deletedSkeletonId =
      result.deletedSkeletonId ??
      (resultSkeletonId === splitNode.node.segmentId
        ? formerParent.node.segmentId
        : splitNode.node.segmentId);
    if (this.stableSegmentId !== undefined) {
      this.layer.spatialSkeletonState.commandHistory.mappings.remapSegmentId(
        this.stableSegmentId,
        resultSkeletonId,
      );
    }
    if (this.stableNewSegmentId !== undefined) {
      this.layer.spatialSkeletonState.commandHistory.mappings.remapSegmentId(
        this.stableNewSegmentId,
        resultSkeletonId,
      );
    }
    ensureVisibleSegment(this.layer, resultSkeletonId);
    if (deletedSkeletonId !== resultSkeletonId) {
      removeVisibleSegment(this.layer, deletedSkeletonId, { deselect: true });
      this.layer.displayState.segmentStatedColors.value.delete(
        BigInt(deletedSkeletonId),
      );
      splitNode.skeletonLayer.suppressBrowseSegment(deletedSkeletonId);
    }
    this.layer.selectSpatialSkeletonNode(
      splitNode.node.nodeId,
      this.layer.manager.root.selectionState.pin.value,
      {
        segmentId: resultSkeletonId,
      },
    );
    await refreshTopologySegments(this.layer, [
      resultSkeletonId,
      deletedSkeletonId,
    ]);
    StatusMessage.showTemporaryMessage(
      `${statusPrefix} split at node ${splitNode.node.nodeId}.`,
    );
  }

  execute() {
    return this.split("Split");
  }

  undo() {
    return this.mergeBack("Undid");
  }

  redo() {
    return this.split("Redid split of");
  }
}

class MergeCommand implements SpatialSkeletonCommand {
  readonly label = "Merge skeletons";
  private stableResultSegmentId: number | undefined;
  private stableDeletedSegmentId: number | undefined;
  private stableAttachedNodeId: number | undefined;
  private stableAttachedRootNodeId: number | undefined;

  constructor(
    private layer: SegmentationUserLayer,
    private stableFirstNodeId: number,
    private stableFirstSegmentId: number | undefined,
    private stableSecondNodeId: number,
    private stableSecondSegmentId: number | undefined,
  ) {}

  private async merge(statusPrefix: string) {
    const firstNode = await getResolvedNodeForEdit(
      this.layer,
      this.stableFirstNodeId,
      this.stableFirstSegmentId,
    );
    const secondNode = await getResolvedNodeForEdit(
      this.layer,
      this.stableSecondNodeId,
      this.stableSecondSegmentId,
    );
    let result: SpatiallyIndexedSkeletonMergeResult;
    try {
      result = await firstNode.skeletonSource.mergeSkeletons(
        firstNode.node.nodeId,
        secondNode.node.nodeId,
        buildSpatiallyIndexedSkeletonMultiNodeEditContext(
          firstNode.node,
          secondNode.node,
        ),
      );
    } catch (error) {
      await refreshTopologySegments(this.layer, [
        firstNode.node.segmentId,
        secondNode.node.segmentId,
      ]);
      throw error;
    }
    const winningNode =
      result.resultSkeletonId === secondNode.node.segmentId
        ? secondNode.node
        : firstNode.node;
    const losingNode =
      winningNode.nodeId === firstNode.node.nodeId ? secondNode.node : firstNode.node;
    const losingSegmentNodes =
      losingNode.segmentId === firstNode.node.segmentId
        ? firstNode.segmentNodes
        : secondNode.segmentNodes;
    const attachedRoot = findRootNode(losingSegmentNodes);
    const resultSkeletonId = result.resultSkeletonId ?? winningNode.segmentId;
    const deletedSkeletonId = result.deletedSkeletonId ?? losingNode.segmentId;
    this.stableAttachedNodeId =
      this.stableAttachedNodeId ?? normalizeStableNodeId(this.layer, losingNode.nodeId);
    this.stableAttachedRootNodeId =
      this.stableAttachedRootNodeId ??
      normalizeStableNodeId(this.layer, attachedRoot?.nodeId);
    this.stableResultSegmentId =
      this.stableResultSegmentId ??
      normalizeStableSegmentId(this.layer, resultSkeletonId);
    this.stableDeletedSegmentId =
      this.stableDeletedSegmentId ??
      normalizeStableSegmentId(this.layer, deletedSkeletonId);
    this.layer.spatialSkeletonState.commandHistory.mappings.remapSegmentId(
      this.stableDeletedSegmentId,
      resultSkeletonId,
    );
    ensureVisibleSegment(this.layer, resultSkeletonId);
    removeVisibleSegment(this.layer, deletedSkeletonId, { deselect: true });
    selectSegment(this.layer, resultSkeletonId, false);
    this.layer.selectSpatialSkeletonNode(
      losingNode.nodeId,
      this.layer.manager.root.selectionState.pin.value,
      {
        segmentId: resultSkeletonId,
      },
    );
    this.layer.displayState.segmentStatedColors.value.delete(BigInt(deletedSkeletonId));
    if (deletedSkeletonId !== resultSkeletonId) {
      firstNode.skeletonLayer.suppressBrowseSegment(deletedSkeletonId);
    }
    this.layer.clearSpatialSkeletonMergeAnchor();
    await refreshTopologySegments(this.layer, [resultSkeletonId, deletedSkeletonId]);
    const swapSuffix = result.stableAnnotationSwap
      ? " Merge direction was adjusted by the active source."
      : "";
    StatusMessage.showTemporaryMessage(
      `${statusPrefix} skeleton ${deletedSkeletonId} into ${resultSkeletonId}.${swapSuffix}`,
    );
  }

  private async undoMerge(statusPrefix: string) {
    if (this.stableAttachedNodeId === undefined) {
      throw new Error("Merge undo is missing the attached node id.");
    }
    if (this.stableDeletedSegmentId === undefined) {
      throw new Error("Merge undo is missing the deleted skeleton id.");
    }
    const attachedNode = await getResolvedNodeForEdit(
      this.layer,
      this.stableAttachedNodeId,
      this.stableResultSegmentId ?? this.stableFirstSegmentId,
    );
    let splitResult: SpatiallyIndexedSkeletonSplitResult;
    try {
      splitResult = await attachedNode.skeletonSource.splitSkeleton(
        attachedNode.node.nodeId,
        buildSpatiallyIndexedSkeletonNeighborhoodEditContext(
          attachedNode.node,
          attachedNode.segmentNodes,
        ),
      );
    } catch (error) {
      await refreshTopologySegments(this.layer, [attachedNode.node.segmentId]);
      throw error;
    }
    const restoredSegmentId =
      splitResult.newSkeletonId ??
      (() => {
        throw new Error(
          "The active skeleton source did not return a new skeleton id for merge undo.",
        );
      })();
    this.layer.spatialSkeletonState.commandHistory.mappings.remapSegmentId(
      this.stableDeletedSegmentId,
      restoredSegmentId,
    );
    const survivingSegmentId =
      splitResult.existingSkeletonId ?? attachedNode.node.segmentId;
    await refreshTopologySegments(this.layer, [survivingSegmentId, restoredSegmentId]);
    if (
      this.stableAttachedRootNodeId !== undefined &&
      this.stableAttachedRootNodeId !== this.stableAttachedNodeId
    ) {
      const restoredRoot = await getResolvedNodeForEdit(
        this.layer,
        this.stableAttachedRootNodeId,
        this.stableDeletedSegmentId,
      );
      if (restoredRoot.node.parentNodeId !== undefined) {
        await restoredRoot.skeletonSource.rerootSkeleton?.(
          restoredRoot.node.nodeId,
          buildSpatiallyIndexedSkeletonNeighborhoodEditContext(
            restoredRoot.node,
            restoredRoot.segmentNodes,
          ),
        );
        await refreshTopologySegments(this.layer, [
          survivingSegmentId,
          restoredSegmentId,
        ]);
      }
    }
    ensureVisibleSegment(this.layer, survivingSegmentId);
    ensureVisibleSegment(this.layer, restoredSegmentId);
    this.layer.selectSpatialSkeletonNode(
      attachedNode.node.nodeId,
      this.layer.manager.root.selectionState.pin.value,
      {
        segmentId: restoredSegmentId,
      },
    );
    StatusMessage.showTemporaryMessage(
      `${statusPrefix} merge involving node ${attachedNode.node.nodeId}.`,
    );
  }

  execute() {
    return this.merge("Merged");
  }

  undo() {
    return this.undoMerge("Undid");
  }

  redo() {
    return this.merge("Redid merge of");
  }
}

export function executeSpatialSkeletonAddNode(
  layer: SegmentationUserLayer,
  options: {
    skeletonId: number;
    parentNodeId: number | undefined;
    position: Float32Array;
  },
) {
  const command = new AddNodeCommand(
    layer,
    normalizeStableNodeId(layer, options.parentNodeId),
    normalizeStableSegmentId(layer, options.skeletonId) ?? options.skeletonId,
    new Float32Array(options.position),
  );
  return layer.spatialSkeletonState.commandHistory.execute(command);
}

export function executeSpatialSkeletonMoveNode(
  layer: SegmentationUserLayer,
  options: {
    node: SpatiallyIndexedSkeletonNodeInfo;
    nextPosition: Float32Array;
  },
) {
  const command = new MoveNodeCommand(
    layer,
    normalizeStableNodeId(layer, options.node.nodeId)!,
    normalizeStableSegmentId(layer, options.node.segmentId),
    new Float32Array(options.node.position),
    new Float32Array(options.nextPosition),
  );
  return layer.spatialSkeletonState.commandHistory.execute(command);
}

export function executeSpatialSkeletonDeleteNode(
  layer: SegmentationUserLayer,
  node: SpatiallyIndexedSkeletonNodeInfo,
) {
  const segmentNodes = layer.getCachedSpatialSkeletonSegmentNodesForEdit(
    node.segmentId,
  );
  const refreshedNode = findSpatiallyIndexedSkeletonNodeInfo(segmentNodes, node.nodeId);
  if (refreshedNode === undefined) {
    throw new Error(
      `Node ${node.nodeId} is not available in the inspected skeleton cache.`,
    );
  }
  const childNodes = getSpatiallyIndexedSkeletonDirectChildren(
    segmentNodes,
    refreshedNode.nodeId,
  );
  const command = new DeleteNodeCommand(layer, refreshedNode, childNodes);
  return layer.spatialSkeletonState.commandHistory.execute(command);
}

export function executeSpatialSkeletonNodeLabelUpdate(
  layer: SegmentationUserLayer,
  options: {
    node: SpatiallyIndexedSkeletonNodeInfo;
    nextLabels: readonly string[] | undefined;
  },
) {
  const command = new NodeLabelsCommand(
    layer,
    normalizeStableNodeId(layer, options.node.nodeId)!,
    normalizeStableSegmentId(layer, options.node.segmentId),
    options.node.labels === undefined ? undefined : [...options.node.labels],
    options.nextLabels === undefined ? undefined : [...options.nextLabels],
  );
  return layer.spatialSkeletonState.commandHistory.execute(command);
}

export function executeSpatialSkeletonNodePropertiesUpdate(
  layer: SegmentationUserLayer,
  options: {
    node: SpatiallyIndexedSkeletonNodeInfo;
    next: { radius: number; confidence: number };
  },
) {
  const command = new NodePropertiesCommand(
    layer,
    normalizeStableNodeId(layer, options.node.nodeId)!,
    normalizeStableSegmentId(layer, options.node.segmentId),
    {
      radius: options.node.radius ?? 0,
      confidence: options.node.confidence ?? 0,
    },
    options.next,
  );
  return layer.spatialSkeletonState.commandHistory.execute(command);
}

export function executeSpatialSkeletonReroot(
  layer: SegmentationUserLayer,
  node: Pick<
    SpatiallyIndexedSkeletonNodeInfo,
    "nodeId" | "segmentId" | "parentNodeId"
  >,
) {
  const segmentNodes = layer.getCachedSpatialSkeletonSegmentNodesForEdit(
    node.segmentId,
  );
  const rootNode =
    findRootNode(segmentNodes) ??
    (() => {
      throw new Error(
        `Unable to resolve the current root for segment ${node.segmentId}.`,
      );
    })();
  const command = new RerootCommand(
    layer,
    normalizeStableNodeId(layer, node.nodeId)!,
    normalizeStableSegmentId(layer, node.segmentId),
    normalizeStableNodeId(layer, rootNode.nodeId)!,
  );
  return layer.spatialSkeletonState.commandHistory.execute(command);
}

export function executeSpatialSkeletonSplit(
  layer: SegmentationUserLayer,
  node: Pick<SpatiallyIndexedSkeletonNodeInfo, "nodeId" | "segmentId">,
) {
  const segmentNodes = layer.getCachedSpatialSkeletonSegmentNodesForEdit(
    node.segmentId,
  );
  const splitNode = findSpatiallyIndexedSkeletonNodeInfo(segmentNodes, node.nodeId);
  if (splitNode === undefined) {
    throw new Error(
      `Node ${node.nodeId} is not available in the inspected skeleton cache.`,
    );
  }
  const command = new SplitCommand(
    layer,
    normalizeStableNodeId(layer, splitNode.nodeId)!,
    normalizeStableSegmentId(layer, splitNode.segmentId),
    normalizeStableNodeId(layer, splitNode.parentNodeId),
  );
  return layer.spatialSkeletonState.commandHistory.execute(command);
}

export function executeSpatialSkeletonMerge(
  layer: SegmentationUserLayer,
  firstNode: Pick<SpatiallyIndexedSkeletonNodeInfo, "nodeId" | "segmentId">,
  secondNode: Pick<SpatiallyIndexedSkeletonNodeInfo, "nodeId" | "segmentId">,
) {
  const command = new MergeCommand(
    layer,
    normalizeStableNodeId(layer, firstNode.nodeId)!,
    normalizeStableSegmentId(layer, firstNode.segmentId),
    normalizeStableNodeId(layer, secondNode.nodeId)!,
    normalizeStableSegmentId(layer, secondNode.segmentId),
  );
  return layer.spatialSkeletonState.commandHistory.execute(command);
}

export async function undoSpatialSkeletonCommand(layer: SegmentationUserLayer) {
  const changed = await layer.spatialSkeletonState.commandHistory.undo();
  if (!changed) {
    return false;
  }
  return true;
}

export async function redoSpatialSkeletonCommand(layer: SegmentationUserLayer) {
  const changed = await layer.spatialSkeletonState.commandHistory.redo();
  if (!changed) {
    return false;
  }
  return true;
}
