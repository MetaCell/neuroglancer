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
  SpatiallyIndexedSkeletonNode,
  SpatiallyIndexedSkeletonNodeRevisionUpdate,
  SpatiallyIndexedSkeletonSplitResult,
} from "#src/skeleton/api.js";
import type {
  SpatialSkeletonCommand,
  SpatialSkeletonCommandContext,
} from "#src/skeleton/command_history.js";
import {
  buildSpatiallyIndexedSkeletonMultiNodeEditContext,
  buildSpatiallyIndexedSkeletonNeighborhoodEditContext,
  buildSpatiallyIndexedSkeletonNodeEditContext,
  buildSpatiallyIndexedSkeletonRerootEditContext,
  findSpatiallyIndexedSkeletonNode,
  getSpatiallyIndexedSkeletonDirectChildren,
} from "#src/skeleton/edit_state.js";
import type { SpatiallyIndexedSkeletonLayer } from "#src/skeleton/frontend.js";
import { getEditableSpatiallyIndexedSkeletonSource } from "#src/skeleton/spatial_skeleton_manager.js";
import { StatusMessage } from "#src/status.js";

function cloneNodeSnapshot(
  node: SpatiallyIndexedSkeletonNode,
): SpatiallyIndexedSkeletonNode {
  return {
    nodeId: node.nodeId,
    segmentId: node.segmentId,
    position: new Float32Array(node.position),
    parentNodeId: node.parentNodeId,
    radius: node.radius,
    confidence: node.confidence,
    description: node.description,
    isTrueEnd: node.isTrueEnd,
    revisionToken: node.revisionToken,
  };
}

function getEditableSkeletonSourceForLayer(layer: SegmentationUserLayer): {
  skeletonLayer: SpatiallyIndexedSkeletonLayer;
  skeletonSource: EditableSpatiallyIndexedSkeletonSource;
} {
  const skeletonLayer = layer.getSpatiallyIndexedSkeletonLayer();
  if (skeletonLayer === undefined) {
    throw new Error(
      "No spatially indexed skeleton source is currently loaded.",
    );
  }
  const skeletonSource =
    getEditableSpatiallyIndexedSkeletonSource(skeletonLayer);
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
  segmentNodes: readonly SpatiallyIndexedSkeletonNode[],
) {
  return segmentNodes.find((candidate) => candidate.parentNodeId === undefined);
}

async function getResolvedNodeForEdit(
  layer: SegmentationUserLayer,
  stableNodeId: number,
  stableSegmentId: number | undefined,
): Promise<ResolvedSpatialSkeletonEditNode> {
  const commandMappings = layer.spatialSkeletonState.commandHistory.mappings;
  const currentNodeId = commandMappings.resolveNodeId(stableNodeId);
  if (currentNodeId === undefined) {
    throw new Error(`Unable to resolve current node ${stableNodeId}.`);
  }
  const { skeletonLayer, skeletonSource } =
    getEditableSkeletonSourceForLayer(layer);
  const cachedNode =
    layer.spatialSkeletonState.getCachedNode(currentNodeId) ??
    skeletonLayer.getNode(currentNodeId);
  const candidateSegmentId =
    cachedNode?.segmentId ?? commandMappings.resolveSegmentId(stableSegmentId);
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
  const node = findSpatiallyIndexedSkeletonNode(
    segmentNodes,
    currentNodeId,
  );
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

interface ResolvedSpatialSkeletonEditNode {
  skeletonLayer: SpatiallyIndexedSkeletonLayer;
  skeletonSource: EditableSpatiallyIndexedSkeletonSource;
  segmentNodes: readonly SpatiallyIndexedSkeletonNode[];
  node: SpatiallyIndexedSkeletonNode;
}

function buildInsertEditContext(
  parentNode: SpatiallyIndexedSkeletonNode,
  childNodes: readonly SpatiallyIndexedSkeletonNode[],
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
  positionInModelSpace: Float32Array,
  options: {
    focusSelection: boolean;
    moveView: boolean;
    pinSegment: boolean;
  },
) {
  const newNode: SpatiallyIndexedSkeletonNode = {
    nodeId: committedNode.treenodeId,
    segmentId: committedNode.skeletonId,
    position: new Float32Array(positionInModelSpace),
    parentNodeId,
    isTrueEnd: false,
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
  deleteContext: {
    node: SpatiallyIndexedSkeletonNode;
    parentNode: SpatiallyIndexedSkeletonNode | undefined;
    childNodes: readonly SpatiallyIndexedSkeletonNode[];
    editContext: SpatiallyIndexedSkeletonEditContext;
  },
  options: {
    moveView: boolean;
  },
  nodeRevisionUpdates: readonly SpatiallyIndexedSkeletonNodeRevisionUpdate[] = [],
) {
  const { node, parentNode, childNodes } = deleteContext;
  const directChildIds = childNodes.map((child) => child.nodeId);
  layer.spatialSkeletonState.removeCachedNode(node.nodeId, {
    parentNodeId: node.parentNodeId,
    childNodeIds: directChildIds,
  });
  if (nodeRevisionUpdates.length > 0) {
    layer.spatialSkeletonState.setCachedNodeRevisions(nodeRevisionUpdates);
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

async function applyNodeDescriptionAndTrueEnd(
  skeletonSource: EditableSpatiallyIndexedSkeletonSource,
  node: SpatiallyIndexedSkeletonNode,
  next: {
    description?: string;
    isTrueEnd: boolean;
  },
) {
  const nextDescription = next.description;
  const nextTrueEnd = next.isTrueEnd;
  let updatedNode: SpatiallyIndexedSkeletonNode = {
    ...node,
    description: nextDescription,
    isTrueEnd: nextTrueEnd,
  };
  const descriptionChanged = node.description !== nextDescription;
  if (descriptionChanged) {
    const descriptionResult = await skeletonSource.updateDescription(
      node.nodeId,
      nextDescription ?? "",
    );
    updatedNode = {
      ...updatedNode,
      description: descriptionResult.description,
      revisionToken:
        descriptionResult.revisionToken ?? updatedNode.revisionToken,
    };
  }
  if (node.isTrueEnd !== nextTrueEnd || (descriptionChanged && nextTrueEnd)) {
    const trueEndResult = nextTrueEnd
      ? await skeletonSource.setTrueEnd(node.nodeId)
      : await skeletonSource.removeTrueEnd(node.nodeId);
    updatedNode = {
      ...updatedNode,
      revisionToken: trueEndResult.revisionToken ?? updatedNode.revisionToken,
    };
  }
  return updatedNode;
}

async function restoreNodeAttributes(
  layer: SegmentationUserLayer,
  skeletonSource: EditableSpatiallyIndexedSkeletonSource,
  createdNode: SpatiallyIndexedSkeletonNode,
  snapshot: SpatiallyIndexedSkeletonNode,
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
  if (
    nextNode.description !== snapshot.description ||
    nextNode.isTrueEnd !== snapshot.isTrueEnd
  ) {
    nextNode = await applyNodeDescriptionAndTrueEnd(
      skeletonSource,
      nextNode,
      snapshot,
    );
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
    private positionInModelSpace: Float32Array,
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
          this.layer.spatialSkeletonState.commandHistory.mappings.getStableOrCurrentSegmentId(
            this.targetSkeletonId,
          ),
        )
      ).node;
      resolvedSkeletonId = parentNode.segmentId;
      resolvedEditContext =
        buildSpatiallyIndexedSkeletonNodeEditContext(parentNode);
    }
    const result = await skeletonSource.addNode(
      resolvedSkeletonId,
      Number(this.positionInModelSpace[0]),
      Number(this.positionInModelSpace[1]),
      Number(this.positionInModelSpace[2]),
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
      this.positionInModelSpace,
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
      await this.layer.getSpatialSkeletonDeleteOperationContext(
        resolvedNode.node,
      );
    const result = await resolvedNode.skeletonSource.deleteNode(
      resolvedNode.node.nodeId,
      {
        childNodeIds: [],
        editContext: deleteContext.editContext,
      },
    );
    applyDeleteNodeToCache(
      this.layer,
      deleteContext,
      { moveView: false },
      result.nodeRevisionUpdates,
    );
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
    private beforePositionInModelSpace: Float32Array,
    private afterPositionInModelSpace: Float32Array,
  ) {}

  private async moveTo(
    positionInModelSpace: Float32Array,
    statusPrefix: string,
  ) {
    const { node, skeletonLayer, skeletonSource } =
      await getResolvedNodeForEdit(
        this.layer,
        this.stableNodeId,
        this.stableSegmentId,
      );
    const result = await skeletonSource.moveNode(
      node.nodeId,
      Number(positionInModelSpace[0]),
      Number(positionInModelSpace[1]),
      Number(positionInModelSpace[2]),
      buildSpatiallyIndexedSkeletonNodeEditContext(node),
    );
    skeletonLayer.retainOverlaySegment(node.segmentId);
    this.layer.spatialSkeletonState.moveCachedNode(
      node.nodeId,
      positionInModelSpace,
    );
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
      `${statusPrefix} node ${node.nodeId} to (${Math.round(positionInModelSpace[0])}, ${Math.round(positionInModelSpace[1])}, ${Math.round(positionInModelSpace[2])}).`,
    );
  }

  execute() {
    return this.moveTo(this.afterPositionInModelSpace, "Moved");
  }

  undo() {
    return this.moveTo(this.beforePositionInModelSpace, "Undid move of");
  }

  redo() {
    return this.moveTo(this.afterPositionInModelSpace, "Redid move of");
  }
}

class DeleteNodeCommand implements SpatialSkeletonCommand {
  readonly label = "Delete node";
  private stableDeletedNodeId: number;
  private stableSegmentId: number | undefined;
  private stableParentNodeId: number | undefined;
  private stableChildNodeIds: number[];
  private deletedSnapshot: SpatiallyIndexedSkeletonNode;

  constructor(
    private layer: SegmentationUserLayer,
    node: SpatiallyIndexedSkeletonNode,
    childNodes: readonly SpatiallyIndexedSkeletonNode[],
  ) {
    const commandMappings = layer.spatialSkeletonState.commandHistory.mappings;
    this.stableDeletedNodeId = commandMappings.getStableOrCurrentNodeId(
      node.nodeId,
    )!;
    this.stableSegmentId = commandMappings.getStableOrCurrentSegmentId(
      node.segmentId,
    );
    this.stableParentNodeId = commandMappings.getStableOrCurrentNodeId(
      node.parentNodeId,
    );
    this.stableChildNodeIds = childNodes.map(
      (child) => commandMappings.getStableOrCurrentNodeId(child.nodeId)!,
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
      await this.layer.getSpatialSkeletonDeleteOperationContext(
        resolvedNode.node,
      );
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
      result.nodeRevisionUpdates,
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
    const restoredNode: SpatiallyIndexedSkeletonNode = {
      nodeId: createResult.treenodeId,
      segmentId: createResult.skeletonId,
      position: new Float32Array(this.deletedSnapshot.position),
      parentNodeId: currentParentNode?.nodeId,
      revisionToken: createResult.revisionToken,
      radius: undefined,
      confidence: undefined,
      description: undefined,
      isTrueEnd: false,
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
    if (createResult.nodeRevisionUpdates?.length) {
      this.layer.spatialSkeletonState.setCachedNodeRevisions(
        createResult.nodeRevisionUpdates,
      );
    }
    const restoredNodeWithAttributes = await restoreNodeAttributes(
      this.layer,
      skeletonSource,
      restoredNode,
      this.deletedSnapshot,
    );
    ensureVisibleSegment(this.layer, restoredNodeWithAttributes.segmentId);
    this.layer.selectSpatialSkeletonNode(
      restoredNodeWithAttributes.nodeId,
      this.layer.manager.root.selectionState.pin.value,
      {
        segmentId: restoredNodeWithAttributes.segmentId,
        position: restoredNodeWithAttributes.position,
      },
    );
    this.layer.markSpatialSkeletonNodeDataChanged({
      invalidateFullSkeletonCache: false,
    });
    StatusMessage.showTemporaryMessage(
      `${statusPrefix} node ${restoredNodeWithAttributes.nodeId}.`,
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

class NodeDescriptionCommand implements SpatialSkeletonCommand {
  readonly label = "Edit node description";

  constructor(
    private layer: SegmentationUserLayer,
    private stableNodeId: number,
    private stableSegmentId: number | undefined,
    private beforeDescription: string | undefined,
    private afterDescription: string | undefined,
  ) {}

  private async applyDescription(
    nextDescription: string | undefined,
    statusPrefix: string,
  ) {
    const { node, skeletonSource } = await getResolvedNodeForEdit(
      this.layer,
      this.stableNodeId,
      this.stableSegmentId,
    );
    if (node.description === nextDescription) {
      return;
    }
    const result = await skeletonSource.updateDescription(
      node.nodeId,
      nextDescription ?? "",
    );
    this.layer.spatialSkeletonState.updateCachedNode(
      node.nodeId,
      (candidate) => {
        if (candidate.description === result.description) {
          return candidate;
        }
        return {
          ...candidate,
          description: result.description,
        };
      },
    );
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
      `${statusPrefix} node ${node.nodeId} description.`,
    );
  }

  execute() {
    return this.applyDescription(this.afterDescription, "Updated");
  }

  undo() {
    return this.applyDescription(
      this.beforeDescription,
      "Undid description update for",
    );
  }

  redo() {
    return this.applyDescription(
      this.afterDescription,
      "Redid description update for",
    );
  }
}

class NodeTrueEndCommand implements SpatialSkeletonCommand {
  readonly label = "Edit node true end state";

  constructor(
    private layer: SegmentationUserLayer,
    private stableNodeId: number,
    private stableSegmentId: number | undefined,
    private beforeIsTrueEnd: boolean,
    private afterIsTrueEnd: boolean,
  ) {}

  private async applyTrueEnd(
    nextIsTrueEnd: boolean,
    statusPrefix: string,
  ) {
    const { node, skeletonSource } = await getResolvedNodeForEdit(
      this.layer,
      this.stableNodeId,
      this.stableSegmentId,
    );
    if (node.isTrueEnd === nextIsTrueEnd) {
      return;
    }
    const result = nextIsTrueEnd
      ? await skeletonSource.setTrueEnd(node.nodeId)
      : await skeletonSource.removeTrueEnd(node.nodeId);
    this.layer.spatialSkeletonState.updateCachedNode(
      node.nodeId,
      (candidate) => {
        if (candidate.isTrueEnd === nextIsTrueEnd) {
          return candidate;
        }
        return {
          ...candidate,
          isTrueEnd: nextIsTrueEnd,
        };
      },
    );
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
      `${statusPrefix} node ${node.nodeId} true end state.`,
    );
  }

  execute() {
    return this.applyTrueEnd(this.afterIsTrueEnd, "Updated");
  }

  undo() {
    return this.applyTrueEnd(
      this.beforeIsTrueEnd,
      "Undid true end update for",
    );
  }

  redo() {
    return this.applyTrueEnd(
      this.afterIsTrueEnd,
      "Redid true end update for",
    );
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

  private async rerootAt(stableTargetNodeId: number, statusPrefix: string) {
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
    const result = await resolvedNode.skeletonSource.rerootSkeleton(
      resolvedNode.node.nodeId,
      buildSpatiallyIndexedSkeletonRerootEditContext(
        resolvedNode.node,
        resolvedNode.segmentNodes,
      ),
    );
    this.layer.spatialSkeletonState.rerootCachedSegment(
      resolvedNode.node.nodeId,
    );
    if (
      result.nodeRevisionUpdates !== undefined &&
      result.nodeRevisionUpdates.length > 0
    ) {
      this.layer.spatialSkeletonState.setCachedNodeRevisions(
        result.nodeRevisionUpdates,
      );
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
    await refreshTopologySegments(this.layer, [
      existingSkeletonId,
      newSkeletonId,
    ]);
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
      result = await formerParent.skeletonSource.mergeSkeletons(
        formerParent.node.nodeId,
        splitNode.node.nodeId,
        buildSpatiallyIndexedSkeletonMultiNodeEditContext(
          formerParent.node,
          splitNode.node,
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
      winningNode.nodeId === firstNode.node.nodeId
        ? secondNode.node
        : firstNode.node;
    const losingSegmentNodes =
      losingNode.segmentId === firstNode.node.segmentId
        ? firstNode.segmentNodes
        : secondNode.segmentNodes;
    const attachedRoot = findRootNode(losingSegmentNodes);
    const resultSkeletonId = result.resultSkeletonId ?? winningNode.segmentId;
    const deletedSkeletonId = result.deletedSkeletonId ?? losingNode.segmentId;
    this.stableAttachedNodeId =
      this.stableAttachedNodeId ??
      this.layer.spatialSkeletonState.commandHistory.mappings.getStableOrCurrentNodeId(
        losingNode.nodeId,
      );
    this.stableAttachedRootNodeId =
      this.stableAttachedRootNodeId ??
      this.layer.spatialSkeletonState.commandHistory.mappings.getStableOrCurrentNodeId(
        attachedRoot?.nodeId,
      );
    this.stableResultSegmentId =
      this.stableResultSegmentId ??
      this.layer.spatialSkeletonState.commandHistory.mappings.getStableOrCurrentSegmentId(
        resultSkeletonId,
      );
    this.stableDeletedSegmentId =
      this.stableDeletedSegmentId ??
      this.layer.spatialSkeletonState.commandHistory.mappings.getStableOrCurrentSegmentId(
        deletedSkeletonId,
      );
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
    this.layer.displayState.segmentStatedColors.value.delete(
      BigInt(deletedSkeletonId),
    );
    if (deletedSkeletonId !== resultSkeletonId) {
      firstNode.skeletonLayer.suppressBrowseSegment(deletedSkeletonId);
    }
    this.layer.clearSpatialSkeletonMergeAnchor();
    await refreshTopologySegments(this.layer, [
      resultSkeletonId,
      deletedSkeletonId,
    ]);
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
    await refreshTopologySegments(this.layer, [
      survivingSegmentId,
      restoredSegmentId,
    ]);
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
          buildSpatiallyIndexedSkeletonRerootEditContext(
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
    // Callers must convert viewer/global coordinates to skeleton model space.
    positionInModelSpace: Float32Array;
  },
) {
  const commandMappings = layer.spatialSkeletonState.commandHistory.mappings;
  const command = new AddNodeCommand(
    layer,
    commandMappings.getStableOrCurrentNodeId(options.parentNodeId),
    commandMappings.getStableOrCurrentSegmentId(options.skeletonId) ??
      options.skeletonId,
    new Float32Array(options.positionInModelSpace),
  );
  return layer.spatialSkeletonState.commandHistory.execute(command);
}

export function executeSpatialSkeletonMoveNode(
  layer: SegmentationUserLayer,
  options: {
    node: SpatiallyIndexedSkeletonNode;
    // Callers must convert viewer/global coordinates to skeleton model space.
    nextPositionInModelSpace: Float32Array;
  },
) {
  const commandMappings = layer.spatialSkeletonState.commandHistory.mappings;
  const command = new MoveNodeCommand(
    layer,
    commandMappings.getStableOrCurrentNodeId(options.node.nodeId)!,
    commandMappings.getStableOrCurrentSegmentId(options.node.segmentId),
    new Float32Array(options.node.position),
    new Float32Array(options.nextPositionInModelSpace),
  );
  return layer.spatialSkeletonState.commandHistory.execute(command);
}

export function executeSpatialSkeletonDeleteNode(
  layer: SegmentationUserLayer,
  node: SpatiallyIndexedSkeletonNode,
) {
  const segmentNodes = layer.getCachedSpatialSkeletonSegmentNodesForEdit(
    node.segmentId,
  );
  const refreshedNode = findSpatiallyIndexedSkeletonNode(
    segmentNodes,
    node.nodeId,
  );
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

export function executeSpatialSkeletonNodeDescriptionUpdate(
  layer: SegmentationUserLayer,
  options: {
    node: SpatiallyIndexedSkeletonNode;
    nextDescription?: string;
  },
) {
  const commandMappings = layer.spatialSkeletonState.commandHistory.mappings;
  const command = new NodeDescriptionCommand(
    layer,
    commandMappings.getStableOrCurrentNodeId(options.node.nodeId)!,
    commandMappings.getStableOrCurrentSegmentId(options.node.segmentId),
    options.node.description,
    options.nextDescription ?? options.node.description,
  );
  return layer.spatialSkeletonState.commandHistory.execute(command);
}

export function executeSpatialSkeletonNodeTrueEndUpdate(
  layer: SegmentationUserLayer,
  options: {
    node: SpatiallyIndexedSkeletonNode;
    nextIsTrueEnd: boolean;
  },
) {
  const commandMappings = layer.spatialSkeletonState.commandHistory.mappings;
  const command = new NodeTrueEndCommand(
    layer,
    commandMappings.getStableOrCurrentNodeId(options.node.nodeId)!,
    commandMappings.getStableOrCurrentSegmentId(options.node.segmentId),
    options.node.isTrueEnd,
    options.nextIsTrueEnd,
  );
  return layer.spatialSkeletonState.commandHistory.execute(command);
}

export function executeSpatialSkeletonNodePropertiesUpdate(
  layer: SegmentationUserLayer,
  options: {
    node: SpatiallyIndexedSkeletonNode;
    next: { radius: number; confidence: number };
  },
) {
  const commandMappings = layer.spatialSkeletonState.commandHistory.mappings;
  const command = new NodePropertiesCommand(
    layer,
    commandMappings.getStableOrCurrentNodeId(options.node.nodeId)!,
    commandMappings.getStableOrCurrentSegmentId(options.node.segmentId),
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
    SpatiallyIndexedSkeletonNode,
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
  const commandMappings = layer.spatialSkeletonState.commandHistory.mappings;
  const command = new RerootCommand(
    layer,
    commandMappings.getStableOrCurrentNodeId(node.nodeId)!,
    commandMappings.getStableOrCurrentSegmentId(node.segmentId),
    commandMappings.getStableOrCurrentNodeId(rootNode.nodeId)!,
  );
  return layer.spatialSkeletonState.commandHistory.execute(command);
}

export function executeSpatialSkeletonSplit(
  layer: SegmentationUserLayer,
  node: Pick<SpatiallyIndexedSkeletonNode, "nodeId" | "segmentId">,
) {
  const segmentNodes = layer.getCachedSpatialSkeletonSegmentNodesForEdit(
    node.segmentId,
  );
  const splitNode = findSpatiallyIndexedSkeletonNode(
    segmentNodes,
    node.nodeId,
  );
  if (splitNode === undefined) {
    throw new Error(
      `Node ${node.nodeId} is not available in the inspected skeleton cache.`,
    );
  }
  const commandMappings = layer.spatialSkeletonState.commandHistory.mappings;
  const command = new SplitCommand(
    layer,
    commandMappings.getStableOrCurrentNodeId(splitNode.nodeId)!,
    commandMappings.getStableOrCurrentSegmentId(splitNode.segmentId),
    commandMappings.getStableOrCurrentNodeId(splitNode.parentNodeId),
  );
  return layer.spatialSkeletonState.commandHistory.execute(command);
}

export function executeSpatialSkeletonMerge(
  layer: SegmentationUserLayer,
  firstNode: Pick<SpatiallyIndexedSkeletonNode, "nodeId" | "segmentId">,
  secondNode: Pick<SpatiallyIndexedSkeletonNode, "nodeId" | "segmentId">,
) {
  const commandMappings = layer.spatialSkeletonState.commandHistory.mappings;
  const command = new MergeCommand(
    layer,
    commandMappings.getStableOrCurrentNodeId(firstNode.nodeId)!,
    commandMappings.getStableOrCurrentSegmentId(firstNode.segmentId),
    commandMappings.getStableOrCurrentNodeId(secondNode.nodeId)!,
    commandMappings.getStableOrCurrentSegmentId(secondNode.segmentId),
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
