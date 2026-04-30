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

import type { SpatialSkeletonAction } from "#src/skeleton/actions.js";
import type { SpatialSkeletonCommand } from "#src/skeleton/command_history.js";

export class SpatialSkeletonEditConflictError extends Error {
  constructor(detail?: string) {
    super(
      detail ??
        "The skeleton edit could not be applied because the source state is out of date.",
    );
    this.name = "SpatialSkeletonEditConflictError";
  }
}

export interface SpatiallyIndexedSkeletonNodeBase {
  nodeId: number;
  segmentId: number;
  position: Float32Array;
  parentNodeId?: number;
  sourceState?: unknown;
}

export interface SpatiallyIndexedSkeletonNode
  extends SpatiallyIndexedSkeletonNodeBase {
  radius?: number;
  confidence?: number;
  description?: string;
  isTrueEnd?: boolean;
}

export interface SpatiallyIndexedSkeletonOpenLeaf {
  nodeId: number;
  x: number;
  y: number;
  z: number;
  distance: number;
  creationTime?: string;
}

export interface SpatiallyIndexedSkeletonNavigationTarget {
  nodeId: number;
  x: number;
  y: number;
  z: number;
}

export interface SpatiallyIndexedSkeletonNodeSourceStateUpdate {
  nodeId: number;
  sourceState: unknown;
}

export interface SpatiallyIndexedSkeletonEditResult {
  nodeSourceStateUpdates?: readonly SpatiallyIndexedSkeletonNodeSourceStateUpdate[];
}

export interface SpatiallyIndexedSkeletonAddNodeResult
  extends SpatiallyIndexedSkeletonEditResult {
  nodeId: number;
  segmentId: number;
  sourceState?: unknown;
  parentSourceState?: unknown;
}

export type SpatiallyIndexedSkeletonInsertNodeResult =
  SpatiallyIndexedSkeletonAddNodeResult;

export interface SpatiallyIndexedSkeletonNodeSourceStateResult
  extends SpatiallyIndexedSkeletonEditResult {
  sourceState?: unknown;
}

export interface SpatiallyIndexedSkeletonDescriptionUpdateResult
  extends SpatiallyIndexedSkeletonNodeSourceStateResult {
  description?: string;
}

export type SpatiallyIndexedSkeletonDeleteNodeResult =
  SpatiallyIndexedSkeletonEditResult;

export type SpatiallyIndexedSkeletonRerootResult =
  SpatiallyIndexedSkeletonEditResult;

export interface SpatiallyIndexedSkeletonMergeResult
  extends SpatiallyIndexedSkeletonEditResult {
  resultSegmentId: number | undefined;
  deletedSegmentId: number | undefined;
  directionAdjusted: boolean;
}

export interface SpatiallyIndexedSkeletonSplitResult
  extends SpatiallyIndexedSkeletonEditResult {
  existingSegmentId: number | undefined;
  newSegmentId: number | undefined;
}

export interface SpatiallyIndexedSkeletonMetadata {
  bounds: {
    min: { x: number; y: number; z: number };
    max: { x: number; y: number; z: number };
  };
  resolution: { x: number; y: number; z: number };
  gridCellSizes: Array<{ x: number; y: number; z: number }>;
}

export interface SpatialSkeletonNodeFeatureCapabilities {
  description?: boolean;
  trueEnd?: boolean;
  radius?: boolean;
  confidenceValues?: readonly number[];
}

export interface SpatialSkeletonEditCapabilities {
  nodeFeatures?: SpatialSkeletonNodeFeatureCapabilities;
}

export interface SpatialSkeletonAddNodeCommandOptions {
  skeletonId: number;
  parentNodeId: number | undefined;
  positionInModelSpace: Float32Array;
}

export interface SpatialSkeletonMoveNodeCommandOptions {
  node: SpatiallyIndexedSkeletonNode;
  nextPositionInModelSpace: Float32Array;
}

export interface SpatialSkeletonNodeDescriptionCommandOptions {
  node: SpatiallyIndexedSkeletonNode;
  nextDescription?: string;
}

export interface SpatialSkeletonNodeTrueEndCommandOptions {
  node: SpatiallyIndexedSkeletonNode;
  nextIsTrueEnd: boolean;
}

export interface SpatialSkeletonNodePropertiesCommandOptions {
  node: SpatiallyIndexedSkeletonNode;
  next: { radius: number; confidence: number };
}

export interface SpatialSkeletonMergeEndpoint {
  nodeId: number;
  segmentId: number;
  sourceState?: unknown;
}

export interface SpatialSkeletonEditController {
  readonly capabilities?: SpatialSkeletonEditCapabilities;
  supports(action: SpatialSkeletonAction): boolean;
  createAddNodeCommand?(
    layer: any,
    options: SpatialSkeletonAddNodeCommandOptions,
  ): SpatialSkeletonCommand;
  createMoveNodeCommand?(
    layer: any,
    options: SpatialSkeletonMoveNodeCommandOptions,
  ): SpatialSkeletonCommand;
  createDeleteNodeCommand?(
    layer: any,
    node: SpatiallyIndexedSkeletonNode,
  ): SpatialSkeletonCommand;
  createNodeDescriptionCommand?(
    layer: any,
    options: SpatialSkeletonNodeDescriptionCommandOptions,
  ): SpatialSkeletonCommand;
  createNodeTrueEndCommand?(
    layer: any,
    options: SpatialSkeletonNodeTrueEndCommandOptions,
  ): SpatialSkeletonCommand;
  createNodePropertiesCommand?(
    layer: any,
    options: SpatialSkeletonNodePropertiesCommandOptions,
  ): SpatialSkeletonCommand;
  createRerootCommand?(
    layer: any,
    node: Pick<
      SpatiallyIndexedSkeletonNode,
      "nodeId" | "segmentId" | "parentNodeId"
    >,
  ): SpatialSkeletonCommand;
  createSplitCommand?(
    layer: any,
    node: Pick<SpatiallyIndexedSkeletonNode, "nodeId" | "segmentId">,
  ): SpatialSkeletonCommand;
  createMergeCommand?(
    layer: any,
    firstNode: SpatialSkeletonMergeEndpoint,
    secondNode: SpatialSkeletonMergeEndpoint,
  ): SpatialSkeletonCommand;
}

export interface SpatiallyIndexedSkeletonSource {
  readonly spatialSkeletonEditController?: SpatialSkeletonEditController;
  listSkeletons(): Promise<number[]>;
  getSkeleton(
    skeletonId: number,
    options?: { signal?: AbortSignal },
  ): Promise<SpatiallyIndexedSkeletonNode[]>;
  getSpatialIndexMetadata(): Promise<SpatiallyIndexedSkeletonMetadata | null>;
  fetchNodes(
    boundingBox: {
      min: { x: number; y: number; z: number };
      max: { x: number; y: number; z: number };
    },
    lod?: number,
    options?: {
      signal?: AbortSignal;
    },
  ): Promise<SpatiallyIndexedSkeletonNodeBase[]>;
}

export interface EditableSpatiallyIndexedSkeletonSource
  extends SpatiallyIndexedSkeletonSource {
  getSkeletonRootNode(
    skeletonId: number,
  ): Promise<SpatiallyIndexedSkeletonNavigationTarget>;
  addNode(
    skeletonId: number,
    x: number,
    y: number,
    z: number,
    parentId?: number,
    editContext?: unknown,
  ): Promise<SpatiallyIndexedSkeletonAddNodeResult>;
  insertNode(
    skeletonId: number,
    x: number,
    y: number,
    z: number,
    parentId: number,
    childNodeIds: readonly number[],
    editContext?: unknown,
  ): Promise<SpatiallyIndexedSkeletonInsertNodeResult>;
  moveNode(
    nodeId: number,
    x: number,
    y: number,
    z: number,
    editContext?: unknown,
  ): Promise<SpatiallyIndexedSkeletonNodeSourceStateResult>;
  deleteNode(
    nodeId: number,
    options: {
      childNodeIds?: readonly number[];
      editContext?: unknown;
    },
  ): Promise<SpatiallyIndexedSkeletonDeleteNodeResult>;
  rerootSkeleton?(
    nodeId: number,
    editContext?: unknown,
  ): Promise<SpatiallyIndexedSkeletonRerootResult>;
  updateDescription(
    nodeId: number,
    description: string,
  ): Promise<SpatiallyIndexedSkeletonDescriptionUpdateResult>;
  setTrueEnd(
    nodeId: number,
  ): Promise<SpatiallyIndexedSkeletonNodeSourceStateResult>;
  removeTrueEnd(
    nodeId: number,
  ): Promise<SpatiallyIndexedSkeletonNodeSourceStateResult>;
  updateRadius(
    nodeId: number,
    radius: number,
    editContext?: unknown,
  ): Promise<SpatiallyIndexedSkeletonNodeSourceStateResult>;
  updateConfidence(
    nodeId: number,
    confidence: number,
    editContext?: unknown,
  ): Promise<SpatiallyIndexedSkeletonNodeSourceStateResult>;
  mergeSkeletons(
    fromNodeId: number,
    toNodeId: number,
    editContext?: unknown,
  ): Promise<SpatiallyIndexedSkeletonMergeResult>;
  splitSkeleton(
    nodeId: number,
    editContext?: unknown,
  ): Promise<SpatiallyIndexedSkeletonSplitResult>;
}
