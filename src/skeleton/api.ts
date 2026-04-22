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

export interface SpatiallyIndexedSkeletonNodeBase {
  nodeId: number;
  segmentId: number;
  position: Float32Array;
  parentNodeId?: number;
  revisionToken?: string;
}

export interface SpatiallyIndexedSkeletonNode
  extends SpatiallyIndexedSkeletonNodeBase {
  radius?: number;
  confidence?: number;
  description?: string;
  isTrueEnd: boolean;
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

export interface SpatiallyIndexedSkeletonNodeRevisionUpdate {
  nodeId: number;
  revisionToken: string;
}

export interface SpatiallyIndexedSkeletonEditResult {
  nodeRevisionUpdates?: readonly SpatiallyIndexedSkeletonNodeRevisionUpdate[];
}

export interface SpatiallyIndexedSkeletonAddNodeResult
  extends SpatiallyIndexedSkeletonEditResult {
  treenodeId: number;
  skeletonId: number;
  revisionToken?: string;
  parentRevisionToken?: string;
}

export interface SpatiallyIndexedSkeletonInsertNodeResult
  extends SpatiallyIndexedSkeletonAddNodeResult {}

export interface SpatiallyIndexedSkeletonNodeRevisionResult
  extends SpatiallyIndexedSkeletonEditResult {
  revisionToken?: string;
}

export interface SpatiallyIndexedSkeletonDescriptionUpdateResult
  extends SpatiallyIndexedSkeletonNodeRevisionResult {
  description?: string;
}

export interface SpatiallyIndexedSkeletonDeleteNodeResult
  extends SpatiallyIndexedSkeletonEditResult {}

export interface SpatiallyIndexedSkeletonRerootResult
  extends SpatiallyIndexedSkeletonEditResult {}

export interface SpatiallyIndexedSkeletonEditNodeContext {
  nodeId: number;
  parentNodeId?: number;
  revisionToken: string;
}

export interface SpatiallyIndexedSkeletonEditParentContext {
  nodeId: number;
  revisionToken: string;
}

export interface SpatiallyIndexedSkeletonEditContext {
  node?: SpatiallyIndexedSkeletonEditNodeContext;
  parent?: SpatiallyIndexedSkeletonEditParentContext;
  children?: readonly SpatiallyIndexedSkeletonEditParentContext[];
  nodes?: readonly SpatiallyIndexedSkeletonEditParentContext[];
}

export interface SpatiallyIndexedSkeletonMergeResult
  extends SpatiallyIndexedSkeletonEditResult {
  resultSkeletonId: number | undefined;
  deletedSkeletonId: number | undefined;
  stableAnnotationSwap: boolean;
}

export interface SpatiallyIndexedSkeletonSplitResult
  extends SpatiallyIndexedSkeletonEditResult {
  existingSkeletonId: number | undefined;
  newSkeletonId: number | undefined;
}

export interface SpatiallyIndexedSkeletonMetadata {
  bounds: {
    min: { x: number; y: number; z: number };
    max: { x: number; y: number; z: number };
  };
  resolution: { x: number; y: number; z: number };
  gridCellSizes: Array<{ x: number; y: number; z: number }>;
}

export const SPATIALLY_INDEXED_SKELETON_CONFIDENCE_VALUES = [
  0,
  25,
  50,
  75,
  100,
] as const;

export interface SpatiallyIndexedSkeletonSource {
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
      cacheProvider?: string;
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
    editContext?: SpatiallyIndexedSkeletonEditContext,
  ): Promise<SpatiallyIndexedSkeletonAddNodeResult>;
  insertNode(
    skeletonId: number,
    x: number,
    y: number,
    z: number,
    parentId: number,
    childNodeIds: readonly number[],
    editContext?: SpatiallyIndexedSkeletonEditContext,
  ): Promise<SpatiallyIndexedSkeletonInsertNodeResult>;
  moveNode(
    nodeId: number,
    x: number,
    y: number,
    z: number,
    editContext?: SpatiallyIndexedSkeletonEditContext,
  ): Promise<SpatiallyIndexedSkeletonNodeRevisionResult>;
  deleteNode(
    nodeId: number,
    options: {
      childNodeIds?: readonly number[];
      editContext?: SpatiallyIndexedSkeletonEditContext;
    },
  ): Promise<SpatiallyIndexedSkeletonDeleteNodeResult>;
  rerootSkeleton?(
    nodeId: number,
    editContext?: SpatiallyIndexedSkeletonEditContext,
  ): Promise<SpatiallyIndexedSkeletonRerootResult>;
  updateDescription(
    nodeId: number,
    description: string,
  ): Promise<SpatiallyIndexedSkeletonDescriptionUpdateResult>;
  setTrueEnd(
    nodeId: number,
  ): Promise<SpatiallyIndexedSkeletonNodeRevisionResult>;
  removeTrueEnd(
    nodeId: number,
  ): Promise<SpatiallyIndexedSkeletonNodeRevisionResult>;
  updateRadius(
    nodeId: number,
    radius: number,
    editContext?: SpatiallyIndexedSkeletonEditContext,
  ): Promise<SpatiallyIndexedSkeletonNodeRevisionResult>;
  updateConfidence(
    nodeId: number,
    confidence: number,
    editContext?: SpatiallyIndexedSkeletonEditContext,
  ): Promise<SpatiallyIndexedSkeletonNodeRevisionResult>;
  mergeSkeletons(
    fromNodeId: number,
    toNodeId: number,
    editContext?: SpatiallyIndexedSkeletonEditContext,
  ): Promise<SpatiallyIndexedSkeletonMergeResult>;
  splitSkeleton(
    nodeId: number,
    editContext?: SpatiallyIndexedSkeletonEditContext,
  ): Promise<SpatiallyIndexedSkeletonSplitResult>;
}
