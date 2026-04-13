export type SpatiallyIndexedSkeletonRevisionToken = string | number;

export interface SpatiallyIndexedSkeletonNode {
  id: number;
  parent_id: number | null;
  x: number;
  y: number;
  z: number;
  skeleton_id: number;
  radius?: number;
  confidence?: number;
  labels?: string[];
  revisionToken?: SpatiallyIndexedSkeletonRevisionToken;
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

export interface SpatiallyIndexedSkeletonBranchNavigationTarget {
  child: SpatiallyIndexedSkeletonNavigationTarget;
  branchStartOrEnd: SpatiallyIndexedSkeletonNavigationTarget;
  branchEnd: SpatiallyIndexedSkeletonNavigationTarget;
}

export interface SpatiallyIndexedSkeletonAddNodeResult {
  treenodeId: number;
  skeletonId: number;
  revisionToken?: SpatiallyIndexedSkeletonRevisionToken;
  parentRevisionToken?: SpatiallyIndexedSkeletonRevisionToken;
}

export interface SpatiallyIndexedSkeletonNodeRevisionUpdate {
  nodeId: number;
  revisionToken: SpatiallyIndexedSkeletonRevisionToken;
}

export interface SpatiallyIndexedSkeletonInsertNodeResult
  extends SpatiallyIndexedSkeletonAddNodeResult {
  childRevisionUpdates?: readonly SpatiallyIndexedSkeletonNodeRevisionUpdate[];
}

export interface SpatiallyIndexedSkeletonNodeRevisionResult {
  revisionToken?: SpatiallyIndexedSkeletonRevisionToken;
}

export interface SpatiallyIndexedSkeletonDeleteNodeResult {
  childRevisionUpdates?: readonly SpatiallyIndexedSkeletonNodeRevisionUpdate[];
}

export interface SpatiallyIndexedSkeletonEditNodeContext {
  nodeId: number;
  parentNodeId?: number;
  revisionToken: SpatiallyIndexedSkeletonRevisionToken;
}

export interface SpatiallyIndexedSkeletonEditParentContext {
  nodeId: number;
  revisionToken: SpatiallyIndexedSkeletonRevisionToken;
}

export interface SpatiallyIndexedSkeletonEditContext {
  node?: SpatiallyIndexedSkeletonEditNodeContext;
  parent?: SpatiallyIndexedSkeletonEditParentContext;
  children?: readonly SpatiallyIndexedSkeletonEditParentContext[];
  nodes?: readonly SpatiallyIndexedSkeletonEditParentContext[];
}

export interface SpatiallyIndexedSkeletonMergeResult {
  resultSkeletonId: number | undefined;
  deletedSkeletonId: number | undefined;
  stableAnnotationSwap: boolean;
}

export interface SpatiallyIndexedSkeletonSplitResult {
  existingSkeletonId: number | undefined;
  newSkeletonId: number | undefined;
}

export interface SpatiallyIndexedSkeletonDescriptionUpdateOptions {
  trueEnd: boolean;
}

export interface SpatiallyIndexedSkeletonConfidencePropertyEditingOptions {
  values: readonly number[];
}

export interface SpatiallyIndexedSkeletonPropertyEditingOptions {
  confidence?: SpatiallyIndexedSkeletonConfidencePropertyEditingOptions;
}

export interface SpatiallyIndexedSkeletonSource {
  listSkeletons(): Promise<number[]>;
  getSkeleton(
    skeletonId: number,
    options?: { signal?: AbortSignal },
  ): Promise<SpatiallyIndexedSkeletonNode[]>;
  getDimensions(): Promise<{
    min: { x: number; y: number; z: number };
    max: { x: number; y: number; z: number };
  } | null>;
  getResolution(): Promise<{ x: number; y: number; z: number } | null>;
  getGridCellSizes(): Promise<Array<{ x: number; y: number; z: number }>>;
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
  ): Promise<SpatiallyIndexedSkeletonNode[]>;
}

export interface EditableSpatiallyIndexedSkeletonSource
  extends SpatiallyIndexedSkeletonSource {
  getPropertyEditingOptions?(): SpatiallyIndexedSkeletonPropertyEditingOptions;
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
  ): Promise<void>;
  updateDescription(
    nodeId: number,
    description: string,
    options: SpatiallyIndexedSkeletonDescriptionUpdateOptions,
  ): Promise<SpatiallyIndexedSkeletonNodeRevisionResult>;
  setTrueEnd(nodeId: number): Promise<SpatiallyIndexedSkeletonNodeRevisionResult>;
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
