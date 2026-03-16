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

export interface SpatiallyIndexedSkeletonSource {
  listSkeletons(): Promise<number[]>;
  getSkeleton(skeletonId: number): Promise<SpatiallyIndexedSkeletonNode[]>;
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
      includeLabels?: boolean;
    },
  ): Promise<SpatiallyIndexedSkeletonNode[]>;
}

export interface EditableSpatiallyIndexedSkeletonSource
  extends SpatiallyIndexedSkeletonSource {
  addNode(
    skeletonId: number,
    x: number,
    y: number,
    z: number,
    parentId?: number,
  ): Promise<SpatiallyIndexedSkeletonAddNodeResult>;
  moveNode(nodeId: number, x: number, y: number, z: number): Promise<void>;
  deleteNode(
    nodeId: number,
    options: {
      parentNodeId?: number;
      childNodeIds?: readonly number[];
    },
  ): Promise<void>;
  updateDescription(
    nodeId: number,
    description: string,
    options: SpatiallyIndexedSkeletonDescriptionUpdateOptions,
  ): Promise<void>;
  setTrueEnd(nodeId: number): Promise<void>;
  removeTrueEnd(nodeId: number): Promise<void>;
  updateRadius(nodeId: number, radius: number): Promise<void>;
  updateConfidence(nodeId: number, confidence: number): Promise<void>;
  mergeSkeletons(
    fromNodeId: number,
    toNodeId: number,
  ): Promise<SpatiallyIndexedSkeletonMergeResult>;
  splitSkeleton(nodeId: number): Promise<SpatiallyIndexedSkeletonSplitResult>;
}
