export interface SpatiallyIndexedSkeletonNode {
  id: number;
  parent_id: number | null;
  x: number;
  y: number;
  z: number;
  skeleton_id: number;
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

export interface SpatiallyIndexedSkeletonSource {
  listSkeletons(): Promise<number[]>;
  getSkeleton(skeletonId: number): Promise<SpatiallyIndexedSkeletonNode[]>;
  getSkeletonRootNode(
    skeletonId: number,
  ): Promise<SpatiallyIndexedSkeletonNavigationTarget>;
  getPreviousBranchOrRoot(
    nodeId: number,
    options?: { alt?: boolean },
  ): Promise<SpatiallyIndexedSkeletonNavigationTarget>;
  getNextBranchOrEnd(
    nodeId: number,
  ): Promise<SpatiallyIndexedSkeletonBranchNavigationTarget[]>;
  getOpenLeaves(
    skeletonId: number,
    nodeId: number,
  ): Promise<SpatiallyIndexedSkeletonOpenLeaf[]>;
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
  addNode(
    skeletonId: number,
    x: number,
    y: number,
    z: number,
    parentId?: number,
  ): Promise<number>;
  moveNode(nodeId: number, x: number, y: number, z: number): Promise<void>;
  deleteNode(
    nodeId: number,
    options: {
      parentNodeId?: number;
      childNodeIds?: readonly number[];
    },
  ): Promise<void>;

  mergeSkeletons(skeletonId1: number, skeletonId2: number): Promise<void>;
  splitSkeleton(nodeId: number): Promise<void>;
}

export interface EditableSpatiallyIndexedSkeletonSource
  extends SpatiallyIndexedSkeletonSource
{
  addNodeWithInfo(
    skeletonId: number,
    x: number,
    y: number,
    z: number,
    parentId?: number,
  ): Promise<SpatiallyIndexedSkeletonAddNodeResult>;
  addNodeLabel(nodeId: number, label: string): Promise<void>;
  removeNodeLabel(nodeId: number, label: string): Promise<void>;
  mergeSkeletonsWithInfo(
    fromNodeId: number,
    toNodeId: number,
  ): Promise<SpatiallyIndexedSkeletonMergeResult>;
  splitSkeletonWithInfo(
    nodeId: number,
  ): Promise<SpatiallyIndexedSkeletonSplitResult>;
}
