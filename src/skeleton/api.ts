export interface SpatiallyIndexedSkeletonNode {
    id: number;
    parent_id: number | null;
    x: number;
    y: number;
    z: number;
    skeleton_id: number;
}


export interface SpatiallyIndexedSkeletonSource {
    listSkeletons(): Promise<number[]>;
    getSkeleton(skeletonId: number): Promise<SpatiallyIndexedSkeletonNode[]>;
    getDimensions(): Promise<{ min: { x: number; y: number; z: number }; max: { x: number; y: number; z: number } } | null>;
    getResolution(): Promise<{ x: number; y: number; z: number } | null>;
    getGridCellSizes(): Promise<Array<{ x: number; y: number; z: number }>>;
    fetchNodes(boundingBox: { min: { x: number, y: number, z: number }, max: { x: number, y: number, z: number } }, lod?: number): Promise<SpatiallyIndexedSkeletonNode[]>;
    addNode(
        skeletonId: number,
        x: number,
        y: number,
        z: number,
        parentId?: number,
    ): Promise<number>;
    moveNode(nodeId: number, x: number, y: number, z: number): Promise<void>;
    deleteNode(nodeId: number): Promise<void>;

    mergeSkeletons(skeletonId1: number, skeletonId2: number): Promise<void>;
    splitSkeleton(nodeId: number): Promise<void>;
}
