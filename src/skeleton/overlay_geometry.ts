export interface SpatiallyIndexedSkeletonOverlayNodeLike {
  nodeId: number;
  segmentId: number;
  position: ArrayLike<number>;
  parentNodeId?: number;
}

export interface SpatiallyIndexedSkeletonOverlayGeometry {
  positions: Float32Array;
  segmentIds: Uint32Array;
  selected: Float32Array;
  nodeIds: Int32Array;
  nodePositions: Float32Array;
  pickSegmentIds: Uint32Array;
  pickEdgeSegmentIds: Uint32Array;
  indices: Uint32Array;
  numVertices: number;
}

export function buildSpatiallyIndexedSkeletonOverlayGeometry(
  segmentNodeSets: readonly (readonly SpatiallyIndexedSkeletonOverlayNodeLike[])[],
  options: {
    selectedNodeId?: number;
    getPendingNodePosition?: (nodeId: number) => ArrayLike<number> | undefined;
  } = {},
): SpatiallyIndexedSkeletonOverlayGeometry {
  const { selectedNodeId, getPendingNodePosition } = options;
  const nodeIndex = new Map<number, number>();
  const orderedNodes: SpatiallyIndexedSkeletonOverlayNodeLike[] = [];

  for (const segmentNodes of segmentNodeSets) {
    for (const node of segmentNodes) {
      if (nodeIndex.has(node.nodeId)) continue;
      nodeIndex.set(node.nodeId, orderedNodes.length);
      orderedNodes.push(node);
    }
  }

  const numVertices = orderedNodes.length;
  const positions = new Float32Array(numVertices * 3);
  const segmentIds = new Uint32Array(numVertices);
  const selected = new Float32Array(numVertices);
  const nodeIds = new Int32Array(numVertices);
  const nodePositions = new Float32Array(numVertices * 3);
  const pickSegmentIds = new Uint32Array(numVertices);
  const indices: number[] = [];
  const pickEdgeSegmentIds: number[] = [];

  orderedNodes.forEach((node, index) => {
    const position = getPendingNodePosition?.(node.nodeId) ?? node.position;
    const baseOffset = index * 3;
    positions[baseOffset] = Number(position[0] ?? 0);
    positions[baseOffset + 1] = Number(position[1] ?? 0);
    positions[baseOffset + 2] = Number(position[2] ?? 0);
    nodePositions[baseOffset] = positions[baseOffset];
    nodePositions[baseOffset + 1] = positions[baseOffset + 1];
    nodePositions[baseOffset + 2] = positions[baseOffset + 2];
    segmentIds[index] = Math.max(0, Math.round(Number(node.segmentId)));
    pickSegmentIds[index] = segmentIds[index];
    nodeIds[index] = Math.round(Number(node.nodeId));
    selected[index] =
      selectedNodeId !== undefined && node.nodeId === selectedNodeId ? 1 : 0;
  });

  orderedNodes.forEach((node) => {
    const childIndex = nodeIndex.get(node.nodeId);
    if (childIndex === undefined) return;
    const parentNodeId = node.parentNodeId;
    if (
      parentNodeId === undefined ||
      !Number.isSafeInteger(parentNodeId) ||
      parentNodeId <= 0
    ) {
      return;
    }
    const parentIndex = nodeIndex.get(parentNodeId);
    if (parentIndex === undefined) return;
    indices.push(childIndex, parentIndex);
    pickEdgeSegmentIds.push(segmentIds[childIndex] || segmentIds[parentIndex]);
  });

  return {
    positions,
    segmentIds,
    selected,
    nodeIds,
    nodePositions,
    pickSegmentIds,
    pickEdgeSegmentIds: new Uint32Array(pickEdgeSegmentIds),
    indices: new Uint32Array(indices),
    numVertices,
  };
}
