export interface SpatiallyIndexedMissingConnection {
  nodeId: number;
  parentId: number;
  vertexIndex: number;
  skeletonId: number;
}

export interface SpatiallyIndexedEditableChunkData {
  nodeMap: Map<number, number>;
  positions: Float32Array;
  segmentIds: Uint32Array;
  indices: Uint32Array;
  missingConnections: SpatiallyIndexedMissingConnection[];
}

export interface SpatiallyIndexedEditableChunkEntry {
  chunkId: string;
  chunkKey: string;
  sourceId: string;
  data: SpatiallyIndexedEditableChunkData;
}

export interface SpatiallyIndexedRebuiltChunkConnections {
  chunkId: string;
  indices: Uint32Array;
  missingConnections: SpatiallyIndexedMissingConnection[];
}

export function appendNodeToSpatialChunk(
  data: SpatiallyIndexedEditableChunkData,
  options: {
    nodeId: number;
    segmentId: number;
    position: ArrayLike<number>;
  },
) {
  const newVertexIndex = data.positions.length / 3;
  const nextPositions = new Float32Array(data.positions.length + 3);
  nextPositions.set(data.positions);
  nextPositions[data.positions.length] = Number(options.position[0]);
  nextPositions[data.positions.length + 1] = Number(options.position[1]);
  nextPositions[data.positions.length + 2] = Number(options.position[2]);

  const nextSegmentIds = new Uint32Array(data.segmentIds.length + 1);
  nextSegmentIds.set(data.segmentIds);
  nextSegmentIds[data.segmentIds.length] = Math.round(options.segmentId);

  const nextNodeMap = new Map(data.nodeMap);
  nextNodeMap.set(options.nodeId, newVertexIndex);

  return {
    data: {
      nodeMap: nextNodeMap,
      positions: nextPositions,
      segmentIds: nextSegmentIds,
      indices: data.indices,
      missingConnections: data.missingConnections.slice(),
    },
    vertexIndex: newVertexIndex,
  };
}

export function updateNodePositionInSpatialChunk(
  data: SpatiallyIndexedEditableChunkData,
  nodeId: number,
  position: ArrayLike<number>,
) {
  const vertexIndex = data.nodeMap.get(nodeId);
  if (vertexIndex === undefined) {
    return undefined;
  }
  const nextPositions = new Float32Array(data.positions);
  const offset = vertexIndex * 3;
  nextPositions[offset] = Number(position[0]);
  nextPositions[offset + 1] = Number(position[1]);
  nextPositions[offset + 2] = Number(position[2]);
  return {
    data: {
      nodeMap: new Map(data.nodeMap),
      positions: nextPositions,
      segmentIds: new Uint32Array(data.segmentIds),
      indices: data.indices,
      missingConnections: data.missingConnections.slice(),
    },
    vertexIndex,
  };
}

export function removeNodeFromSpatialChunk(
  data: SpatiallyIndexedEditableChunkData,
  nodeId: number,
) {
  const removedVertexIndex = data.nodeMap.get(nodeId);
  if (removedVertexIndex === undefined) {
    return undefined;
  }
  const removedSegmentId = data.segmentIds[removedVertexIndex];
  const nextPositions = new Float32Array(Math.max(0, data.positions.length - 3));
  const nextSegmentIds = new Uint32Array(
    Math.max(0, data.segmentIds.length - 1),
  );
  let dstVertex = 0;
  for (let srcVertex = 0; srcVertex < data.segmentIds.length; ++srcVertex) {
    if (srcVertex === removedVertexIndex) {
      continue;
    }
    const srcOffset = srcVertex * 3;
    const dstOffset = dstVertex * 3;
    nextPositions[dstOffset] = data.positions[srcOffset];
    nextPositions[dstOffset + 1] = data.positions[srcOffset + 1];
    nextPositions[dstOffset + 2] = data.positions[srcOffset + 2];
    nextSegmentIds[dstVertex] = data.segmentIds[srcVertex];
    ++dstVertex;
  }

  const remapVertexIndex = (vertexIndex: number) =>
    vertexIndex > removedVertexIndex ? vertexIndex - 1 : vertexIndex;

  const nextNodeMap = new Map<number, number>();
  for (const [candidateNodeId, vertexIndex] of data.nodeMap.entries()) {
    if (candidateNodeId === nodeId) {
      continue;
    }
    nextNodeMap.set(candidateNodeId, remapVertexIndex(vertexIndex));
  }

  const nextIndices: number[] = [];
  for (let i = 0; i < data.indices.length; i += 2) {
    const childVertexIndex = data.indices[i];
    const parentVertexIndex = data.indices[i + 1];
    if (
      childVertexIndex === removedVertexIndex ||
      parentVertexIndex === removedVertexIndex
    ) {
      continue;
    }
    nextIndices.push(
      remapVertexIndex(childVertexIndex),
      remapVertexIndex(parentVertexIndex),
    );
  }

  const nextMissingConnections: SpatiallyIndexedMissingConnection[] = [];
  for (const connection of data.missingConnections) {
    if (connection.nodeId === nodeId) {
      continue;
    }
    if (connection.vertexIndex === removedVertexIndex) {
      continue;
    }
    nextMissingConnections.push({
      ...connection,
      vertexIndex: remapVertexIndex(connection.vertexIndex),
    });
  }

  return {
    data: {
      nodeMap: nextNodeMap,
      positions: nextPositions,
      segmentIds: nextSegmentIds,
      indices: new Uint32Array(nextIndices),
      missingConnections: nextMissingConnections,
    },
    removedVertexIndex,
    removedSegmentId,
  };
}

export function rebuildSpatialChunkConnections(
  chunks: readonly SpatiallyIndexedEditableChunkEntry[],
  parentByNodeId: ReadonlyMap<number, number | undefined>,
) {
  const locatorsBySource = new Map<
    string,
    Map<number, { chunkKey: string; vertexIndex: number }>
  >();
  for (const chunk of chunks) {
    let locators = locatorsBySource.get(chunk.sourceId);
    if (locators === undefined) {
      locators = new Map();
      locatorsBySource.set(chunk.sourceId, locators);
    }
    for (const [nodeId, vertexIndex] of chunk.data.nodeMap.entries()) {
      if (vertexIndex < 0 || vertexIndex >= chunk.data.segmentIds.length) {
        continue;
      }
      locators.set(nodeId, { chunkKey: chunk.chunkKey, vertexIndex });
    }
  }

  const rebuilt: SpatiallyIndexedRebuiltChunkConnections[] = [];
  for (const chunk of chunks) {
    const sourceLocators = locatorsBySource.get(chunk.sourceId) ?? new Map();
    const indices: number[] = [];
    const missingConnections: SpatiallyIndexedMissingConnection[] = [];
    for (const [nodeId, vertexIndex] of chunk.data.nodeMap.entries()) {
      if (vertexIndex < 0 || vertexIndex >= chunk.data.segmentIds.length) {
        continue;
      }
      const parentNodeId = parentByNodeId.get(nodeId);
      if (
        parentNodeId === undefined ||
        !Number.isFinite(parentNodeId) ||
        parentNodeId === nodeId
      ) {
        continue;
      }
      const parentLocator = sourceLocators.get(parentNodeId);
      if (parentLocator?.chunkKey === chunk.chunkKey) {
        indices.push(vertexIndex, parentLocator.vertexIndex);
        continue;
      }
      missingConnections.push({
        nodeId,
        parentId: parentNodeId,
        vertexIndex,
        skeletonId: chunk.data.segmentIds[vertexIndex],
      });
    }
    rebuilt.push({
      chunkId: chunk.chunkId,
      indices: new Uint32Array(indices),
      missingConnections,
    });
  }
  return rebuilt;
}
