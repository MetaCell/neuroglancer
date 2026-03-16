export function resolveSpatiallyIndexedSkeletonSegmentPick(
  chunk: { indices: Uint32Array; numVertices: number },
  segmentIds: Uint32Array,
  pickedOffset: number,
  kind: "node" | "edge",
) {
  if (pickedOffset < 0) return undefined;
  if (kind === "node") {
    if (pickedOffset >= segmentIds.length || pickedOffset >= chunk.numVertices) {
      return undefined;
    }
    const segmentId = segmentIds[pickedOffset];
    return Number.isSafeInteger(segmentId) && segmentId > 0
      ? segmentId
      : undefined;
  }
  const indexOffset = pickedOffset * 2;
  if (indexOffset + 1 >= chunk.indices.length) {
    return undefined;
  }
  const vertexA = chunk.indices[indexOffset];
  const vertexB = chunk.indices[indexOffset + 1];
  let segmentId = segmentIds[vertexA];
  if (!Number.isSafeInteger(segmentId) || segmentId <= 0) {
    segmentId = segmentIds[vertexB];
  }
  return Number.isSafeInteger(segmentId) && segmentId > 0
    ? segmentId
    : undefined;
}
