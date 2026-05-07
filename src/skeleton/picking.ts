export function resolveSpatiallyIndexedSkeletonSegmentPick(
  chunk: { indices: Uint32Array; numVertices: number },
  segmentIds: BigUint64Array,
  pickedOffset: number,
  kind: "node" | "edge",
) {
  if (pickedOffset < 0) return undefined;
  if (kind === "node") {
    if (
      pickedOffset >= segmentIds.length ||
      pickedOffset >= chunk.numVertices
    ) {
      return undefined;
    }
    const segmentId = segmentIds[pickedOffset];
    return segmentId > 0n ? segmentId : undefined;
  }
  const indexOffset = pickedOffset * 2;
  if (indexOffset + 1 >= chunk.indices.length) {
    return undefined;
  }
  const vertexA = chunk.indices[indexOffset];
  const vertexB = chunk.indices[indexOffset + 1];
  let segmentId = segmentIds[vertexA];
  if (segmentId <= 0n) {
    segmentId = segmentIds[vertexB];
  }
  return segmentId > 0n ? segmentId : undefined;
}
