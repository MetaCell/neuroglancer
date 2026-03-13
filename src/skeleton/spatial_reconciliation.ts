import * as matrix from "#src/util/matrix.js";

export function computeSpatiallyIndexedOwnerChunkKey(
  multiscaleToChunkTransform: Float32Array,
  position: ArrayLike<number>,
  rank: number,
  lod: number | undefined,
) {
  if (lod === undefined) {
    return undefined;
  }
  const input = new Float32Array(rank);
  for (let i = 0; i < rank; ++i) {
    input[i] = Number(position[i] ?? 0);
  }
  const point = new Float32Array(rank);
  matrix.transformPoint(
    point,
    multiscaleToChunkTransform,
    rank + 1,
    input,
    rank,
  );
  for (let i = 0; i < rank; ++i) {
    point[i] = Math.floor(point[i]);
  }
  return `${point.join()}:${lod}`;
}

export function committedMoveSourcesSatisfied(
  states: Iterable<{
    oldOwnerSatisfied: boolean;
    newOwnerSatisfied: boolean;
  }>,
) {
  for (const state of states) {
    if (!state.oldOwnerSatisfied || !state.newOwnerSatisfied) {
      return false;
    }
  }
  return true;
}

export function committedAddSourcesSatisfied(
  states: Iterable<{
    ownerSatisfied: boolean;
  }>,
) {
  for (const state of states) {
    if (!state.ownerSatisfied) {
      return false;
    }
  }
  return true;
}
