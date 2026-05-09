import { describe, expect, it } from "vitest";

import { forEachSpatiallyIndexedScale } from "#src/spatially_indexed_chunk.js";
import { mat4 } from "#src/util/geom.js";

function makeProjection(width: number, height: number) {
  return {
    displayDimensionRenderInfo: {
      voxelPhysicalScales: Float32Array.of(1, 1, 1),
    },
    projectionMat: mat4.create(),
    viewMatrix: mat4.create(),
    width,
    height,
  } as any;
}

function makeTransformedSource(limit: number) {
  return {
    source: {
      spec: {
        limit,
        rank: 3,
      },
    },
    chunkLayout: {
      size: Float32Array.of(1, 1, 1),
      detTransform: 1,
    },
    lowerClipDisplayBound: Float32Array.of(0, 0, 0),
    upperClipDisplayBound: Float32Array.of(1, 1, 1),
    nonDisplayLowerClipBound: Float32Array.of(0, 0, 0),
    nonDisplayUpperClipBound: Float32Array.of(
      Number.POSITIVE_INFINITY,
      Number.POSITIVE_INFINITY,
      Number.POSITIVE_INFINITY,
    ),
  } as any;
}

describe("spatially indexed chunk scale selection", () => {
  it("selects coarse-to-fine levels until the target density is met", () => {
    const visited: Array<{ index: number; drawFraction: number }> = [];
    const sources = [
      makeTransformedSource(8),
      makeTransformedSource(1),
    ] as const;

    forEachSpatiallyIndexedScale(
      makeProjection(10, 10),
      5,
      sources,
      (_source, index, drawFraction) => {
        visited.push({ index, drawFraction });
      },
    );

    expect(visited).toEqual([
      { index: 1, drawFraction: 1 },
      { index: 0, drawFraction: 0.375 },
    ]);
  });

  it("preserves annotation behavior for zero-limit levels", () => {
    const visited: Array<{ index: number; drawFraction: number }> = [];

    forEachSpatiallyIndexedScale(
      makeProjection(10, 10),
      5,
      [makeTransformedSource(0)],
      (_source, index, drawFraction) => {
        visited.push({ index, drawFraction });
      },
    );

    expect(visited).toEqual([{ index: 0, drawFraction: 1 }]);
  });
});
