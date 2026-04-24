import { describe, expect, it } from "vitest";

import { getDefaultSpatiallyIndexedSkeletonChunkSize } from "#src/skeleton/spatial_chunk_sizing.js";

describe("skeleton/spatial_chunk_sizing", () => {
  it("derives an isotropic chunk size that stays within the default chunk budget", () => {
    expect(
      getDefaultSpatiallyIndexedSkeletonChunkSize({
        min: { x: 5, y: 6, z: 7 },
        max: { x: 25, y: 66, z: 127 },
      }),
    ).toEqual({ x: 15, y: 15, z: 15 });
  });

  it("handles elongated bounds while keeping the chunk size isotropic", () => {
    expect(
      getDefaultSpatiallyIndexedSkeletonChunkSize({
        min: { x: 0, y: 0, z: 0 },
        max: { x: 1000, y: 10, z: 10 },
      }),
    ).toEqual({ x: 16, y: 16, z: 16 });
  });

  it("returns the minimum chunk size for tiny bounds", () => {
    expect(
      getDefaultSpatiallyIndexedSkeletonChunkSize({
        min: { x: 0, y: 0, z: 0 },
        max: { x: 2, y: 2, z: 2 },
      }),
    ).toEqual({ x: 1, y: 1, z: 1 });
  });

  it("supports overriding the chunk budget", () => {
    expect(
      getDefaultSpatiallyIndexedSkeletonChunkSize(
        {
          min: { x: 0, y: 0, z: 0 },
          max: { x: 100, y: 100, z: 100 },
        },
        { maxChunks: 8 },
      ),
    ).toEqual({ x: 50, y: 50, z: 50 });
  });

  it("rejects NaN bounds", () => {
    expect(() =>
      getDefaultSpatiallyIndexedSkeletonChunkSize({
        min: { x: Number.NaN, y: 0, z: 0 },
        max: { x: 10, y: 10, z: 10 },
      }),
    ).toThrow(/bounds must be finite/i);
  });

  it("rejects infinite bounds", () => {
    expect(() =>
      getDefaultSpatiallyIndexedSkeletonChunkSize({
        min: { x: 0, y: 0, z: 0 },
        max: { x: Number.POSITIVE_INFINITY, y: 10, z: 10 },
      }),
    ).toThrow(/bounds must be finite/i);
  });

  it("rejects NaN minChunkSize", () => {
    expect(() =>
      getDefaultSpatiallyIndexedSkeletonChunkSize(
        {
          min: { x: 0, y: 0, z: 0 },
          max: { x: 10, y: 10, z: 10 },
        },
        { minChunkSize: Number.NaN },
      ),
    ).toThrow(/minChunkSize must be finite/i);
  });

  it("rejects infinite maxChunks", () => {
    expect(() =>
      getDefaultSpatiallyIndexedSkeletonChunkSize(
        {
          min: { x: 0, y: 0, z: 0 },
          max: { x: 10, y: 10, z: 10 },
        },
        { maxChunks: Number.POSITIVE_INFINITY },
      ),
    ).toThrow(/maxChunks must be finite/i);
  });
});
