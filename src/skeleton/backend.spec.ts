import { describe, expect, it } from "vitest";

import {
  getSpatiallyIndexedSkeletonChunkPriority,
  SpatiallyIndexedSkeletonChunk,
  SpatiallyIndexedSkeletonSourceBackend,
} from "#src/skeleton/backend.js";

describe("skeleton/backend chunk priority", () => {
  it("uses the standard chunk-origin distance rule for 3d chunks", () => {
    expect(
      getSpatiallyIndexedSkeletonChunkPriority(
        Float32Array.of(3, 4, 0),
        Float32Array.of(2, 5, 1),
        Float32Array.of(1, 0, 0),
      ),
    ).toBeCloseTo(-Math.sqrt(17));
  });

  it("prioritizes chunks nearer the view center ahead of farther chunks", () => {
    const localCenter = Float32Array.of(10, 20, 30);
    const chunkSize = Float32Array.of(4, 4, 8);
    const nearChunk = Float32Array.of(2, 5, 4);
    const farChunk = Float32Array.of(5, 1, 0);

    expect(
      getSpatiallyIndexedSkeletonChunkPriority(
        localCenter,
        chunkSize,
        nearChunk,
      ),
    ).toBeGreaterThan(
      getSpatiallyIndexedSkeletonChunkPriority(
        localCenter,
        chunkSize,
        farChunk,
      ),
    );
  });

  it("keys spatial chunks by source grid position", () => {
    const source = Object.assign(
      Object.create(SpatiallyIndexedSkeletonSourceBackend.prototype),
      {
        chunks: new Map<string, SpatiallyIndexedSkeletonChunk>(),
        chunkConstructor: SpatiallyIndexedSkeletonChunk,
        getNewChunk_(
          this: SpatiallyIndexedSkeletonSourceBackend,
          ChunkType: typeof SpatiallyIndexedSkeletonChunk,
        ) {
          const chunk = new ChunkType();
          chunk.source = this;
          return chunk;
        },
        addChunk(
          this: SpatiallyIndexedSkeletonSourceBackend,
          chunk: SpatiallyIndexedSkeletonChunk,
        ) {
          this.chunks.set(chunk.key!, chunk);
        },
      },
    ) as SpatiallyIndexedSkeletonSourceBackend;

    const first = source.getChunk(Float32Array.of(1, 2, 3));
    const second = source.getChunk(Float32Array.of(1, 2, 3));

    expect(second).toBe(first);
    expect(first.key).toBe("1,2,3");
    expect(source.chunks.has("1,2,3:0")).toBe(false);
  });
});
