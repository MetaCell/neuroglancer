import { describe, expect, it } from "vitest";

import { ChunkState } from "#src/chunk_manager/base.js";
import type { ChunkManager } from "#src/chunk_manager/backend.js";
import {
  cancelStaleSpatiallyIndexedSkeletonDownloads,
  getSpatiallyIndexedSkeletonChunkPriority,
  markSpatiallyIndexedSkeletonChunkRequested,
  SpatiallyIndexedSkeletonChunk,
  SpatiallyIndexedSkeletonSourceBackend,
} from "#src/skeleton/backend.js";

function makeSpatiallyIndexedSkeletonSource() {
  return Object.assign(
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
      chunkStateChanged() {},
    },
  ) as SpatiallyIndexedSkeletonSourceBackend;
}

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
    const source = makeSpatiallyIndexedSkeletonSource();

    const first = source.getChunk(Float32Array.of(1, 2, 3));
    const second = source.getChunk(Float32Array.of(1, 2, 3));

    expect(second).toBe(first);
    expect(first.key).toBe("1,2,3");
    expect(first.requestGeneration).toBe(-1);
    expect(source.chunks.has("1,2,3:0")).toBe(false);
  });
});

describe("skeleton/backend stale spatial skeleton downloads", () => {
  it("marks chunks with the priority recompute generation that requested them", () => {
    const chunk = new SpatiallyIndexedSkeletonChunk();

    markSpatiallyIndexedSkeletonChunkRequested(chunk, 7);

    expect(chunk.requestGeneration).toBe(7);
  });

  it("aborts downloading spatial skeleton chunks not requested this generation", () => {
    const source = makeSpatiallyIndexedSkeletonSource();
    const chunk = source.getChunk(Float32Array.of(1, 2, 3));
    const abortController = new AbortController();
    const stateUpdates: ChunkState[] = [];
    const chunkManager = {
      queueManager: {
        updateChunkState(
          updatedChunk: SpatiallyIndexedSkeletonChunk,
          newState: ChunkState,
        ) {
          stateUpdates.push(newState);
          updatedChunk.state = newState;
        },
      },
    } as unknown as ChunkManager;
    chunk.requestGeneration = 1;
    chunk.downloadAbortController = abortController;
    chunk.state = ChunkState.DOWNLOADING;

    cancelStaleSpatiallyIndexedSkeletonDownloads(chunkManager, [source], 2);

    expect(abortController.signal.aborted).toBe(true);
    expect(chunk.downloadAbortController).toBeUndefined();
    expect(stateUpdates).toEqual([ChunkState.QUEUED]);
    expect(chunk.state).toBe(ChunkState.QUEUED);
  });

  it("keeps current-generation spatial skeleton downloads in flight", () => {
    const source = makeSpatiallyIndexedSkeletonSource();
    const chunk = source.getChunk(Float32Array.of(1, 2, 3));
    const abortController = new AbortController();
    const stateUpdates: ChunkState[] = [];
    const chunkManager = {
      queueManager: {
        updateChunkState() {
          stateUpdates.push(ChunkState.QUEUED);
        },
      },
    } as unknown as ChunkManager;
    chunk.requestGeneration = 2;
    chunk.downloadAbortController = abortController;
    chunk.state = ChunkState.DOWNLOADING;

    cancelStaleSpatiallyIndexedSkeletonDownloads(chunkManager, [source], 2);

    expect(abortController.signal.aborted).toBe(false);
    expect(chunk.downloadAbortController).toBe(abortController);
    expect(stateUpdates).toEqual([]);
    expect(chunk.state).toBe(ChunkState.DOWNLOADING);
  });
});
