import { describe, expect, it, vi } from "vitest";

import { ChunkState } from "#src/chunk_manager/base.js";
import {
  cancelStaleSpatiallyIndexedSkeletonDownloads,
  markSpatiallyIndexedSkeletonChunkRequested,
  SpatiallyIndexedSkeletonChunkRequestOwner,
} from "#src/skeleton/backend.js";

describe("skeleton/backend stale LOD cancellation", () => {
  function makeChunk(state = ChunkState.DOWNLOADING) {
    return {
      state,
      requestGeneration: -1,
      requestOwners: SpatiallyIndexedSkeletonChunkRequestOwner.NONE,
      downloadAbortController: new AbortController(),
    } as any;
  }

  function makeSource(chunk: any) {
    return {
      chunks: new Map([["0,0,0:0", chunk]]),
    } as any;
  }

  function makeChunkManager() {
    return {
      queueManager: {
        updateChunkState: vi.fn(),
      },
    } as any;
  }

  it("tracks both owners within the same recompute generation", () => {
    const chunk = makeChunk();

    markSpatiallyIndexedSkeletonChunkRequested(
      chunk,
      5,
      SpatiallyIndexedSkeletonChunkRequestOwner.VIEW_2D,
    );
    markSpatiallyIndexedSkeletonChunkRequested(
      chunk,
      5,
      SpatiallyIndexedSkeletonChunkRequestOwner.VIEW_3D,
    );

    expect(chunk.requestGeneration).toBe(5);
    expect(chunk.requestOwners).toBe(
      SpatiallyIndexedSkeletonChunkRequestOwner.VIEW_2D |
        SpatiallyIndexedSkeletonChunkRequestOwner.VIEW_3D,
    );

    markSpatiallyIndexedSkeletonChunkRequested(
      chunk,
      6,
      SpatiallyIndexedSkeletonChunkRequestOwner.VIEW_3D,
    );

    expect(chunk.requestGeneration).toBe(6);
    expect(chunk.requestOwners).toBe(
      SpatiallyIndexedSkeletonChunkRequestOwner.VIEW_3D,
    );
  });

  it("aborts stale downloading chunks that were not requested this recompute", () => {
    const chunkManager = makeChunkManager();
    const chunk = makeChunk();
    const source = makeSource(chunk);

    markSpatiallyIndexedSkeletonChunkRequested(
      chunk,
      4,
      SpatiallyIndexedSkeletonChunkRequestOwner.VIEW_2D,
    );

    cancelStaleSpatiallyIndexedSkeletonDownloads(chunkManager, [source], 5);

    expect(chunk.downloadAbortController).toBeUndefined();
    expect(chunkManager.queueManager.updateChunkState).toHaveBeenCalledWith(
      chunk,
      ChunkState.QUEUED,
    );
  });

  it("keeps downloads requested by 3D in the current recompute", () => {
    const chunkManager = makeChunkManager();
    const chunk = makeChunk();
    const source = makeSource(chunk);

    markSpatiallyIndexedSkeletonChunkRequested(
      chunk,
      8,
      SpatiallyIndexedSkeletonChunkRequestOwner.VIEW_3D,
    );

    cancelStaleSpatiallyIndexedSkeletonDownloads(chunkManager, [source], 8);

    expect(chunk.downloadAbortController?.signal.aborted).toBe(false);
    expect(chunkManager.queueManager.updateChunkState).not.toHaveBeenCalled();
  });

  it("keeps downloads requested by 2D in the current recompute", () => {
    const chunkManager = makeChunkManager();
    const chunk = makeChunk();
    const source = makeSource(chunk);

    markSpatiallyIndexedSkeletonChunkRequested(
      chunk,
      9,
      SpatiallyIndexedSkeletonChunkRequestOwner.VIEW_2D,
    );

    cancelStaleSpatiallyIndexedSkeletonDownloads(chunkManager, [source], 9);

    expect(chunk.downloadAbortController?.signal.aborted).toBe(false);
    expect(chunkManager.queueManager.updateChunkState).not.toHaveBeenCalled();
  });

  it("keeps shared downloads when both owners still request the chunk", () => {
    const chunkManager = makeChunkManager();
    const chunk = makeChunk();
    const source = makeSource(chunk);

    markSpatiallyIndexedSkeletonChunkRequested(
      chunk,
      11,
      SpatiallyIndexedSkeletonChunkRequestOwner.VIEW_2D,
    );
    markSpatiallyIndexedSkeletonChunkRequested(
      chunk,
      11,
      SpatiallyIndexedSkeletonChunkRequestOwner.VIEW_3D,
    );

    cancelStaleSpatiallyIndexedSkeletonDownloads(chunkManager, [source], 11);

    expect(chunk.downloadAbortController?.signal.aborted).toBe(false);
    expect(chunkManager.queueManager.updateChunkState).not.toHaveBeenCalled();
  });

  it("does not touch queued chunks that never started downloading", () => {
    const chunkManager = makeChunkManager();
    const chunk = makeChunk(ChunkState.QUEUED);
    const source = makeSource(chunk);

    markSpatiallyIndexedSkeletonChunkRequested(
      chunk,
      2,
      SpatiallyIndexedSkeletonChunkRequestOwner.VIEW_2D,
    );

    cancelStaleSpatiallyIndexedSkeletonDownloads(chunkManager, [source], 3);

    expect(chunk.downloadAbortController?.signal.aborted).toBe(false);
    expect(chunkManager.queueManager.updateChunkState).not.toHaveBeenCalled();
  });
});
