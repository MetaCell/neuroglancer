import { describe, expect, it } from "vitest";

import {
  ChunkMemoryStatistics,
  ChunkPriorityTier,
  ChunkState,
  getChunkStateStatisticIndex,
  numChunkMemoryStatistics,
  numChunkStatistics,
} from "#src/chunk_manager/base.js";
import {
  computeGpuMemoryBytes,
  getGpuMemoryPressure,
  getLayerGpuMemoryPressure,
} from "#src/ui/layer_bar_gpu_memory_pressure.js";

function makeStatistics(
  gpuMemoryBytesByTier: Partial<Record<ChunkPriorityTier, number>>,
) {
  const statistics = new Float64Array(numChunkStatistics);
  for (const [tier, bytes] of Object.entries(gpuMemoryBytesByTier)) {
    statistics[
      getChunkStateStatisticIndex(ChunkState.GPU_MEMORY, Number(tier)) *
        numChunkMemoryStatistics +
        ChunkMemoryStatistics.gpuMemoryBytes
    ] = bytes;
  }
  return statistics;
}

describe("ui/layer_bar_gpu_memory_pressure", () => {
  it("sums GPU memory across all GPU_MEMORY priority tiers and sources", () => {
    const firstSource = makeStatistics({
      [ChunkPriorityTier.VISIBLE]: 10,
      [ChunkPriorityTier.PREFETCH]: 20,
      [ChunkPriorityTier.RECENT]: 30,
    });
    const secondSource = makeStatistics({
      [ChunkPriorityTier.VISIBLE]: 40,
      [ChunkPriorityTier.RECENT]: 50,
    });

    expect(computeGpuMemoryBytes([firstSource, secondSource])).toBe(150);
  });

  it("maps GPU usage ratio to warning and critical pressure", () => {
    expect(getGpuMemoryPressure(89, 100)).toBe("normal");
    expect(getGpuMemoryPressure(90, 100)).toBe("warning");
    expect(getGpuMemoryPressure(97, 100)).toBe("warning");
    expect(getGpuMemoryPressure(98, 100)).toBe("critical");
  });

  it("treats invalid or unlimited GPU memory limits as normal pressure", () => {
    expect(getGpuMemoryPressure(100, Number.POSITIVE_INFINITY)).toBe("normal");
    expect(getGpuMemoryPressure(100, Number.NaN)).toBe("normal");
    expect(getGpuMemoryPressure(100, 0)).toBe("normal");
    expect(getGpuMemoryPressure(Number.NaN, 100)).toBe("normal");
  });

  it("only applies global pressure to layers missing visible chunks", () => {
    expect(getLayerGpuMemoryPressure("warning", 10, 9)).toBe("warning");
    expect(getLayerGpuMemoryPressure("critical", 10, 9)).toBe("critical");
    expect(getLayerGpuMemoryPressure("warning", 10, 10)).toBe("normal");
    expect(getLayerGpuMemoryPressure("warning", 0, 0)).toBe("normal");
    expect(getLayerGpuMemoryPressure("normal", 10, 9)).toBe("normal");
  });
});
