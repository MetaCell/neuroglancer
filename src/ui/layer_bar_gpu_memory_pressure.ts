/**
 * @license
 * Copyright 2026 Google Inc.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

import {
  ChunkMemoryStatistics,
  ChunkPriorityTier,
  ChunkState,
  getChunkStateStatisticIndex,
  numChunkMemoryStatistics,
} from "#src/chunk_manager/base.js";

export type GpuMemoryPressure = "normal" | "warning" | "critical";

export const GPU_MEMORY_PRESSURE_WARNING_RATIO = 0.9;
export const GPU_MEMORY_PRESSURE_CRITICAL_RATIO = 0.98;

export function computeGpuMemoryBytes(
  chunkStatistics: Iterable<Float64Array>,
) {
  let total = 0;
  for (const statistics of chunkStatistics) {
    for (
      let tier = ChunkPriorityTier.FIRST_TIER;
      tier <= ChunkPriorityTier.LAST_TIER;
      ++tier
    ) {
      total +=
        statistics[
          getChunkStateStatisticIndex(ChunkState.GPU_MEMORY, tier) *
            numChunkMemoryStatistics +
            ChunkMemoryStatistics.gpuMemoryBytes
        ] || 0;
    }
  }
  return total;
}

export function getGpuMemoryPressure(
  gpuMemoryBytes: number,
  gpuMemoryLimitBytes: number,
): GpuMemoryPressure {
  if (
    !Number.isFinite(gpuMemoryBytes) ||
    !Number.isFinite(gpuMemoryLimitBytes) ||
    gpuMemoryBytes < 0 ||
    gpuMemoryLimitBytes <= 0
  ) {
    return "normal";
  }
  const ratio = gpuMemoryBytes / gpuMemoryLimitBytes;
  if (ratio >= GPU_MEMORY_PRESSURE_CRITICAL_RATIO) {
    return "critical";
  }
  if (ratio >= GPU_MEMORY_PRESSURE_WARNING_RATIO) {
    return "warning";
  }
  return "normal";
}

export function getLayerGpuMemoryPressure(
  globalPressure: GpuMemoryPressure,
  numVisibleChunksNeeded: number,
  numVisibleChunksAvailable: number,
): GpuMemoryPressure {
  if (
    globalPressure === "normal" ||
    numVisibleChunksNeeded <= 0 ||
    numVisibleChunksAvailable >= numVisibleChunksNeeded
  ) {
    return "normal";
  }
  return globalPressure;
}
