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

const DEFAULT_SPATIALLY_INDEXED_SKELETON_MAX_CHUNKS = 64;
const DEFAULT_SPATIALLY_INDEXED_SKELETON_MIN_CHUNK_SIZE = 1;

export interface SpatiallyIndexedSkeletonBounds {
  min: { x: number; y: number; z: number };
  max: { x: number; y: number; z: number };
}

export interface SpatiallyIndexedSkeletonChunkSize {
  x: number;
  y: number;
  z: number;
}

export interface DefaultSpatiallyIndexedSkeletonChunkSizeOptions {
  maxChunks?: number;
  minChunkSize?: number;
}

function validateFiniteOptions(
  options: DefaultSpatiallyIndexedSkeletonChunkSizeOptions,
) {
  if (
    options.minChunkSize !== undefined &&
    !Number.isFinite(options.minChunkSize)
  ) {
    throw new Error(
      "Spatially indexed skeleton minChunkSize must be finite.",
    );
  }
  if (options.maxChunks !== undefined && !Number.isFinite(options.maxChunks)) {
    throw new Error("Spatially indexed skeleton maxChunks must be finite.");
  }
}

function validateFiniteBounds(bounds: SpatiallyIndexedSkeletonBounds) {
  const values = [
    ["min.x", bounds.min.x],
    ["min.y", bounds.min.y],
    ["min.z", bounds.min.z],
    ["max.x", bounds.max.x],
    ["max.y", bounds.max.y],
    ["max.z", bounds.max.z],
  ] as const;
  for (const [name, value] of values) {
    if (!Number.isFinite(value)) {
      throw new Error(
        `Spatially indexed skeleton bounds must be finite, but ${name} is ${value}.`,
      );
    }
  }
}

function getChunkCoverageForChunkSize(
  extents: readonly number[],
  chunkSize: number,
) {
  return extents.reduce((product, extent) => {
    const axisChunks = extent <= 0 ? 1 : Math.ceil(extent / chunkSize);
    return product * axisChunks;
  }, 1);
}

export function getDefaultSpatiallyIndexedSkeletonChunkSize(
  bounds: SpatiallyIndexedSkeletonBounds,
  options: DefaultSpatiallyIndexedSkeletonChunkSizeOptions = {},
): SpatiallyIndexedSkeletonChunkSize {
  validateFiniteOptions(options);
  validateFiniteBounds(bounds);
  const minChunkSize = Math.max(
    DEFAULT_SPATIALLY_INDEXED_SKELETON_MIN_CHUNK_SIZE,
    Math.ceil(
      options.minChunkSize ??
        DEFAULT_SPATIALLY_INDEXED_SKELETON_MIN_CHUNK_SIZE,
    ),
  );
  const maxChunks = Math.max(
    1,
    Math.floor(options.maxChunks ?? DEFAULT_SPATIALLY_INDEXED_SKELETON_MAX_CHUNKS),
  );
  const extents = [
    Math.max(0, bounds.max.x - bounds.min.x),
    Math.max(0, bounds.max.y - bounds.min.y),
    Math.max(0, bounds.max.z - bounds.min.z),
  ] as const;
  const maxExtent = Math.max(...extents);

  if (!(maxExtent > 0)) {
    return { x: minChunkSize, y: minChunkSize, z: minChunkSize };
  }

  // Choose the smallest isotropic chunk size that keeps the full bounding box
  // coverage within the requested chunk budget.
  let low = minChunkSize;
  let high = Math.max(minChunkSize, Math.ceil(maxExtent));
  while (low < high) {
    const mid = Math.floor((low + high) / 2);
    if (getChunkCoverageForChunkSize(extents, mid) <= maxChunks) {
      high = mid;
    } else {
      low = mid + 1;
    }
  }

  return { x: low, y: low, z: low };
}
