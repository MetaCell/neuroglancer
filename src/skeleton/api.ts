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

export type SpatialSkeletonVector = ArrayLike<number>;

export interface SpatialSkeletonBounds {
  lowerBounds: SpatialSkeletonVector;
  upperBounds: SpatialSkeletonVector;
}

export interface SpatialSkeletonSpatialIndexLevel {
  chunkSize: SpatialSkeletonVector;
  gridShape: readonly number[];
  limit?: number;
}

export interface SpatiallyIndexedSkeletonNodeBase {
  nodeId: number;
  segmentId: number;
  position: SpatialSkeletonVector;
  parentNodeId?: number;
  sourceState?: unknown;
}

export interface SpatiallyIndexedSkeletonNode
  extends SpatiallyIndexedSkeletonNodeBase {
  radius?: number;
  confidence?: number;
  description?: string;
  isTrueEnd?: boolean;
}

export interface SpatiallyIndexedSkeletonMetadata
  extends SpatialSkeletonBounds {
  spatial: readonly SpatialSkeletonSpatialIndexLevel[];
}

export interface SpatiallyIndexedSkeletonSource {
  readonly spatialSkeletonEditController?: unknown;
  listSkeletons(): Promise<number[]>;
  getSkeleton(
    skeletonId: number,
    options?: { signal?: AbortSignal },
  ): Promise<SpatiallyIndexedSkeletonNode[]>;
  getSpatialIndexMetadata(): Promise<SpatiallyIndexedSkeletonMetadata | null>;
  fetchNodes(
    bounds: SpatialSkeletonBounds,
    lod?: number,
    options?: {
      signal?: AbortSignal;
    },
  ): Promise<SpatiallyIndexedSkeletonNodeBase[]>;
}
