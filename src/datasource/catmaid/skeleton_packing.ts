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

import type { SpatiallyIndexedSkeletonNode } from "#src/skeleton/api.js";

export interface CatmaidMissingConnection {
  nodeId: number;
  parentId: number;
  vertexIndex: number;
  skeletonId: number;
}

export interface PackedCatmaidSkeletonData {
  vertexPositions: Float32Array;
  segmentIds: Uint32Array;
  indices: Uint32Array;
  nodeMap: Map<number, number>;
  missingConnections: CatmaidMissingConnection[];
}

export function packCatmaidSkeletonNodes(
  nodes: readonly SpatiallyIndexedSkeletonNode[],
  options: { recordMissingConnections?: boolean } = {},
): PackedCatmaidSkeletonData {
  const { recordMissingConnections = false } = options;
  const numVertices = nodes.length;
  const vertexPositions = new Float32Array(numVertices * 3);
  const segmentIds = new Uint32Array(numVertices);
  const indices: number[] = [];
  const nodeMap = new Map<number, number>();
  const missingConnections: CatmaidMissingConnection[] = [];

  for (let i = 0; i < numVertices; ++i) {
    const node = nodes[i];
    nodeMap.set(node.id, i);
    vertexPositions[i * 3] = node.x;
    vertexPositions[i * 3 + 1] = node.y;
    vertexPositions[i * 3 + 2] = node.z;
    segmentIds[i] = node.skeleton_id;
  }

  for (let i = 0; i < numVertices; ++i) {
    const node = nodes[i];
    if (node.parent_id === null) continue;
    const parentIndex = nodeMap.get(node.parent_id);
    if (parentIndex !== undefined) {
      indices.push(i, parentIndex);
      continue;
    }
    if (!recordMissingConnections) continue;
    missingConnections.push({
      nodeId: node.id,
      parentId: node.parent_id,
      vertexIndex: i,
      skeletonId: node.skeleton_id,
    });
  }

  return {
    vertexPositions,
    segmentIds,
    indices: new Uint32Array(indices),
    nodeMap,
    missingConnections,
  };
}
