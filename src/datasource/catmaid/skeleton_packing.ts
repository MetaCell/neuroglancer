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

import type { SpatiallyIndexedSkeletonNodeBase } from "#src/skeleton/api.js";

interface PackedCatmaidSkeletonData {
  vertexPositions: Float32Array;
  segmentIds: Uint32Array;
  indices: Uint32Array;
  nodeIds: Int32Array;
  nodeParentIds: Int32Array;
  nodeRadii: Float32Array;
  nodeConfidences: Float32Array;
  revisionTokens: Array<string | undefined>;
}

export function packCatmaidSkeletonNodes(
  nodes: readonly SpatiallyIndexedSkeletonNodeBase[],
): PackedCatmaidSkeletonData {
  const numVertices = nodes.length;
  const vertexPositions = new Float32Array(numVertices * 3);
  const segmentIds = new Uint32Array(numVertices);
  const nodeIds = new Int32Array(numVertices);
  const nodeParentIds = new Int32Array(numVertices);
  const nodeRadii = new Float32Array(numVertices);
  const nodeConfidences = new Float32Array(numVertices);
  const revisionTokens = new Array<string | undefined>(numVertices);
  const indices: number[] = [];
  const nodeMap = new Map<number, number>();

  for (let i = 0; i < numVertices; ++i) {
    const node = nodes[i];
    nodeMap.set(node.nodeId, i);
    nodeIds[i] = node.nodeId;
    nodeParentIds[i] = node.parentNodeId ?? 0;
    nodeRadii[i] =
      node.radius !== undefined && Number.isFinite(node.radius)
        ? node.radius
        : NaN;
    nodeConfidences[i] =
      node.confidence !== undefined && Number.isFinite(node.confidence)
        ? node.confidence
        : NaN;
    vertexPositions[i * 3] = node.position[0];
    vertexPositions[i * 3 + 1] = node.position[1];
    vertexPositions[i * 3 + 2] = node.position[2];
    segmentIds[i] = node.segmentId;
    revisionTokens[i] = node.revisionToken;
  }

  for (let i = 0; i < numVertices; ++i) {
    const node = nodes[i];
    if (node.parentNodeId === undefined) continue;
    const parentIndex = nodeMap.get(node.parentNodeId);
    if (parentIndex !== undefined) {
      indices.push(i, parentIndex);
    }
  }

  return {
    vertexPositions,
    segmentIds,
    indices: new Uint32Array(indices),
    nodeIds,
    nodeParentIds,
    nodeRadii,
    nodeConfidences,
    revisionTokens,
  };
}
