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

/**
 * @file Summary statistics for an in-memory spatially indexed skeleton.
 *
 * Node positions are model-space coordinates; for CATMAID sources those are
 * nanometers, so computed lengths are nanometers and are formatted as
 * micrometers.
 */

import type { SpatiallyIndexedSkeletonNode } from "#src/skeleton/api.js";

export interface SkeletonBoundingBox {
  readonly lower: readonly number[];
  readonly upper: readonly number[];
  readonly size: readonly number[];
  readonly diagonalLength: number;
}

export interface SkeletonLongestPath {
  readonly startNodeId: number;
  readonly endNodeId: number;
  readonly nodeCount: number;
  readonly length: number;
  readonly fractionOfCableLength: number;
}

export interface SkeletonStatistics {
  readonly segmentId: number | undefined;
  readonly nodeCount: number;
  readonly rootNodeIds: readonly number[];
  readonly branchPointCount: number;
  readonly endpointCount: number;
  readonly cableLength: number;
  readonly boundingBox: SkeletonBoundingBox | undefined;
  /** Longest root-to-leaf path; the usual heuristic guess for the axon. */
  readonly longestPath: SkeletonLongestPath | undefined;
}

function distanceBetween(a: ArrayLike<number>, b: ArrayLike<number>): number {
  let sum = 0;
  const rank = Math.min(a.length, b.length);
  for (let i = 0; i < rank; ++i) {
    const delta = a[i] - b[i];
    sum += delta * delta;
  }
  return Math.sqrt(sum);
}

export function computeSkeletonStatistics(
  nodes: Iterable<SpatiallyIndexedSkeletonNode>,
): SkeletonStatistics {
  const nodeList = Array.from(nodes);
  const nodesById = new Map<number, SpatiallyIndexedSkeletonNode>();
  for (const node of nodeList) {
    nodesById.set(node.nodeId, node);
  }

  const childCounts = new Map<number, number>();
  const rootNodeIds: number[] = [];
  let cableLength = 0;
  let lower: number[] | undefined;
  let upper: number[] | undefined;

  for (const node of nodeList) {
    const { position } = node;
    if (lower === undefined || upper === undefined) {
      lower = Array.from(position);
      upper = Array.from(position);
    } else {
      for (let i = 0; i < position.length; ++i) {
        if (position[i] < lower[i]) lower[i] = position[i];
        if (position[i] > upper[i]) upper[i] = position[i];
      }
    }
    const parent =
      node.parentNodeId === undefined
        ? undefined
        : nodesById.get(node.parentNodeId);
    if (parent === undefined) {
      rootNodeIds.push(node.nodeId);
      continue;
    }
    childCounts.set(parent.nodeId, (childCounts.get(parent.nodeId) ?? 0) + 1);
    cableLength += distanceBetween(node.position, parent.position);
  }

  let branchPointCount = 0;
  let endpointCount = 0;
  for (const node of nodeList) {
    const childCount = childCounts.get(node.nodeId) ?? 0;
    if (childCount === 0) ++endpointCount;
    if (childCount >= 2) ++branchPointCount;
  }

  const boundingBox =
    lower === undefined || upper === undefined
      ? undefined
      : {
          lower,
          upper,
          size: lower.map((value, index) => upper![index] - value),
          diagonalLength: distanceBetween(lower, upper),
        };

  return {
    segmentId: nodeList[0]?.segmentId,
    nodeCount: nodeList.length,
    rootNodeIds,
    branchPointCount,
    endpointCount,
    cableLength,
    boundingBox,
    longestPath: computeLongestRootToLeafPath(
      nodeList,
      nodesById,
      childCounts,
      cableLength,
    ),
  };
}

/**
 * Distance from each node to its root, memoized by walking the ancestor chain
 * iteratively.  Nodes on a parent cycle are skipped rather than recursed into.
 */
function computeDistancesToRoot(
  nodeList: readonly SpatiallyIndexedSkeletonNode[],
  nodesById: ReadonlyMap<number, SpatiallyIndexedSkeletonNode>,
) {
  const distanceToRoot = new Map<number, number>();
  const rootOfNode = new Map<number, number>();
  const ancestorChain: SpatiallyIndexedSkeletonNode[] = [];
  for (const startNode of nodeList) {
    if (distanceToRoot.has(startNode.nodeId)) continue;
    ancestorChain.length = 0;
    const visited = new Set<number>();
    let current: SpatiallyIndexedSkeletonNode | undefined = startNode;
    while (
      current !== undefined &&
      !distanceToRoot.has(current.nodeId) &&
      !visited.has(current.nodeId)
    ) {
      visited.add(current.nodeId);
      ancestorChain.push(current);
      current =
        current.parentNodeId === undefined
          ? undefined
          : nodesById.get(current.parentNodeId);
    }
    if (current !== undefined && visited.has(current.nodeId)) {
      // Parent cycle: leave the whole chain unmeasured.
      continue;
    }
    for (let i = ancestorChain.length - 1; i >= 0; --i) {
      const node = ancestorChain[i];
      const parent =
        node.parentNodeId === undefined
          ? undefined
          : nodesById.get(node.parentNodeId);
      if (parent === undefined) {
        distanceToRoot.set(node.nodeId, 0);
        rootOfNode.set(node.nodeId, node.nodeId);
        continue;
      }
      const parentDistance = distanceToRoot.get(parent.nodeId);
      if (parentDistance === undefined) break;
      distanceToRoot.set(
        node.nodeId,
        parentDistance + distanceBetween(node.position, parent.position),
      );
      rootOfNode.set(node.nodeId, rootOfNode.get(parent.nodeId)!);
    }
  }
  return { distanceToRoot, rootOfNode };
}

function computeLongestRootToLeafPath(
  nodeList: readonly SpatiallyIndexedSkeletonNode[],
  nodesById: ReadonlyMap<number, SpatiallyIndexedSkeletonNode>,
  childCounts: ReadonlyMap<number, number>,
  cableLength: number,
): SkeletonLongestPath | undefined {
  const { distanceToRoot, rootOfNode } = computeDistancesToRoot(
    nodeList,
    nodesById,
  );
  let farthestLeaf: SpatiallyIndexedSkeletonNode | undefined;
  let farthestDistance = -1;
  for (const node of nodeList) {
    if ((childCounts.get(node.nodeId) ?? 0) !== 0) continue;
    const distance = distanceToRoot.get(node.nodeId);
    if (distance === undefined || distance <= farthestDistance) continue;
    farthestDistance = distance;
    farthestLeaf = node;
  }
  if (farthestLeaf === undefined) return undefined;

  let nodeCount = 1;
  let current = farthestLeaf;
  const visited = new Set<number>([current.nodeId]);
  for (;;) {
    const parent =
      current.parentNodeId === undefined
        ? undefined
        : nodesById.get(current.parentNodeId);
    if (parent === undefined || visited.has(parent.nodeId)) break;
    visited.add(parent.nodeId);
    ++nodeCount;
    current = parent;
  }

  return {
    startNodeId: rootOfNode.get(farthestLeaf.nodeId) ?? current.nodeId,
    endNodeId: farthestLeaf.nodeId,
    nodeCount,
    length: farthestDistance,
    fractionOfCableLength: cableLength > 0 ? farthestDistance / cableLength : 0,
  };
}

function formatLength(nanometers: number): string {
  return `${(nanometers / 1000).toFixed(2)} um`;
}

export function formatSkeletonStatistics(
  statistics: SkeletonStatistics,
): string {
  const lines: string[] = [];
  lines.push(`Skeleton ${statistics.segmentId ?? "(unknown)"}`);
  lines.push(`  Nodes:          ${statistics.nodeCount}`);
  lines.push(`  Cable length:   ${formatLength(statistics.cableLength)}`);
  lines.push(`  Branch points:  ${statistics.branchPointCount}`);
  lines.push(`  Endpoints:      ${statistics.endpointCount}`);
  const { boundingBox } = statistics;
  if (boundingBox !== undefined) {
    const extent = boundingBox.size
      .map((value) => (value / 1000).toFixed(2))
      .join(" x ");
    lines.push(`  Spatial extent: ${extent} um`);
    lines.push(
      `                  diagonal ${formatLength(boundingBox.diagonalLength)}`,
    );
  }
  const { longestPath } = statistics;
  if (longestPath !== undefined) {
    lines.push(
      `  Longest branch (axon guess): node ${longestPath.startNodeId} -> ${longestPath.endNodeId}`,
    );
    lines.push(
      `                  ${formatLength(longestPath.length)} over ${longestPath.nodeCount} nodes ` +
        `(${(longestPath.fractionOfCableLength * 100).toFixed(1)}% of cable)`,
    );
  }
  return lines.join("\n");
}
