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

import { CATMAID_TRUE_END_LABEL } from "#src/datasource/catmaid/api.js";
import type {
  SpatiallyIndexedSkeletonBranchNavigationTarget,
  SpatiallyIndexedSkeletonNavigationTarget,
  SpatiallyIndexedSkeletonOpenLeaf,
} from "#src/skeleton/api.js";
import type { SpatiallyIndexedSkeletonNodeInfo } from "#src/skeleton/frontend.js";

export interface SpatiallyIndexedSkeletonNavigationGraph {
  nodeById: Map<number, SpatiallyIndexedSkeletonNodeInfo>;
  childrenByParent: Map<number, number[]>;
  rootNodeIds: number[];
}

export interface SpatiallyIndexedSkeletonBranchContext {
  branchNode: SpatiallyIndexedSkeletonNavigationTarget;
  branches: SpatiallyIndexedSkeletonBranchNavigationTarget[];
  currentBranchIndex: number | undefined;
}

const CLOSED_END_LABEL_PATTERNS = [
  /^uncertain continuation$/i,
  /^not a branch$/i,
  /^soma$/i,
  /^(really|uncertain|anterior|posterior)?\s?ends?$/i,
];

export function buildSpatiallyIndexedSkeletonNavigationGraph(
  nodes: readonly SpatiallyIndexedSkeletonNodeInfo[],
): SpatiallyIndexedSkeletonNavigationGraph {
  const nodeById = new Map<number, SpatiallyIndexedSkeletonNodeInfo>();
  for (const node of nodes) {
    if (!nodeById.has(node.nodeId)) {
      nodeById.set(node.nodeId, node);
    }
  }

  const childrenByParent = new Map<number, number[]>();
  for (const node of nodeById.values()) {
    const parentNodeId = node.parentNodeId;
    if (parentNodeId === undefined || !nodeById.has(parentNodeId)) continue;
    let children = childrenByParent.get(parentNodeId);
    if (children === undefined) {
      children = [];
      childrenByParent.set(parentNodeId, children);
    }
    children.push(node.nodeId);
  }
  for (const children of childrenByParent.values()) {
    children.sort((a, b) => a - b);
  }

  const rootNodeIds: number[] = [];
  for (const node of nodeById.values()) {
    const parentNodeId = node.parentNodeId;
    if (parentNodeId === undefined || !nodeById.has(parentNodeId)) {
      rootNodeIds.push(node.nodeId);
    }
  }
  rootNodeIds.sort((a, b) => a - b);
  if (rootNodeIds.length === 0 && nodeById.size > 0) {
    rootNodeIds.push([...nodeById.keys()].sort((a, b) => a - b)[0]);
  }

  return {
    nodeById,
    childrenByParent,
    rootNodeIds,
  };
}

function hasTrueEndLabel(node: SpatiallyIndexedSkeletonNodeInfo) {
  return (
    node.labels?.some(
      (label) => label.trim().toLowerCase() === CATMAID_TRUE_END_LABEL,
    ) ?? false
  );
}

function getFlatListNodeSortPriority(
  graph: SpatiallyIndexedSkeletonNavigationGraph,
  nodeId: number,
) {
  const node = getNodeOrThrow(graph, nodeId);
  if (hasTrueEndLabel(node)) {
    return 0;
  }
  const childCount = getChildNodeIds(graph, nodeId).length;
  if (childCount === 0) {
    return 0;
  }
  const parentNodeId = getParentNodeId(graph, nodeId);
  if (parentNodeId === undefined) {
    return 3;
  }
  if (childCount > 1) {
    return 1;
  }
  return 2;
}

function compareFlatListNodeIds(
  graph: SpatiallyIndexedSkeletonNavigationGraph,
  a: number,
  b: number,
) {
  const priorityDelta =
    getFlatListNodeSortPriority(graph, a) -
    getFlatListNodeSortPriority(graph, b);
  return priorityDelta !== 0 ? priorityDelta : a - b;
}

export function getFlatListNodeIds(
  graph: SpatiallyIndexedSkeletonNavigationGraph,
) {
  const orderedNodeIds: number[] = [];
  const visited = new Set<number>();

  const appendBreadthFirst = (startNodeIds: readonly number[]) => {
    const queue = [...startNodeIds].sort((a, b) =>
      compareFlatListNodeIds(graph, a, b),
    );
    for (let queueIndex = 0; queueIndex < queue.length; ++queueIndex) {
      const nodeId = queue[queueIndex];
      if (visited.has(nodeId)) continue;
      visited.add(nodeId);
      orderedNodeIds.push(nodeId);
      const childNodeIds = [...getChildNodeIds(graph, nodeId)].sort((a, b) =>
        compareFlatListNodeIds(graph, a, b),
      );
      for (const childNodeId of childNodeIds) {
        if (!visited.has(childNodeId)) {
          queue.push(childNodeId);
        }
      }
    }
  };

  appendBreadthFirst(graph.rootNodeIds);

  const remainingNodeIds = [...graph.nodeById.keys()].sort((a, b) =>
    compareFlatListNodeIds(graph, a, b),
  );
  for (const nodeId of remainingNodeIds) {
    if (!visited.has(nodeId)) {
      appendBreadthFirst([nodeId]);
    }
  }

  return orderedNodeIds;
}

function getNodeOrThrow(
  graph: SpatiallyIndexedSkeletonNavigationGraph,
  nodeId: number,
) {
  const node = graph.nodeById.get(nodeId);
  if (node === undefined) {
    throw new Error(`Node ${nodeId} is not available in the loaded skeleton.`);
  }
  return node;
}

function getNodeTarget(
  graph: SpatiallyIndexedSkeletonNavigationGraph,
  nodeId: number,
): SpatiallyIndexedSkeletonNavigationTarget {
  const node = getNodeOrThrow(graph, nodeId);
  return {
    nodeId: node.nodeId,
    x: Number(node.position[0]),
    y: Number(node.position[1]),
    z: Number(node.position[2]),
  };
}

function getChildNodeIds(
  graph: SpatiallyIndexedSkeletonNavigationGraph,
  nodeId: number,
) {
  return graph.childrenByParent.get(nodeId) ?? [];
}

function getParentNodeId(
  graph: SpatiallyIndexedSkeletonNavigationGraph,
  nodeId: number,
) {
  const parentNodeId = getNodeOrThrow(graph, nodeId).parentNodeId;
  if (parentNodeId === undefined || !graph.nodeById.has(parentNodeId)) {
    return undefined;
  }
  return parentNodeId;
}

function getFirstInterestingNodeId(
  graph: SpatiallyIndexedSkeletonNavigationGraph,
  sequence: readonly number[],
) {
  if (sequence.length === 0) {
    throw new Error("No nodes are available for navigation.");
  }
  for (const nodeId of sequence) {
    const labels = graph.nodeById.get(nodeId)?.labels;
    if ((labels?.length ?? 0) > 0) {
      return nodeId;
    }
  }
  return sequence[sequence.length - 1];
}

function countDownstreamArborSize(
  graph: SpatiallyIndexedSkeletonNavigationGraph,
  nodeId: number,
) {
  let count = 0;
  const stack = [nodeId];
  const visited = new Set<number>();
  while (stack.length > 0) {
    const currentNodeId = stack.pop()!;
    if (visited.has(currentNodeId)) continue;
    visited.add(currentNodeId);
    const childNodeIds = getChildNodeIds(graph, currentNodeId);
    if (childNodeIds.length > 0) {
      count++;
      for (let i = childNodeIds.length - 1; i >= 0; --i) {
        stack.push(childNodeIds[i]);
      }
    }
  }
  return count;
}

function getOrderedChildNodeIds(
  graph: SpatiallyIndexedSkeletonNavigationGraph,
  nodeId: number,
) {
  const childNodeIds = [...getChildNodeIds(graph, nodeId)];
  if (childNodeIds.length <= 1) {
    return childNodeIds;
  }
  childNodeIds.sort((a, b) => {
    const sizeDelta =
      countDownstreamArborSize(graph, b) - countDownstreamArborSize(graph, a);
    return sizeDelta !== 0 ? sizeDelta : a - b;
  });
  return childNodeIds;
}

function getPreviousBranchOrRootNodeId(
  graph: SpatiallyIndexedSkeletonNavigationGraph,
  nodeId: number,
  options: { alt?: boolean } = {},
) {
  getNodeOrThrow(graph, nodeId);
  const sequence: number[] = [];
  let currentNodeId = nodeId;
  const visited = new Set<number>([currentNodeId]);
  while (true) {
    const parentNodeId = getParentNodeId(graph, currentNodeId);
    if (parentNodeId === undefined || visited.has(parentNodeId)) {
      break;
    }
    currentNodeId = parentNodeId;
    visited.add(currentNodeId);
    sequence.push(currentNodeId);
    if (getChildNodeIds(graph, currentNodeId).length !== 1) {
      break;
    }
  }
  if ((options.alt ?? false) && sequence.length > 0) {
    return getFirstInterestingNodeId(graph, sequence);
  }
  return currentNodeId;
}

function getDirectChildOnPath(
  graph: SpatiallyIndexedSkeletonNavigationGraph,
  ancestorNodeId: number,
  descendantNodeId: number,
) {
  if (ancestorNodeId === descendantNodeId) return undefined;
  let currentNodeId = descendantNodeId;
  const visited = new Set<number>();
  while (!visited.has(currentNodeId)) {
    visited.add(currentNodeId);
    const parentNodeId = getParentNodeId(graph, currentNodeId);
    if (parentNodeId === undefined) {
      return undefined;
    }
    if (parentNodeId === ancestorNodeId) {
      return currentNodeId;
    }
    currentNodeId = parentNodeId;
  }
  return undefined;
}

function getNextBranchTargets(
  graph: SpatiallyIndexedSkeletonNavigationGraph,
  nodeId: number,
) {
  getNodeOrThrow(graph, nodeId);
  const childNodeIds = getOrderedChildNodeIds(graph, nodeId);
  return childNodeIds.map((childNodeId) => {
    const sequence = [childNodeId];
    let branchEndNodeId = childNodeId;
    while (true) {
      const branchChildNodeIds = getChildNodeIds(graph, branchEndNodeId);
      if (branchChildNodeIds.length !== 1) {
        break;
      }
      branchEndNodeId = branchChildNodeIds[0];
      sequence.push(branchEndNodeId);
    }
    return {
      child: getNodeTarget(graph, childNodeId),
      branchStartOrEnd: getNodeTarget(
        graph,
        getFirstInterestingNodeId(graph, sequence),
      ),
      branchEnd: getNodeTarget(graph, branchEndNodeId),
    };
  });
}

function hasClosedEndLabel(node: SpatiallyIndexedSkeletonNodeInfo) {
  return (
    node.labels?.some((label) =>
      CLOSED_END_LABEL_PATTERNS.some((pattern) => pattern.test(label.trim())),
    ) ?? false
  );
}

export function getSkeletonRootNode(
  graph: SpatiallyIndexedSkeletonNavigationGraph,
) {
  const rootNodeId = graph.rootNodeIds[0];
  if (rootNodeId === undefined) {
    throw new Error("The loaded skeleton does not contain a root node.");
  }
  return getNodeTarget(graph, rootNodeId);
}

export function getPreviousBranchOrRoot(
  graph: SpatiallyIndexedSkeletonNavigationGraph,
  nodeId: number,
  options: { alt?: boolean } = {},
) {
  return getNodeTarget(
    graph,
    getPreviousBranchOrRootNodeId(graph, nodeId, options),
  );
}

export function getNextBranchOrEnd(
  graph: SpatiallyIndexedSkeletonNavigationGraph,
  nodeId: number,
): SpatiallyIndexedSkeletonBranchNavigationTarget[] {
  return getNextBranchTargets(graph, nodeId);
}

export function getOpenLeaves(
  graph: SpatiallyIndexedSkeletonNavigationGraph,
  nodeId: number,
): SpatiallyIndexedSkeletonOpenLeaf[] {
  getNodeOrThrow(graph, nodeId);
  const distances = new Map<number, number>([[nodeId, 0]]);
  const rootedChildCount = new Map<number, number>([[nodeId, 0]]);
  const queue = [nodeId];
  for (let queueIndex = 0; queueIndex < queue.length; ++queueIndex) {
    const currentNodeId = queue[queueIndex];
    const nextDistance = (distances.get(currentNodeId) ?? 0) + 1;
    const neighborNodeIds = [
      ...getChildNodeIds(graph, currentNodeId),
      getParentNodeId(graph, currentNodeId),
    ].filter((value): value is number => value !== undefined);
    neighborNodeIds.sort((a, b) => a - b);
    for (const neighborNodeId of neighborNodeIds) {
      if (distances.has(neighborNodeId)) continue;
      distances.set(neighborNodeId, nextDistance);
      rootedChildCount.set(
        currentNodeId,
        (rootedChildCount.get(currentNodeId) ?? 0) + 1,
      );
      rootedChildCount.set(
        neighborNodeId,
        rootedChildCount.get(neighborNodeId) ?? 0,
      );
      queue.push(neighborNodeId);
    }
  }

  const leaves: SpatiallyIndexedSkeletonOpenLeaf[] = [];
  for (const candidateNodeId of queue) {
    const childCount = rootedChildCount.get(candidateNodeId) ?? 0;
    const isLeaf =
      childCount === 0 || (candidateNodeId === nodeId && childCount === 1);
    if (!isLeaf) continue;
    const candidateNode = getNodeOrThrow(graph, candidateNodeId);
    if (hasClosedEndLabel(candidateNode)) continue;
    leaves.push({
      ...getNodeTarget(graph, candidateNodeId),
      distance: distances.get(candidateNodeId) ?? 0,
    });
  }

  leaves.sort((a, b) =>
    a.distance === b.distance ? a.nodeId - b.nodeId : a.distance - b.distance,
  );
  return leaves;
}

export function getCurrentBranchContext(
  graph: SpatiallyIndexedSkeletonNavigationGraph,
  nodeId: number,
  options: { anchorNodeId?: number } = {},
): SpatiallyIndexedSkeletonBranchContext {
  const branchNodeId =
    getChildNodeIds(graph, nodeId).length > 1
      ? nodeId
      : getPreviousBranchOrRootNodeId(graph, nodeId);
  const branches = getNextBranchTargets(graph, branchNodeId);
  let currentChildNodeId = getDirectChildOnPath(graph, branchNodeId, nodeId);

  const anchorNodeId = options.anchorNodeId;
  if (
    currentChildNodeId === undefined &&
    anchorNodeId !== undefined &&
    graph.nodeById.has(anchorNodeId)
  ) {
    currentChildNodeId = getDirectChildOnPath(
      graph,
      branchNodeId,
      anchorNodeId,
    );
  }

  let currentBranchIndex =
    currentChildNodeId === undefined
      ? undefined
      : branches.findIndex(
          (branch) => branch.child.nodeId === currentChildNodeId,
        );
  if (currentBranchIndex !== undefined && currentBranchIndex < 0) {
    currentBranchIndex = undefined;
  }
  if (currentBranchIndex === undefined && branches.length > 0) {
    currentBranchIndex = 0;
  }

  return {
    branchNode: getNodeTarget(graph, branchNodeId),
    branches,
    currentBranchIndex,
  };
}
