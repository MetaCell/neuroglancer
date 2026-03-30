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

interface CollapsedChildPath {
  path: readonly number[];
  representativeNodeId: number;
}

interface CollapsedLevelContext {
  levelByNodeId: Map<number, number>;
  nodeIdsByLevel: Map<number, number[]>;
}

interface NavigationGraphDerivedState {
  sortPriorityByNodeId: Map<number, number>;
  orderedChildNodeIdsByNodeId: Map<number, readonly number[]>;
  collapsedPathByNodeId: Map<number, readonly number[]>;
  collapsedOrderedChildPathsByNodeId: Map<number, readonly CollapsedChildPath[]>;
  flatListNodeIds?: readonly number[];
  collapsedFlatListNodeIds?: readonly number[];
  collapsedLevelContext?: CollapsedLevelContext;
}

const navigationGraphDerivedState = new WeakMap<
  SpatiallyIndexedSkeletonNavigationGraph,
  NavigationGraphDerivedState
>();

const CLOSED_END_LABEL_PATTERNS = [
  /^uncertain continuation$/i,
  /^not a branch$/i,
  /^soma$/i,
  /^(really|uncertain|anterior|posterior)?\s?ends?$/i,
];

function buildNavigationGraphDerivedState(
  graph: SpatiallyIndexedSkeletonNavigationGraph,
): NavigationGraphDerivedState {
  const sortPriorityByNodeId = new Map<number, number>();
  for (const [nodeId, node] of graph.nodeById) {
    const childCount = graph.childrenByParent.get(nodeId)?.length ?? 0;
    const parentNodeId = node.parentNodeId;
    const parentInTree =
      parentNodeId !== undefined && graph.nodeById.has(parentNodeId);
    const sortPriority = hasTrueEndLabel(node)
      ? 0
      : childCount === 0
        ? 0
        : !parentInTree
          ? 3
          : childCount > 1
            ? 1
            : 2;
    sortPriorityByNodeId.set(nodeId, sortPriority);
  }
  return {
    sortPriorityByNodeId,
    orderedChildNodeIdsByNodeId: new Map(),
    collapsedPathByNodeId: new Map(),
    collapsedOrderedChildPathsByNodeId: new Map(),
  };
}

function getNavigationGraphDerivedState(
  graph: SpatiallyIndexedSkeletonNavigationGraph,
) {
  let state = navigationGraphDerivedState.get(graph);
  if (state === undefined) {
    state = buildNavigationGraphDerivedState(graph);
    navigationGraphDerivedState.set(graph, state);
  }
  return state;
}

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

  const graph = {
    nodeById,
    childrenByParent,
    rootNodeIds,
  };
  navigationGraphDerivedState.set(graph, buildNavigationGraphDerivedState(graph));
  return graph;
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
  const priority = getNavigationGraphDerivedState(graph).sortPriorityByNodeId.get(
    nodeId,
  );
  if (priority === undefined) {
    throw new Error(`Node ${nodeId} is not available in the loaded skeleton.`);
  }
  return priority;
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
  options: { collapseRegularNodesForOrdering?: boolean } = {},
) {
  if (options.collapseRegularNodesForOrdering ?? false) {
    return getCollapsedOrderedFlatListNodeIds(graph);
  }
  const derivedState = getNavigationGraphDerivedState(graph);
  if (derivedState.flatListNodeIds !== undefined) {
    return derivedState.flatListNodeIds;
  }

  const orderedNodeIds: number[] = [];
  const visited = new Set<number>();

  const appendLeafFirstPreOrder = (startNodeIds: readonly number[]) => {
    const stack = [...startNodeIds]
      .sort((a, b) => compareFlatListNodeIds(graph, a, b))
      .reverse();
    while (stack.length > 0) {
      const nodeId = stack.pop()!;
      if (visited.has(nodeId)) continue;
      visited.add(nodeId);
      orderedNodeIds.push(nodeId);
      const childNodeIds = [...getChildNodeIds(graph, nodeId)].sort((a, b) =>
        compareFlatListNodeIds(graph, a, b),
      );
      for (let childIndex = childNodeIds.length - 1; childIndex >= 0; --childIndex) {
        const childNodeId = childNodeIds[childIndex];
        if (!visited.has(childNodeId)) {
          stack.push(childNodeId);
        }
      }
    }
  };

  appendLeafFirstPreOrder(graph.rootNodeIds);
  if (visited.size === graph.nodeById.size) {
    derivedState.flatListNodeIds = orderedNodeIds;
    return orderedNodeIds;
  }

  const remainingNodeIds = [...graph.nodeById.keys()].sort((a, b) =>
    compareFlatListNodeIds(graph, a, b),
  );
  for (const nodeId of remainingNodeIds) {
    if (!visited.has(nodeId)) {
      appendLeafFirstPreOrder([nodeId]);
    }
  }

  derivedState.flatListNodeIds = orderedNodeIds;
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

function getFlatListOrderedChildNodeIds(
  graph: SpatiallyIndexedSkeletonNavigationGraph,
  nodeId: number,
) {
  const childNodeIds = getChildNodeIds(graph, nodeId);
  if (childNodeIds.length <= 1) {
    return childNodeIds;
  }
  const derivedState = getNavigationGraphDerivedState(graph);
  const cached = derivedState.orderedChildNodeIdsByNodeId.get(nodeId);
  if (cached !== undefined) {
    return cached;
  }
  const orderedChildNodeIds = [...childNodeIds].sort((a, b) =>
    compareFlatListNodeIds(graph, a, b),
  );
  derivedState.orderedChildNodeIdsByNodeId.set(nodeId, orderedChildNodeIds);
  return orderedChildNodeIds;
}

function getCollapsedBranchPath(
  graph: SpatiallyIndexedSkeletonNavigationGraph,
  nodeId: number,
) {
  const derivedState = getNavigationGraphDerivedState(graph);
  const cached = derivedState.collapsedPathByNodeId.get(nodeId);
  if (cached !== undefined) {
    return cached;
  }
  const path = [nodeId];
  const visited = new Set<number>(path);
  let currentNodeId = nodeId;
  while (isCollapsedRegularNode(graph, currentNodeId)) {
    const nextNodeId = getChildNodeIds(graph, currentNodeId)[0];
    if (nextNodeId === undefined || visited.has(nextNodeId)) {
      break;
    }
    path.push(nextNodeId);
    visited.add(nextNodeId);
    currentNodeId = nextNodeId;
  }
  derivedState.collapsedPathByNodeId.set(nodeId, path);
  return path;
}

function getCollapsedOrderedChildPaths(
  graph: SpatiallyIndexedSkeletonNavigationGraph,
  nodeId: number,
) {
  const derivedState = getNavigationGraphDerivedState(graph);
  const cached = derivedState.collapsedOrderedChildPathsByNodeId.get(nodeId);
  if (cached !== undefined) {
    return cached;
  }
  const childPaths = getChildNodeIds(graph, nodeId).map((childNodeId) => {
    const path = getCollapsedBranchPath(graph, childNodeId);
    return {
      path,
      representativeNodeId: path[path.length - 1],
    };
  });
  childPaths.sort((a, b) =>
    compareFlatListNodeIds(
      graph,
      a.representativeNodeId,
      b.representativeNodeId,
    ),
  );
  derivedState.collapsedOrderedChildPathsByNodeId.set(nodeId, childPaths);
  return childPaths;
}

function isCollapsedRegularNode(
  graph: SpatiallyIndexedSkeletonNavigationGraph,
  nodeId: number,
) {
  const node = getNodeOrThrow(graph, nodeId);
  return (
    getParentNodeId(graph, nodeId) !== undefined &&
    getChildNodeIds(graph, nodeId).length === 1 &&
    !hasTrueEndLabel(node)
  );
}

function getCollapsedChildNodeIds(
  graph: SpatiallyIndexedSkeletonNavigationGraph,
  nodeId: number,
) {
  return getCollapsedOrderedChildPaths(graph, nodeId).map(
    ({ representativeNodeId }) => representativeNodeId,
  );
}

function getCollapsedOrderedFlatListNodeIds(
  graph: SpatiallyIndexedSkeletonNavigationGraph,
) {
  const derivedState = getNavigationGraphDerivedState(graph);
  if (derivedState.collapsedFlatListNodeIds !== undefined) {
    return derivedState.collapsedFlatListNodeIds;
  }
  const orderedNodeIds: number[] = [];
  const visited = new Set<number>();

  const appendLeafFirstPreOrder = (startPaths: readonly number[][]) => {
    const stack = [...startPaths].reverse();
    while (stack.length > 0) {
      const path = stack.pop()!;
      const representativeNodeId = path[path.length - 1];
      let appendedNode = false;
      for (const nodeId of path) {
        if (visited.has(nodeId)) continue;
        visited.add(nodeId);
        orderedNodeIds.push(nodeId);
        appendedNode = true;
      }
      if (!appendedNode || !graph.nodeById.has(representativeNodeId)) {
        continue;
      }
      const childPaths = getCollapsedOrderedChildPaths(
        graph,
        representativeNodeId,
      );
      for (
        let childPathIndex = childPaths.length - 1;
        childPathIndex >= 0;
        --childPathIndex
      ) {
        const childPath = childPaths[childPathIndex];
        const firstNodeId = childPath.path[0];
        if (firstNodeId !== undefined && !visited.has(firstNodeId)) {
          stack.push(childPath.path);
        }
      }
    }
  };

  appendLeafFirstPreOrder(graph.rootNodeIds.map((nodeId) => [nodeId]));
  if (visited.size === graph.nodeById.size) {
    derivedState.collapsedFlatListNodeIds = orderedNodeIds;
    return orderedNodeIds;
  }

  const remainingNodeIds = [...graph.nodeById.keys()].sort((a, b) =>
    compareFlatListNodeIds(graph, a, b),
  );
  for (const nodeId of remainingNodeIds) {
    if (!visited.has(nodeId)) {
      appendLeafFirstPreOrder([getCollapsedBranchPath(graph, nodeId)]);
    }
  }

  derivedState.collapsedFlatListNodeIds = orderedNodeIds;
  return orderedNodeIds;
}

function getCollapsedLevelContext(
  graph: SpatiallyIndexedSkeletonNavigationGraph,
) {
  const derivedState = getNavigationGraphDerivedState(graph);
  if (derivedState.collapsedLevelContext !== undefined) {
    return derivedState.collapsedLevelContext;
  }
  const levelByNodeId = new Map<number, number>();
  const nodeIdsByLevel = new Map<number, number[]>();
  const queue = graph.rootNodeIds.map((nodeId) => ({ nodeId, level: 0 }));
  const visited = new Set<number>();

  for (let queueIndex = 0; queueIndex < queue.length; ++queueIndex) {
    const { nodeId, level } = queue[queueIndex];
    if (visited.has(nodeId)) continue;
    visited.add(nodeId);
    levelByNodeId.set(nodeId, level);
    let nodeIds = nodeIdsByLevel.get(level);
    if (nodeIds === undefined) {
      nodeIds = [];
      nodeIdsByLevel.set(level, nodeIds);
    }
    nodeIds.push(nodeId);
    for (const childNodeId of getCollapsedChildNodeIds(graph, nodeId)) {
      if (!visited.has(childNodeId)) {
        queue.push({ nodeId: childNodeId, level: level + 1 });
      }
    }
  }

  const context = { levelByNodeId, nodeIdsByLevel };
  derivedState.collapsedLevelContext = context;
  return context;
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
  const childNodeIds = getFlatListOrderedChildNodeIds(graph, nodeId);
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

export function getBranchStart(
  graph: SpatiallyIndexedSkeletonNavigationGraph,
  nodeId: number,
) {
  getNodeOrThrow(graph, nodeId);
  let currentNodeId = nodeId;
  const visited = new Set<number>([currentNodeId]);
  while (true) {
    const parentNodeId = getParentNodeId(graph, currentNodeId);
    if (parentNodeId === undefined || visited.has(parentNodeId)) {
      return getNodeTarget(graph, nodeId);
    }
    currentNodeId = parentNodeId;
    visited.add(currentNodeId);
    if (getChildNodeIds(graph, currentNodeId).length > 1) {
      return getNodeTarget(graph, currentNodeId);
    }
  }
}

export function getNextBranchOrEnd(
  graph: SpatiallyIndexedSkeletonNavigationGraph,
  nodeId: number,
): SpatiallyIndexedSkeletonBranchNavigationTarget[] {
  return getNextBranchTargets(graph, nodeId);
}

export function getBranchEnd(
  graph: SpatiallyIndexedSkeletonNavigationGraph,
  nodeId: number,
) {
  getNodeOrThrow(graph, nodeId);
  const branchTargets = getNextBranchTargets(graph, nodeId);
  if (branchTargets.length === 0) {
    return getNodeTarget(graph, nodeId);
  }
  const preferredTarget =
    branchTargets.find(
      (target) => getChildNodeIds(graph, target.branchEnd.nodeId).length > 1,
    ) ?? branchTargets[0];
  return preferredTarget.branchEnd;
}

export function getParentNode(
  graph: SpatiallyIndexedSkeletonNavigationGraph,
  nodeId: number,
) {
  const parentNodeId = getParentNodeId(graph, nodeId);
  return parentNodeId === undefined
    ? undefined
    : getNodeTarget(graph, parentNodeId);
}

export function getChildNode(
  graph: SpatiallyIndexedSkeletonNavigationGraph,
  nodeId: number,
) {
  const childNodeId = getFlatListOrderedChildNodeIds(graph, nodeId)[0];
  return childNodeId === undefined
    ? undefined
    : getNodeTarget(graph, childNodeId);
}

export function getRandomChildNode(
  graph: SpatiallyIndexedSkeletonNavigationGraph,
  nodeId: number,
  options: { random?: () => number } = {},
) {
  const childNodeIds = getFlatListOrderedChildNodeIds(graph, nodeId);
  if (childNodeIds.length === 0) {
    return undefined;
  }
  const { random = Math.random } = options;
  const randomValue = random();
  const childIndex = Number.isFinite(randomValue)
    ? Math.min(
        childNodeIds.length - 1,
        Math.max(0, Math.floor(randomValue * childNodeIds.length)),
      )
    : 0;
  return getNodeTarget(graph, childNodeIds[childIndex]);
}

export function getNextCollapsedLevelNode(
  graph: SpatiallyIndexedSkeletonNavigationGraph,
  nodeId: number,
) {
  getNodeOrThrow(graph, nodeId);
  if (getParentNodeId(graph, nodeId) === undefined) {
    return getNodeTarget(graph, nodeId);
  }
  if (isCollapsedRegularNode(graph, nodeId)) {
    return getNodeTarget(graph, nodeId);
  }

  const { levelByNodeId, nodeIdsByLevel } = getCollapsedLevelContext(graph);
  const level = levelByNodeId.get(nodeId);
  if (level === undefined) {
    return getNodeTarget(graph, nodeId);
  }
  const nodeIds = nodeIdsByLevel.get(level);
  if (nodeIds === undefined || nodeIds.length <= 1) {
    return getNodeTarget(graph, nodeId);
  }
  const currentIndex = nodeIds.indexOf(nodeId);
  if (currentIndex === -1) {
    return getNodeTarget(graph, nodeId);
  }
  return getNodeTarget(graph, nodeIds[(currentIndex + 1) % nodeIds.length]);
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
    const childNodeIds = getChildNodeIds(graph, currentNodeId);
    const parentNodeId = getParentNodeId(graph, currentNodeId);
    let parentAdded = false;
    for (const childNodeId of childNodeIds) {
      if (
        !parentAdded &&
        parentNodeId !== undefined &&
        parentNodeId < childNodeId
      ) {
        if (!distances.has(parentNodeId)) {
          distances.set(parentNodeId, nextDistance);
          rootedChildCount.set(
            currentNodeId,
            (rootedChildCount.get(currentNodeId) ?? 0) + 1,
          );
          rootedChildCount.set(
            parentNodeId,
            rootedChildCount.get(parentNodeId) ?? 0,
          );
          queue.push(parentNodeId);
        }
        parentAdded = true;
      }
      const neighborNodeId = childNodeId;
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
    if (!parentAdded && parentNodeId !== undefined && !distances.has(parentNodeId)) {
      distances.set(parentNodeId, nextDistance);
      rootedChildCount.set(
        currentNodeId,
        (rootedChildCount.get(currentNodeId) ?? 0) + 1,
      );
      rootedChildCount.set(
        parentNodeId,
        rootedChildCount.get(parentNodeId) ?? 0,
      );
      queue.push(parentNodeId);
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
