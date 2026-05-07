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

import type {
  SpatialSkeletonId,
  SpatiallyIndexedSkeletonNode,
  SpatialSkeletonVector,
} from "#src/skeleton/api.js";
import { compareUint64Ids } from "#src/util/bigint.js";

export interface SpatiallyIndexedSkeletonNavigationTarget {
  nodeId: SpatialSkeletonId;
  position: SpatialSkeletonVector;
}

export interface SpatiallyIndexedSkeletonOpenLeaf
  extends SpatiallyIndexedSkeletonNavigationTarget {
  distance: number;
  creationTime?: string;
}

export interface SpatiallyIndexedSkeletonNavigationGraph {
  nodeById: Map<SpatialSkeletonId, SpatiallyIndexedSkeletonNode>;
  childrenByParent: Map<SpatialSkeletonId, SpatialSkeletonId[]>;
  rootNodeIds: SpatialSkeletonId[];
}

interface CollapsedChildPath {
  path: readonly SpatialSkeletonId[];
  representativeNodeId: SpatialSkeletonId;
}

interface CollapsedLevelContext {
  levelByNodeId: Map<SpatialSkeletonId, number>;
  nodeIdsByLevel: Map<number, SpatialSkeletonId[]>;
}

interface NavigationGraphDerivedState {
  sortPriorityByNodeId: Map<SpatialSkeletonId, number>;
  orderedChildNodeIdsByNodeId: Map<SpatialSkeletonId, readonly SpatialSkeletonId[]>;
  collapsedPathByNodeId: Map<SpatialSkeletonId, readonly SpatialSkeletonId[]>;
  collapsedOrderedChildPathsByNodeId: Map<
    SpatialSkeletonId,
    readonly CollapsedChildPath[]
  >;
  flatListNodeIds?: readonly SpatialSkeletonId[];
  collapsedFlatListNodeIds?: readonly SpatialSkeletonId[];
  collapsedLevelContext?: CollapsedLevelContext;
}

const navigationGraphDerivedState = new WeakMap<
  SpatiallyIndexedSkeletonNavigationGraph,
  NavigationGraphDerivedState
>();

function buildNavigationGraphDerivedState(
  graph: SpatiallyIndexedSkeletonNavigationGraph,
): NavigationGraphDerivedState {
  const sortPriorityByNodeId = new Map<SpatialSkeletonId, number>();
  for (const [nodeId, node] of graph.nodeById) {
    const childCount = graph.childrenByParent.get(nodeId)?.length ?? 0;
    const parentNodeId = node.parentNodeId;
    const parentInTree =
      parentNodeId !== undefined && graph.nodeById.has(parentNodeId);
    const sortPriority =
      (node.isTrueEnd ?? false)
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
  nodes: readonly SpatiallyIndexedSkeletonNode[],
): SpatiallyIndexedSkeletonNavigationGraph {
  const nodeById = new Map<SpatialSkeletonId, SpatiallyIndexedSkeletonNode>();
  for (const node of nodes) {
    if (!nodeById.has(node.nodeId)) {
      nodeById.set(node.nodeId, node);
    }
  }

  const childrenByParent = new Map<SpatialSkeletonId, SpatialSkeletonId[]>();
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
    children.sort(compareUint64Ids);
  }

  const rootNodeIds: SpatialSkeletonId[] = [];
  for (const node of nodeById.values()) {
    const parentNodeId = node.parentNodeId;
    if (parentNodeId === undefined || !nodeById.has(parentNodeId)) {
      rootNodeIds.push(node.nodeId);
    }
  }
  rootNodeIds.sort(compareUint64Ids);
  if (rootNodeIds.length === 0 && nodeById.size > 0) {
    rootNodeIds.push([...nodeById.keys()].sort(compareUint64Ids)[0]);
  }

  const graph = {
    nodeById,
    childrenByParent,
    rootNodeIds,
  };
  navigationGraphDerivedState.set(
    graph,
    buildNavigationGraphDerivedState(graph),
  );
  return graph;
}

function getFlatListNodeSortPriority(
  graph: SpatiallyIndexedSkeletonNavigationGraph,
  nodeId: SpatialSkeletonId,
) {
  const priority =
    getNavigationGraphDerivedState(graph).sortPriorityByNodeId.get(nodeId);
  if (priority === undefined) {
    throw new Error(`Node ${nodeId} is not available in the loaded skeleton.`);
  }
  return priority;
}

function compareFlatListNodeIds(
  graph: SpatiallyIndexedSkeletonNavigationGraph,
  a: SpatialSkeletonId,
  b: SpatialSkeletonId,
) {
  const priorityDelta =
    getFlatListNodeSortPriority(graph, a) -
    getFlatListNodeSortPriority(graph, b);
  return priorityDelta !== 0 ? priorityDelta : compareUint64Ids(a, b);
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

  const orderedNodeIds: SpatialSkeletonId[] = [];
  const visited = new Set<SpatialSkeletonId>();

  const appendLeafFirstPreOrder = (
    startNodeIds: readonly SpatialSkeletonId[],
  ) => {
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
      for (
        let childIndex = childNodeIds.length - 1;
        childIndex >= 0;
        --childIndex
      ) {
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
  nodeId: SpatialSkeletonId,
) {
  const node = graph.nodeById.get(nodeId);
  if (node === undefined) {
    throw new Error(`Node ${nodeId} is not available in the loaded skeleton.`);
  }
  return node;
}

function getNodeTarget(
  graph: SpatiallyIndexedSkeletonNavigationGraph,
  nodeId: SpatialSkeletonId,
): SpatiallyIndexedSkeletonNavigationTarget {
  const node = getNodeOrThrow(graph, nodeId);
  return {
    nodeId: node.nodeId,
    position: node.position,
  };
}

function getChildNodeIds(
  graph: SpatiallyIndexedSkeletonNavigationGraph,
  nodeId: SpatialSkeletonId,
) {
  return graph.childrenByParent.get(nodeId) ?? [];
}

function getParentNodeId(
  graph: SpatiallyIndexedSkeletonNavigationGraph,
  nodeId: SpatialSkeletonId,
) {
  const parentNodeId = getNodeOrThrow(graph, nodeId).parentNodeId;
  if (parentNodeId === undefined || !graph.nodeById.has(parentNodeId)) {
    return undefined;
  }
  return parentNodeId;
}

function getFlatListOrderedChildNodeIds(
  graph: SpatiallyIndexedSkeletonNavigationGraph,
  nodeId: SpatialSkeletonId,
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
  nodeId: SpatialSkeletonId,
) {
  const derivedState = getNavigationGraphDerivedState(graph);
  const cached = derivedState.collapsedPathByNodeId.get(nodeId);
  if (cached !== undefined) {
    return cached;
  }
  const path = [nodeId];
  const visited = new Set<SpatialSkeletonId>(path);
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
  nodeId: SpatialSkeletonId,
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
  nodeId: SpatialSkeletonId,
) {
  const node = getNodeOrThrow(graph, nodeId);
  return (
    getParentNodeId(graph, nodeId) !== undefined &&
    getChildNodeIds(graph, nodeId).length === 1 &&
    !node.isTrueEnd
  );
}

function getCollapsedChildNodeIds(
  graph: SpatiallyIndexedSkeletonNavigationGraph,
  nodeId: SpatialSkeletonId,
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
  const orderedNodeIds: SpatialSkeletonId[] = [];
  const visited = new Set<SpatialSkeletonId>();

  const appendLeafFirstPreOrder = (
    startPaths: readonly (readonly SpatialSkeletonId[])[],
  ) => {
    const stack = [...startPaths].map((path) => [...path]).reverse();
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
          stack.push([...childPath.path]);
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
      appendLeafFirstPreOrder([[...getCollapsedBranchPath(graph, nodeId)]]);
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
  const levelByNodeId = new Map<SpatialSkeletonId, number>();
  const nodeIdsByLevel = new Map<number, SpatialSkeletonId[]>();
  const queue = graph.rootNodeIds.map((nodeId) => ({ nodeId, level: 0 }));
  const visited = new Set<SpatialSkeletonId>();

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

function getBranchEndNodeIds(
  graph: SpatiallyIndexedSkeletonNavigationGraph,
  nodeId: SpatialSkeletonId,
) {
  getNodeOrThrow(graph, nodeId);
  const childNodeIds = getFlatListOrderedChildNodeIds(graph, nodeId);
  return childNodeIds.map((childNodeId) => {
    let branchEndNodeId = childNodeId;
    while (true) {
      const branchChildNodeIds = getChildNodeIds(graph, branchEndNodeId);
      if (branchChildNodeIds.length !== 1) {
        break;
      }
      branchEndNodeId = branchChildNodeIds[0];
    }
    return branchEndNodeId;
  });
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

export function getBranchStart(
  graph: SpatiallyIndexedSkeletonNavigationGraph,
  nodeId: SpatialSkeletonId,
) {
  getNodeOrThrow(graph, nodeId);
  let currentNodeId = nodeId;
  const visited = new Set<SpatialSkeletonId>([currentNodeId]);
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

export function getBranchEnd(
  graph: SpatiallyIndexedSkeletonNavigationGraph,
  nodeId: SpatialSkeletonId,
) {
  getNodeOrThrow(graph, nodeId);
  const branchEndNodeIds = getBranchEndNodeIds(graph, nodeId);
  if (branchEndNodeIds.length === 0) {
    return getNodeTarget(graph, nodeId);
  }
  const preferredBranchEndNodeId =
    branchEndNodeIds.find(
      (branchEndNodeId) => getChildNodeIds(graph, branchEndNodeId).length > 1,
    ) ?? branchEndNodeIds[0];
  return getNodeTarget(graph, preferredBranchEndNodeId);
}

export function getParentNode(
  graph: SpatiallyIndexedSkeletonNavigationGraph,
  nodeId: SpatialSkeletonId,
) {
  const parentNodeId = getParentNodeId(graph, nodeId);
  return parentNodeId === undefined
    ? undefined
    : getNodeTarget(graph, parentNodeId);
}

export function getChildNode(
  graph: SpatiallyIndexedSkeletonNavigationGraph,
  nodeId: SpatialSkeletonId,
) {
  const childNodeId = getFlatListOrderedChildNodeIds(graph, nodeId)[0];
  return childNodeId === undefined
    ? undefined
    : getNodeTarget(graph, childNodeId);
}

export function getRandomChildNode(
  graph: SpatiallyIndexedSkeletonNavigationGraph,
  nodeId: SpatialSkeletonId,
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
  nodeId: SpatialSkeletonId,
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
  nodeId: SpatialSkeletonId,
): SpatiallyIndexedSkeletonOpenLeaf[] {
  getNodeOrThrow(graph, nodeId);
  const distances = new Map<SpatialSkeletonId, number>([[nodeId, 0]]);
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
          queue.push(parentNodeId);
        }
        parentAdded = true;
      }
      const neighborNodeId = childNodeId;
      if (distances.has(neighborNodeId)) continue;
      distances.set(neighborNodeId, nextDistance);
      queue.push(neighborNodeId);
    }
    if (
      !parentAdded &&
      parentNodeId !== undefined &&
      !distances.has(parentNodeId)
    ) {
      distances.set(parentNodeId, nextDistance);
      queue.push(parentNodeId);
    }
  }

  const leaves: SpatiallyIndexedSkeletonOpenLeaf[] = [];
  for (const candidateNodeId of queue) {
    if (getChildNodeIds(graph, candidateNodeId).length !== 0) continue;
    const candidateNode = getNodeOrThrow(graph, candidateNodeId);
    if (candidateNode.isTrueEnd) continue;
    leaves.push({
      ...getNodeTarget(graph, candidateNodeId),
      distance: distances.get(candidateNodeId) ?? 0,
    });
  }

  leaves.sort((a, b) =>
    a.distance === b.distance
      ? compareUint64Ids(a.nodeId, b.nodeId)
      : a.distance - b.distance,
  );
  return leaves;
}
