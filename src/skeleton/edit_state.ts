import type {
  SpatiallyIndexedSkeletonEditContext,
  SpatiallyIndexedSkeletonEditNodeContext,
  SpatiallyIndexedSkeletonEditParentContext,
  SpatiallyIndexedSkeletonNode,
} from "#src/skeleton/api.js";

function requireRevisionToken(
  node: SpatiallyIndexedSkeletonNode,
  role: string,
): string {
  if (node.revisionToken === undefined) {
    throw new Error(
      `Inspected spatial skeleton ${role} node ${node.nodeId} is missing revision metadata.`,
    );
  }
  return node.revisionToken;
}

export function findSpatiallyIndexedSkeletonNode(
  segmentNodes: readonly SpatiallyIndexedSkeletonNode[],
  nodeId: number,
) {
  return segmentNodes.find((node) => node.nodeId === nodeId);
}

export function getSpatiallyIndexedSkeletonDirectChildren(
  segmentNodes: readonly SpatiallyIndexedSkeletonNode[],
  nodeId: number,
) {
  return segmentNodes
    .filter((node) => node.parentNodeId === nodeId)
    .sort((a, b) => a.nodeId - b.nodeId);
}

export function getSpatiallyIndexedSkeletonNodeParent(
  segmentNodes: readonly SpatiallyIndexedSkeletonNode[],
  node: SpatiallyIndexedSkeletonNode,
) {
  if (node.parentNodeId === undefined) {
    return undefined;
  }
  return findSpatiallyIndexedSkeletonNode(segmentNodes, node.parentNodeId);
}

export function toSpatiallyIndexedSkeletonEditNodeContext(
  node: SpatiallyIndexedSkeletonNode,
): SpatiallyIndexedSkeletonEditNodeContext {
  return {
    nodeId: node.nodeId,
    parentNodeId: node.parentNodeId,
    revisionToken: requireRevisionToken(node, "target"),
  };
}

export function toSpatiallyIndexedSkeletonEditParentContext(
  node: SpatiallyIndexedSkeletonNode,
): SpatiallyIndexedSkeletonEditParentContext {
  return {
    nodeId: node.nodeId,
    revisionToken: requireRevisionToken(node, "related"),
  };
}

export function buildSpatiallyIndexedSkeletonNodeEditContext(
  node: SpatiallyIndexedSkeletonNode,
): SpatiallyIndexedSkeletonEditContext {
  return {
    node: toSpatiallyIndexedSkeletonEditNodeContext(node),
  };
}

export function buildSpatiallyIndexedSkeletonNeighborhoodEditContext(
  node: SpatiallyIndexedSkeletonNode,
  segmentNodes: readonly SpatiallyIndexedSkeletonNode[],
): SpatiallyIndexedSkeletonEditContext {
  // This intentionally derives parent/child state from the cached inspected
  // segment on demand. If large inspected segments make edit preparation
  // measurably slow, consider maintaining an adjacency index in
  // SpatialSkeletonState instead of rescanning here.
  const parentNode = getSpatiallyIndexedSkeletonNodeParent(segmentNodes, node);
  const childNodes = getSpatiallyIndexedSkeletonDirectChildren(
    segmentNodes,
    node.nodeId,
  );
  return {
    node: toSpatiallyIndexedSkeletonEditNodeContext(node),
    ...(parentNode === undefined
      ? {}
      : { parent: toSpatiallyIndexedSkeletonEditParentContext(parentNode) }),
    children: childNodes.map(toSpatiallyIndexedSkeletonEditParentContext),
  };
}

export function getSpatiallyIndexedSkeletonPathToRoot(
  segmentNodes: readonly SpatiallyIndexedSkeletonNode[],
  node: SpatiallyIndexedSkeletonNode,
) {
  const path = [node];
  const visited = new Set<number>([node.nodeId]);
  let currentNode = node;
  while (true) {
    const parentNode = getSpatiallyIndexedSkeletonNodeParent(
      segmentNodes,
      currentNode,
    );
    if (parentNode === undefined || visited.has(parentNode.nodeId)) {
      return path;
    }
    path.push(parentNode);
    visited.add(parentNode.nodeId);
    currentNode = parentNode;
  }
}

export function buildSpatiallyIndexedSkeletonRerootEditContext(
  node: SpatiallyIndexedSkeletonNode,
  segmentNodes: readonly SpatiallyIndexedSkeletonNode[],
): SpatiallyIndexedSkeletonEditContext {
  return {
    ...buildSpatiallyIndexedSkeletonNeighborhoodEditContext(node, segmentNodes),
    nodes: getSpatiallyIndexedSkeletonPathToRoot(segmentNodes, node).map(
      toSpatiallyIndexedSkeletonEditParentContext,
    ),
  };
}

export function buildSpatiallyIndexedSkeletonMultiNodeEditContext(
  ...nodes: SpatiallyIndexedSkeletonNode[]
): SpatiallyIndexedSkeletonEditContext {
  return {
    nodes: nodes.map(toSpatiallyIndexedSkeletonEditParentContext),
  };
}
