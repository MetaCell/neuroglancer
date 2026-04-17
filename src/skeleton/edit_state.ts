import type {
  SpatiallyIndexedSkeletonEditContext,
  SpatiallyIndexedSkeletonEditNodeContext,
  SpatiallyIndexedSkeletonEditParentContext,
  SpatiallyIndexedSkeletonRevisionToken,
} from "#src/skeleton/api.js";
import type { SpatiallyIndexedSkeletonNodeInfo } from "#src/skeleton/frontend.js";

function requireRevisionToken(
  node: SpatiallyIndexedSkeletonNodeInfo,
  role: string,
): SpatiallyIndexedSkeletonRevisionToken {
  if (node.revisionToken === undefined) {
    throw new Error(
      `Inspected CATMAID ${role} node ${node.nodeId} is missing revision metadata.`,
    );
  }
  return node.revisionToken;
}

export function findSpatiallyIndexedSkeletonNodeInfo(
  segmentNodes: readonly SpatiallyIndexedSkeletonNodeInfo[],
  nodeId: number,
) {
  return segmentNodes.find((node) => node.nodeId === nodeId);
}

export function getSpatiallyIndexedSkeletonDirectChildren(
  segmentNodes: readonly SpatiallyIndexedSkeletonNodeInfo[],
  nodeId: number,
) {
  return segmentNodes
    .filter((node) => node.parentNodeId === nodeId)
    .sort((a, b) => a.nodeId - b.nodeId);
}

export function getSpatiallyIndexedSkeletonNodeParent(
  segmentNodes: readonly SpatiallyIndexedSkeletonNodeInfo[],
  node: Pick<SpatiallyIndexedSkeletonNodeInfo, "parentNodeId">,
) {
  if (node.parentNodeId === undefined) {
    return undefined;
  }
  return findSpatiallyIndexedSkeletonNodeInfo(segmentNodes, node.parentNodeId);
}

export function toSpatiallyIndexedSkeletonEditNodeContext(
  node: SpatiallyIndexedSkeletonNodeInfo,
): SpatiallyIndexedSkeletonEditNodeContext {
  return {
    nodeId: node.nodeId,
    parentNodeId: node.parentNodeId,
    revisionToken: requireRevisionToken(node, "target"),
  };
}

export function toSpatiallyIndexedSkeletonEditParentContext(
  node: SpatiallyIndexedSkeletonNodeInfo,
): SpatiallyIndexedSkeletonEditParentContext {
  return {
    nodeId: node.nodeId,
    revisionToken: requireRevisionToken(node, "related"),
  };
}

export function buildSpatiallyIndexedSkeletonNodeEditContext(
  node: SpatiallyIndexedSkeletonNodeInfo,
): SpatiallyIndexedSkeletonEditContext {
  return {
    node: toSpatiallyIndexedSkeletonEditNodeContext(node),
  };
}

export function buildSpatiallyIndexedSkeletonNeighborhoodEditContext(
  node: SpatiallyIndexedSkeletonNodeInfo,
  segmentNodes: readonly SpatiallyIndexedSkeletonNodeInfo[],
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

export function buildSpatiallyIndexedSkeletonMultiNodeEditContext(
  ...nodes: SpatiallyIndexedSkeletonNodeInfo[]
): SpatiallyIndexedSkeletonEditContext {
  return {
    nodes: nodes.map(toSpatiallyIndexedSkeletonEditParentContext),
  };
}
