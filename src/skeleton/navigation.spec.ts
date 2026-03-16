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

import { describe, expect, it } from "vitest";

import {
  buildSpatiallyIndexedSkeletonNavigationGraph,
  getCurrentBranchContext,
  getNextBranchOrEnd,
  getOpenLeaves,
  getPreviousBranchOrRoot,
  getSkeletonRootNode,
} from "#src/skeleton/navigation.js";
import type { SpatiallyIndexedSkeletonNodeInfo } from "#src/skeleton/frontend.js";

function makeNode(
  nodeId: number,
  parentNodeId: number | undefined,
  options: {
    labels?: readonly string[];
  } = {},
): SpatiallyIndexedSkeletonNodeInfo {
  return {
    nodeId,
    segmentId: 42,
    position: new Float32Array([nodeId, nodeId + 0.5, nodeId + 1]),
    parentNodeId,
    labels: options.labels,
  };
}

describe("skeleton/navigation", () => {
  const graph = buildSpatiallyIndexedSkeletonNavigationGraph([
    makeNode(1, undefined),
    makeNode(2, 1),
    makeNode(3, 2),
    makeNode(4, 3),
    makeNode(5, 4, { labels: ["checkpoint"] }),
    makeNode(6, 5),
    makeNode(7, 3),
    makeNode(8, 3),
    makeNode(9, 8),
    makeNode(10, 9, { labels: ["ends"] }),
    makeNode(11, 9),
  ]);

  it("finds the skeleton root and upstream branch boundaries", () => {
    expect(getSkeletonRootNode(graph).nodeId).toBe(1);
    expect(getPreviousBranchOrRoot(graph, 6).nodeId).toBe(3);
    expect(getPreviousBranchOrRoot(graph, 3).nodeId).toBe(1);
    expect(getPreviousBranchOrRoot(graph, 6, { alt: true }).nodeId).toBe(5);
  });

  it("returns downstream branches in deterministic local order", () => {
    expect(
      getNextBranchOrEnd(graph, 3).map((branch) => [
        branch.child.nodeId,
        branch.branchStartOrEnd.nodeId,
        branch.branchEnd.nodeId,
      ]),
    ).toEqual([
      [4, 5, 6],
      [8, 9, 9],
      [7, 7, 7],
    ]);
  });

  it("tracks the active sibling branch from the selected node or anchor", () => {
    expect(getCurrentBranchContext(graph, 6)).toMatchObject({
      branchNode: { nodeId: 3 },
      currentBranchIndex: 0,
    });
    expect(getCurrentBranchContext(graph, 3)).toMatchObject({
      branchNode: { nodeId: 3 },
      currentBranchIndex: 0,
    });
    expect(
      getCurrentBranchContext(graph, 3, { anchorNodeId: 11 }),
    ).toMatchObject({
      branchNode: { nodeId: 3 },
      currentBranchIndex: 1,
    });
    expect(
      getCurrentBranchContext(graph, 3, { anchorNodeId: 10 }),
    ).toMatchObject({
      branchNode: { nodeId: 3 },
      currentBranchIndex: 1,
    });
  });

  it("finds unfinished leaves from any selected node and filters closed ends", () => {
    expect(
      getOpenLeaves(graph, 3).map((leaf) => [leaf.nodeId, leaf.distance]),
    ).toEqual([
      [7, 1],
      [1, 2],
      [6, 3],
      [11, 3],
    ]);
    expect(getOpenLeaves(graph, 1).some((leaf) => leaf.nodeId === 1)).toBe(
      true,
    );
  });
});
