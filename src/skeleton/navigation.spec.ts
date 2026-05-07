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

import type { SpatiallyIndexedSkeletonNode } from "#src/skeleton/api.js";
import {
  buildSpatiallyIndexedSkeletonNavigationGraph,
  getBranchEnd,
  getBranchStart,
  getChildNode,
  getFlatListNodeIds,
  getNextCollapsedLevelNode,
  getOpenLeaves,
  getParentNode,
  getSkeletonRootNode,
} from "#src/skeleton/navigation.js";

function makeNode(
  nodeId: bigint,
  parentNodeId: bigint | undefined,
  options: {
    description?: string;
    isTrueEnd?: boolean;
  } = {},
): SpatiallyIndexedSkeletonNode {
  return {
    nodeId,
    segmentId: 42n,
    position: new Float32Array([
      Number(nodeId),
      Number(nodeId) + 0.5,
      Number(nodeId) + 1,
    ]),
    parentNodeId,
    description: options.description,
    isTrueEnd: options.isTrueEnd ?? false,
  };
}

describe("skeleton/navigation", () => {
  const graph = buildSpatiallyIndexedSkeletonNavigationGraph([
    makeNode(1n, undefined),
    makeNode(2n, 1n),
    makeNode(3n, 2n),
    makeNode(4n, 3n),
    makeNode(5n, 4n, { description: "checkpoint" }),
    makeNode(6n, 5n),
    makeNode(7n, 3n),
    makeNode(8n, 3n),
    makeNode(9n, 8n),
    makeNode(10n, 9n, { isTrueEnd: true }),
    makeNode(11n, 9n),
  ]);

  it("finds the skeleton root and branch starts", () => {
    expect(getSkeletonRootNode(graph).nodeId).toBe(1n);
    expect(getBranchStart(graph, 6n).nodeId).toBe(3n);
    expect(getBranchStart(graph, 3n).nodeId).toBe(3n);
    expect(getBranchStart(graph, 2n).nodeId).toBe(2n);
    expect(getBranchStart(graph, 1n).nodeId).toBe(1n);
  });

  it("prefers a downstream branch over a leaf for branch-end navigation", () => {
    const preferenceGraph = buildSpatiallyIndexedSkeletonNavigationGraph([
      makeNode(1n, undefined),
      makeNode(2n, 1n),
      makeNode(3n, 1n),
      makeNode(4n, 3n),
      makeNode(5n, 4n),
      makeNode(6n, 4n),
    ]);

    expect(getBranchEnd(preferenceGraph, 1n).nodeId).toBe(4n);
    expect(getBranchEnd(preferenceGraph, 3n).nodeId).toBe(4n);
    expect(getBranchEnd(preferenceGraph, 2n).nodeId).toBe(2n);
  });

  it("orders flat-list rows in leaf-first pre-order", () => {
    const listGraph = buildSpatiallyIndexedSkeletonNavigationGraph([
      makeNode(1n, undefined),
      makeNode(2n, 1n),
      makeNode(3n, 1n),
      makeNode(4n, 1n),
      makeNode(5n, 2n),
      makeNode(6n, 4n),
      makeNode(7n, 4n),
      makeNode(8n, 1n, { isTrueEnd: true }),
      makeNode(9n, 8n),
    ]);

    expect(getFlatListNodeIds(listGraph)).toEqual([
      1n,
      3n,
      8n,
      9n,
      4n,
      6n,
      7n,
      2n,
      5n,
    ]);
  });

  it("orders flat-list rows by collapsed branches in leaf-first pre-order", () => {
    const listGraph = buildSpatiallyIndexedSkeletonNavigationGraph([
      makeNode(1n, undefined),
      makeNode(2n, 1n),
      makeNode(3n, 2n),
      makeNode(4n, 3n),
      makeNode(5n, 3n),
      makeNode(6n, 2n),
      makeNode(7n, 6n),
    ]);

    expect(
      getFlatListNodeIds(listGraph, {
        collapseRegularNodesForOrdering: true,
      }),
    ).toEqual([1n, 2n, 6n, 7n, 3n, 4n, 5n]);
  });

  it("keeps a branch adjacent to its own leaf-first descendants", () => {
    const listGraph = buildSpatiallyIndexedSkeletonNavigationGraph([
      makeNode(1n, undefined),
      makeNode(2n, 1n),
      makeNode(3n, 1n),
      makeNode(4n, 2n),
      makeNode(5n, 2n),
      makeNode(6n, 3n),
      makeNode(7n, 3n),
    ]);

    expect(
      getFlatListNodeIds(listGraph, {
        collapseRegularNodesForOrdering: true,
      }),
    ).toEqual([1n, 2n, 4n, 5n, 3n, 6n, 7n]);
  });

  it("returns deterministic direct parent and child navigation targets", () => {
    expect(getParentNode(graph, 6n)?.nodeId).toBe(5n);
    expect(getParentNode(graph, 1n)).toBeUndefined();
    expect(getChildNode(graph, 3n)?.nodeId).toBe(7n);
    expect(getChildNode(graph, 11n)).toBeUndefined();
  });

  it("cycles through collapsed-level nodes and skips regular nodes", () => {
    const collapsedGraph = buildSpatiallyIndexedSkeletonNavigationGraph([
      makeNode(1n, undefined),
      makeNode(2n, 1n),
      makeNode(3n, 2n),
      makeNode(4n, 1n),
      makeNode(5n, 1n),
      makeNode(6n, 4n),
      makeNode(7n, 4n),
    ]);

    expect(getNextCollapsedLevelNode(collapsedGraph, 1n).nodeId).toBe(1n);
    expect(getNextCollapsedLevelNode(collapsedGraph, 2n).nodeId).toBe(2n);
    expect(getNextCollapsedLevelNode(collapsedGraph, 5n).nodeId).toBe(4n);
    expect(getNextCollapsedLevelNode(collapsedGraph, 4n).nodeId).toBe(3n);
    expect(getNextCollapsedLevelNode(collapsedGraph, 3n).nodeId).toBe(5n);
  });

  it("cycles collapsed-level nodes using collapsed leaf-first ordering", () => {
    const collapsedGraph = buildSpatiallyIndexedSkeletonNavigationGraph([
      makeNode(1n, undefined),
      makeNode(2n, 1n),
      makeNode(3n, 2n),
      makeNode(4n, 3n),
      makeNode(5n, 3n),
      makeNode(6n, 2n),
      makeNode(7n, 6n),
    ]);

    expect(getNextCollapsedLevelNode(collapsedGraph, 6n).nodeId).toBe(6n);
    expect(getNextCollapsedLevelNode(collapsedGraph, 7n).nodeId).toBe(3n);
    expect(getNextCollapsedLevelNode(collapsedGraph, 3n).nodeId).toBe(7n);
  });

  it("finds unfinished leaves from any selected node and filters closed ends", () => {
    expect(
      getOpenLeaves(graph, 3n).map((leaf) => [leaf.nodeId, leaf.distance]),
    ).toEqual([
      [7n, 1],
      [6n, 3],
      [11n, 3],
    ]);
    expect(
      getOpenLeaves(graph, 1n).map((leaf) => [leaf.nodeId, leaf.distance]),
    ).toEqual([
      [7n, 3],
      [6n, 5],
      [11n, 5],
    ]);
  });
});
