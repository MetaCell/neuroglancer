import { describe, expect, it } from "vitest";

import type { SpatiallyIndexedSkeletonNodeInfo } from "#src/skeleton/frontend.js";
import { buildSpatiallyIndexedSkeletonNavigationGraph } from "#src/skeleton/navigation.js";
import { SpatialSkeletonNodeFilterType } from "#src/skeleton/node_types.js";
import { buildSpatialSkeletonSegmentRenderState } from "#src/ui/spatial_skeleton_edit_tab_render_state.js";

function makeNode(
  nodeId: number,
  parentNodeId: number | undefined,
  labels?: readonly string[],
): SpatiallyIndexedSkeletonNodeInfo {
  return {
    nodeId,
    segmentId: 20380,
    parentNodeId,
    position: new Float32Array([nodeId, nodeId + 1, nodeId + 2]),
    labels,
  };
}

describe("spatial skeleton edit tab render state", () => {
  it("shows only directly matching nodes for text filtering", () => {
    const graph = buildSpatiallyIndexedSkeletonNavigationGraph([
      makeNode(1, undefined),
      makeNode(2, 1),
      makeNode(3, 2),
      makeNode(4, 2),
    ]);

    const state = buildSpatialSkeletonSegmentRenderState(20380, graph, {
      filterText: "target",
      nodeFilterType: SpatialSkeletonNodeFilterType.NONE,
      collapseRegularNodes: false,
      getNodeDescription(node) {
        return node.nodeId === 4 ? "target" : undefined;
      },
    });

    expect(state.matchedNodeCount).toBe(1);
    expect(state.displayedNodeCount).toBe(1);
    expect(state.branchCount).toBe(1);
    expect(state.rows.map((row) => row.node.nodeId)).toEqual([4]);
  });

  it("does not match segment ids or labels in the search filter", () => {
    const graph = buildSpatiallyIndexedSkeletonNavigationGraph([
      makeNode(101, undefined, ["target-label"]),
      makeNode(102, 101),
    ]);

    const bySegmentId = buildSpatialSkeletonSegmentRenderState(20380, graph, {
      filterText: "20380",
      nodeFilterType: SpatialSkeletonNodeFilterType.NONE,
      collapseRegularNodes: false,
      getNodeDescription() {
        return undefined;
      },
    });
    const byLabel = buildSpatialSkeletonSegmentRenderState(20380, graph, {
      filterText: "target-label",
      nodeFilterType: SpatialSkeletonNodeFilterType.NONE,
      collapseRegularNodes: false,
      getNodeDescription() {
        return undefined;
      },
    });

    expect(bySegmentId.matchedNodeCount).toBe(0);
    expect(bySegmentId.displayedNodeCount).toBe(0);
    expect(byLabel.matchedNodeCount).toBe(0);
    expect(byLabel.displayedNodeCount).toBe(0);
  });

  it("counts hidden regular nodes in the ratio while omitting them from collapsed rows", () => {
    const graph = buildSpatiallyIndexedSkeletonNavigationGraph([
      makeNode(10, undefined),
      makeNode(11, 10),
      makeNode(12, 11),
    ]);

    const state = buildSpatialSkeletonSegmentRenderState(20380, graph, {
      filterText: "",
      nodeFilterType: SpatialSkeletonNodeFilterType.NONE,
      collapseRegularNodes: true,
      getNodeDescription() {
        return undefined;
      },
    });

    expect(state.matchedNodeCount).toBe(3);
    expect(state.displayedNodeCount).toBe(2);
    expect(state.branchCount).toBe(1);
    expect(state.rows.map((row) => row.node.nodeId)).toEqual([10, 12]);
  });

  it("treats node-type-only matches as disconnected visible branches", () => {
    const graph = buildSpatiallyIndexedSkeletonNavigationGraph([
      makeNode(20, undefined),
      makeNode(21, 20),
      makeNode(22, 20),
    ]);

    const state = buildSpatialSkeletonSegmentRenderState(20380, graph, {
      filterText: "",
      nodeFilterType: SpatialSkeletonNodeFilterType.VIRTUAL_END,
      collapseRegularNodes: false,
      getNodeDescription() {
        return undefined;
      },
    });

    expect(state.matchedNodeCount).toBe(2);
    expect(state.displayedNodeCount).toBe(2);
    expect(state.branchCount).toBe(2);
    expect(state.rows.map((row) => row.node.nodeId)).toEqual([21, 22]);
  });
});
