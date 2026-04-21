import { describe, expect, it } from "vitest";

import type { SpatiallyIndexedSkeletonNode } from "#src/skeleton/api.js";
import { buildSpatiallyIndexedSkeletonNavigationGraph } from "#src/skeleton/navigation.js";
import { SpatialSkeletonNodeFilterType } from "#src/skeleton/node_types.js";
import { buildSpatialSkeletonSegmentRenderState } from "#src/ui/spatial_skeleton_edit_tab_render_state.js";

function makeNode(
  nodeId: number,
  parentNodeId: number | undefined,
  options: {
    description?: string;
    isTrueEnd?: boolean;
  } = {},
): SpatiallyIndexedSkeletonNode {
  return {
    nodeId,
    segmentId: 20380,
    parentNodeId,
    position: new Float32Array([nodeId, nodeId + 1, nodeId + 2]),
    description: options.description,
    isTrueEnd: options.isTrueEnd ?? false,
  };
}

async function getSyncShownSpatialSkeletonSegmentIds() {
  const webglContextStub = new Proxy(
    {},
    {
      get: () => 0,
    },
  );
  (
    globalThis as { WebGL2RenderingContext?: unknown }
  ).WebGL2RenderingContext ??= webglContextStub;
  return (await import("#src/ui/spatial_skeleton_edit_tab.js"))
    .syncShownSpatialSkeletonSegmentIds;
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

  it("does not match coordinates, segment ids, or true-end state in the search filter", () => {
    const graph = buildSpatiallyIndexedSkeletonNavigationGraph([
      makeNode(101, undefined, { isTrueEnd: true }),
      makeNode(102, 101),
    ]);

    const byCoordinates = buildSpatialSkeletonSegmentRenderState(20380, graph, {
      filterText: "101 102 103",
      nodeFilterType: SpatialSkeletonNodeFilterType.NONE,
      collapseRegularNodes: false,
      getNodeDescription() {
        return undefined;
      },
    });
    const bySegmentId = buildSpatialSkeletonSegmentRenderState(20380, graph, {
      filterText: "20380",
      nodeFilterType: SpatialSkeletonNodeFilterType.NONE,
      collapseRegularNodes: false,
      getNodeDescription() {
        return undefined;
      },
    });
    const byTrueEndText = buildSpatialSkeletonSegmentRenderState(20380, graph, {
      filterText: "true end",
      nodeFilterType: SpatialSkeletonNodeFilterType.NONE,
      collapseRegularNodes: false,
      getNodeDescription() {
        return undefined;
      },
    });

    expect(byCoordinates.matchedNodeCount).toBe(0);
    expect(byCoordinates.displayedNodeCount).toBe(0);
    expect(bySegmentId.matchedNodeCount).toBe(0);
    expect(bySegmentId.displayedNodeCount).toBe(0);
    expect(byTrueEndText.matchedNodeCount).toBe(0);
    expect(byTrueEndText.displayedNodeCount).toBe(0);
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

  it("filters to nodes with non-empty descriptions", () => {
    const graph = buildSpatiallyIndexedSkeletonNavigationGraph([
      makeNode(30, undefined),
      makeNode(31, 30),
      makeNode(32, 30),
      makeNode(33, 30),
    ]);

    const state = buildSpatialSkeletonSegmentRenderState(20380, graph, {
      filterText: "",
      nodeFilterType: SpatialSkeletonNodeFilterType.HAS_DESCRIPTION,
      collapseRegularNodes: false,
      getNodeDescription(node) {
        switch (node.nodeId) {
          case 31:
            return "has description";
          case 32:
            return "";
          case 33:
            return "   ";
          default:
            return undefined;
        }
      },
    });

    expect(state.matchedNodeCount).toBe(1);
    expect(state.displayedNodeCount).toBe(1);
    expect(state.branchCount).toBe(1);
    expect(state.rows.map((row) => row.node.nodeId)).toEqual([31]);
  });
});

describe("spatial skeleton edit tab state", () => {
  it("preserves hidden segments across data refreshes while showing new segments", async () => {
    const syncShownSpatialSkeletonSegmentIds =
      await getSyncShownSpatialSkeletonSegmentIds();
    const shownSegmentIds = new Set([1, 3]);

    syncShownSpatialSkeletonSegmentIds(
      shownSegmentIds,
      [1, 2, 3, 4],
      [1, 2, 3],
    );

    expect([...shownSegmentIds]).toEqual([1, 3, 4]);
  });

  it("re-adds a segment when it becomes active again after leaving the view", async () => {
    const syncShownSpatialSkeletonSegmentIds =
      await getSyncShownSpatialSkeletonSegmentIds();
    const shownSegmentIds = new Set([1]);

    syncShownSpatialSkeletonSegmentIds(shownSegmentIds, [1], [1, 2]);
    syncShownSpatialSkeletonSegmentIds(shownSegmentIds, [1, 2], [1]);

    expect([...shownSegmentIds]).toEqual([1, 2]);
  });
});
