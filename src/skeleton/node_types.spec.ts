import { describe, expect, it } from "vitest";

import type { SpatiallyIndexedSkeletonNode } from "#src/skeleton/api.js";
import {
  classifySpatialSkeletonDisplayNodeType,
  getSpatialSkeletonNodeFilterLabel,
  getSpatialSkeletonNodeIconFilterType,
  matchesSpatialSkeletonNodeFilter,
  SpatialSkeletonNodeFilterType,
} from "#src/skeleton/node_types.js";

function makeNode(
  overrides: Partial<SpatiallyIndexedSkeletonNode> = {},
): SpatiallyIndexedSkeletonNode {
  return {
    nodeId: 1,
    segmentId: 1,
    position: new Float32Array([0, 0, 0]),
    isTrueEnd: false,
    ...overrides,
  };
}

describe("skeleton node types", () => {
  it("classifies display node types for roots, branches, regular nodes, and virtual ends", () => {
    expect(
      classifySpatialSkeletonDisplayNodeType(
        makeNode({ parentNodeId: undefined }),
        0,
      ),
    ).toBe("root");
    expect(
      classifySpatialSkeletonDisplayNodeType(makeNode({ parentNodeId: 1 }), 2),
    ).toBe("branchStart");
    expect(
      classifySpatialSkeletonDisplayNodeType(makeNode({ parentNodeId: 1 }), 1),
    ).toBe("regular");
    expect(
      classifySpatialSkeletonDisplayNodeType(makeNode({ parentNodeId: 1 }), 0),
    ).toBe("virtualEnd");
    expect(
      classifySpatialSkeletonDisplayNodeType(
        makeNode({ parentNodeId: 1 }),
        0,
        false,
      ),
    ).toBe("root");
  });

  it("matches the dropdown filter semantics", () => {
    const rootLeaf = {
      isLeaf: true,
      nodeHasDescription: false,
      nodeIsTrueEnd: false,
      nodeType: "root" as const,
    };
    const virtualEnd = {
      isLeaf: true,
      nodeHasDescription: false,
      nodeIsTrueEnd: false,
      nodeType: "virtualEnd" as const,
    };
    const trueEnd = {
      isLeaf: true,
      nodeHasDescription: false,
      nodeIsTrueEnd: true,
      nodeType: "virtualEnd" as const,
    };
    const describedNode = {
      isLeaf: false,
      nodeHasDescription: true,
      nodeIsTrueEnd: false,
      nodeType: "regular" as const,
    };

    expect(
      matchesSpatialSkeletonNodeFilter(
        SpatialSkeletonNodeFilterType.LEAF,
        rootLeaf,
      ),
    ).toBe(true);
    expect(
      matchesSpatialSkeletonNodeFilter(
        SpatialSkeletonNodeFilterType.VIRTUAL_END,
        rootLeaf,
      ),
    ).toBe(true);
    expect(
      matchesSpatialSkeletonNodeFilter(
        SpatialSkeletonNodeFilterType.VIRTUAL_END,
        virtualEnd,
      ),
    ).toBe(true);
    expect(
      matchesSpatialSkeletonNodeFilter(
        SpatialSkeletonNodeFilterType.VIRTUAL_END,
        trueEnd,
      ),
    ).toBe(false);
    expect(
      matchesSpatialSkeletonNodeFilter(
        SpatialSkeletonNodeFilterType.TRUE_END,
        trueEnd,
      ),
    ).toBe(true);
    expect(
      matchesSpatialSkeletonNodeFilter(
        SpatialSkeletonNodeFilterType.HAS_DESCRIPTION,
        describedNode,
      ),
    ).toBe(true);
    expect(
      matchesSpatialSkeletonNodeFilter(
        SpatialSkeletonNodeFilterType.HAS_DESCRIPTION,
        rootLeaf,
      ),
    ).toBe(false);
  });

  it("reuses the terminal filter enum for row icon decisions", () => {
    expect(
      getSpatialSkeletonNodeIconFilterType({
        nodeIsTrueEnd: false,
        nodeType: "virtualEnd",
      }),
    ).toBe(SpatialSkeletonNodeFilterType.VIRTUAL_END);
    expect(
      getSpatialSkeletonNodeIconFilterType({
        nodeIsTrueEnd: true,
        nodeType: "regular",
      }),
    ).toBe(SpatialSkeletonNodeFilterType.TRUE_END);
    expect(
      getSpatialSkeletonNodeIconFilterType({
        nodeIsTrueEnd: false,
        nodeType: "root",
      }),
    ).toBeUndefined();
    expect(
      getSpatialSkeletonNodeFilterLabel(SpatialSkeletonNodeFilterType.TRUE_END),
    ).toBe("True end");
    expect(
      getSpatialSkeletonNodeFilterLabel(
        SpatialSkeletonNodeFilterType.HAS_DESCRIPTION,
      ),
    ).toBe("Has description");
  });
});
