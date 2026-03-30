import { describe, expect, it } from "vitest";

import {
  classifySpatialSkeletonDisplayNodeType,
  getSpatialSkeletonNodeFilterLabel,
  getSpatialSkeletonNodeIconFilterType,
  hasSpatialSkeletonTrueEndLabel,
  isSpatialSkeletonClosedEndLabel,
  matchesSpatialSkeletonNodeFilter,
  SpatialSkeletonNodeFilterType,
  updateSpatialSkeletonTrueEndLabels,
} from "#src/skeleton/node_types.js";

describe("skeleton node types", () => {
  it("classifies display node types for roots, branches, regular nodes, and virtual ends", () => {
    expect(
      classifySpatialSkeletonDisplayNodeType({ parentNodeId: undefined }, 0),
    ).toBe("root");
    expect(classifySpatialSkeletonDisplayNodeType({ parentNodeId: 1 }, 2)).toBe(
      "branchStart",
    );
    expect(classifySpatialSkeletonDisplayNodeType({ parentNodeId: 1 }, 1)).toBe(
      "regular",
    );
    expect(classifySpatialSkeletonDisplayNodeType({ parentNodeId: 1 }, 0)).toBe(
      "virtualEnd",
    );
    expect(
      classifySpatialSkeletonDisplayNodeType({ parentNodeId: 1 }, 0, false),
    ).toBe("root");
  });

  it("detects true-end and closed-end labels case-insensitively", () => {
    expect(hasSpatialSkeletonTrueEndLabel(["ENDS"])).toBe(true);
    expect(hasSpatialSkeletonTrueEndLabel(["leaf"])).toBe(false);
    expect(isSpatialSkeletonClosedEndLabel("uncertain continuation")).toBe(
      true,
    );
    expect(isSpatialSkeletonClosedEndLabel("posterior end")).toBe(true);
    expect(isSpatialSkeletonClosedEndLabel("axon branch")).toBe(false);
  });

  it("adds or removes the true-end label without disturbing other labels", () => {
    expect(
      updateSpatialSkeletonTrueEndLabels(["posterior end", "ends"], false),
    ).toEqual(["posterior end"]);
    expect(updateSpatialSkeletonTrueEndLabels(["posterior end"], true)).toEqual(
      ["posterior end", "ends"],
    );
    expect(
      updateSpatialSkeletonTrueEndLabels(undefined, false),
    ).toBeUndefined();
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
