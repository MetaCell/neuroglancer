import { describe, expect, it } from "vitest";

import { buildSpatiallyIndexedSkeletonOverlayGeometry } from "#src/skeleton/overlay_geometry.js";

describe("buildSpatiallyIndexedSkeletonOverlayGeometry", () => {
  it("packs inspected segment nodes into overlay geometry with deduped nodes", () => {
    const geometry = buildSpatiallyIndexedSkeletonOverlayGeometry(
      [
        [
          {
            nodeId: 1,
            segmentId: 11,
            position: new Float32Array([1, 2, 3]),
          },
          {
            nodeId: 2,
            segmentId: 11,
            position: new Float32Array([4, 5, 6]),
            parentNodeId: 1,
          },
        ],
        [
          {
            nodeId: 2,
            segmentId: 11,
            position: new Float32Array([40, 50, 60]),
            parentNodeId: 1,
          },
          {
            nodeId: 3,
            segmentId: 13,
            position: new Float32Array([7, 8, 9]),
          },
        ],
      ],
      {
        selectedNodeId: 2,
        getPendingNodePosition: (nodeId) =>
          nodeId === 3 ? new Float32Array([70, 80, 90]) : undefined,
      },
    );

    expect(geometry.numVertices).toBe(3);
    expect([...geometry.nodeIds]).toEqual([1, 2, 3]);
    expect([...geometry.segmentIds]).toEqual([11, 11, 13]);
    expect([...geometry.selected]).toEqual([0, 1, 0]);
    expect([...geometry.positions]).toEqual([1, 2, 3, 4, 5, 6, 70, 80, 90]);
    expect([...geometry.indices]).toEqual([1, 0]);
    expect([...geometry.pickEdgeSegmentIds]).toEqual([11]);
  });
});
