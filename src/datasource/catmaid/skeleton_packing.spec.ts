import { describe, expect, it } from "vitest";

import { packCatmaidSkeletonNodes } from "#src/datasource/catmaid/skeleton_packing.js";
import type { SpatiallyIndexedSkeletonNodeBase } from "#src/skeleton/api.js";

describe("datasource/catmaid/skeleton_packing", () => {
  it("packs vertex, segment, index, and pick-node data", () => {
    const nodes: SpatiallyIndexedSkeletonNodeBase[] = [
      {
        nodeId: 1n,
        parentNodeId: undefined,
        position: new Float32Array([1, 2, 3]),
        segmentId: 10n,
        sourceState: { revisionToken: "node-1" },
      },
      {
        nodeId: 2n,
        parentNodeId: 1n,
        position: new Float32Array([4, 5, 6]),
        segmentId: 10n,
        sourceState: { revisionToken: "node-2" },
      },
      {
        nodeId: 3n,
        parentNodeId: 99n,
        position: new Float32Array([7, 8, 9]),
        segmentId: 11n,
      },
    ];

    const packed = packCatmaidSkeletonNodes(nodes);

    expect(packed.vertexPositions).toEqual(
      Float32Array.of(1, 2, 3, 4, 5, 6, 7, 8, 9),
    );
    expect(packed.segmentIds).toEqual(BigUint64Array.of(10n, 10n, 11n));
    expect(packed.indices).toEqual(Uint32Array.of(1, 0));
    expect(packed.nodeIds).toEqual(BigUint64Array.of(1n, 2n, 3n));
    expect(packed.sourceStates).toEqual([
      { revisionToken: "node-1" },
      { revisionToken: "node-2" },
      undefined,
    ]);
  });

  it("preserves large segment ids exactly", () => {
    const largeSegmentId = 9_007_199_254_740_993n;
    const nodes: SpatiallyIndexedSkeletonNodeBase[] = [
      {
        nodeId: 1n,
        parentNodeId: undefined,
        position: new Float32Array([1, 2, 3]),
        segmentId: largeSegmentId,
      },
    ];

    const packed = packCatmaidSkeletonNodes(nodes);

    expect(packed.segmentIds).toEqual(BigUint64Array.of(largeSegmentId));
  });
});
