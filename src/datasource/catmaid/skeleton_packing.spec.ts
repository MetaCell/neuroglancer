import { describe, expect, it } from "vitest";

import type { SpatiallyIndexedSkeletonNode } from "#src/skeleton/api.js";
import { packCatmaidSkeletonNodes } from "#src/datasource/catmaid/skeleton_packing.js";

describe("datasource/catmaid/skeleton_packing", () => {
  it("packs vertex, segment, index, and missing-connection data", () => {
    const nodes: SpatiallyIndexedSkeletonNode[] = [
      { id: 1, parent_id: null, x: 1, y: 2, z: 3, skeleton_id: 10 },
      { id: 2, parent_id: 1, x: 4, y: 5, z: 6, skeleton_id: 10 },
      { id: 3, parent_id: 99, x: 7, y: 8, z: 9, skeleton_id: 11 },
    ];

    const packed = packCatmaidSkeletonNodes(nodes, {
      recordMissingConnections: true,
    });

    expect(packed.vertexPositions).toEqual(
      Float32Array.of(1, 2, 3, 4, 5, 6, 7, 8, 9),
    );
    expect(packed.segmentIds).toEqual(Uint32Array.of(10, 10, 11));
    expect(packed.indices).toEqual(Uint32Array.of(1, 0));
    expect(Array.from(packed.nodeMap.entries())).toEqual([
      [1, 0],
      [2, 1],
      [3, 2],
    ]);
    expect(packed.missingConnections).toEqual([
      {
        nodeId: 3,
        parentId: 99,
        vertexIndex: 2,
        skeletonId: 11,
      },
    ]);
  });

  it("skips missing-connection bookkeeping when not requested", () => {
    const nodes: SpatiallyIndexedSkeletonNode[] = [
      { id: 1, parent_id: null, x: 1, y: 2, z: 3, skeleton_id: 10 },
      { id: 2, parent_id: 99, x: 4, y: 5, z: 6, skeleton_id: 10 },
    ];

    const packed = packCatmaidSkeletonNodes(nodes);

    expect(packed.indices).toEqual(new Uint32Array());
    expect(packed.missingConnections).toEqual([]);
  });

  it("preserves large segment ids exactly", () => {
    const largeSegmentId = 16_777_217;
    const nodes: SpatiallyIndexedSkeletonNode[] = [
      { id: 1, parent_id: null, x: 1, y: 2, z: 3, skeleton_id: largeSegmentId },
    ];

    const packed = packCatmaidSkeletonNodes(nodes);

    expect(packed.segmentIds).toEqual(Uint32Array.of(largeSegmentId));
  });
});
