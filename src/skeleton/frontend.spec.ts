import { describe, expect, it } from "vitest";

import { resolveSpatiallyIndexedSkeletonSegmentPick } from "#src/skeleton/picking.js";
import { spatiallyIndexedSkeletonTextureAttributeSpecs } from "#src/skeleton/spatial_attribute_layout.js";
import { DataType } from "#src/util/data_type.js";

describe("resolveSpatiallyIndexedSkeletonSegmentPick", () => {
  it("returns the node segment id for direct node picks", () => {
    const chunk = {
      indices: new Uint32Array([0, 1, 1, 2]),
      numVertices: 3,
    };
    const segmentIds = new Uint32Array([11, 13, 17]);

    expect(
      resolveSpatiallyIndexedSkeletonSegmentPick(
        chunk,
        segmentIds,
        1,
        "node",
      ),
    ).toBe(13);
  });

  it("returns the first valid endpoint segment id for direct edge picks", () => {
    const chunk = {
      indices: new Uint32Array([0, 1, 1, 2]),
      numVertices: 3,
    };
    const segmentIds = new Uint32Array([0, 19, 23]);

    expect(
      resolveSpatiallyIndexedSkeletonSegmentPick(
        chunk,
        segmentIds,
        0,
        "edge",
      ),
    ).toBe(19);
    expect(
      resolveSpatiallyIndexedSkeletonSegmentPick(
        chunk,
        segmentIds,
        1,
        "edge",
      ),
    ).toBe(19);
  });

  it("returns undefined for out-of-range direct picks", () => {
    const chunk = {
      indices: new Uint32Array([0, 1]),
      numVertices: 2,
    };
    const segmentIds = new Uint32Array([5, 7]);

    expect(
      resolveSpatiallyIndexedSkeletonSegmentPick(
        chunk,
        segmentIds,
        4,
        "node",
      ),
    ).toBeUndefined();
    expect(
      resolveSpatiallyIndexedSkeletonSegmentPick(
        chunk,
        segmentIds,
        2,
        "edge",
      ),
    ).toBeUndefined();
  });
});

describe("spatiallyIndexedSkeletonTextureAttributeSpecs", () => {
  it("keeps the browse path upload layout to position plus segment", () => {
    expect(spatiallyIndexedSkeletonTextureAttributeSpecs).toEqual([
      { name: "position", dataType: DataType.FLOAT32, numComponents: 3 },
      { name: "segment", dataType: DataType.UINT32, numComponents: 1 },
    ]);
  });
});
