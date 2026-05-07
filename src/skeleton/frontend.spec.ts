import { describe, expect, it, vi } from "vitest";

import { resolveSpatiallyIndexedSkeletonSegmentPick } from "#src/skeleton/picking.js";
import { spatiallyIndexedSkeletonTextureAttributeSpecs } from "#src/skeleton/spatial_attribute_layout.js";
import { Uint64Set } from "#src/uint64_set.js";
import { DataType } from "#src/util/data_type.js";

if (!("WebGL2RenderingContext" in globalThis)) {
  Object.defineProperty(globalThis, "WebGL2RenderingContext", {
    value: new Proxy(class WebGL2RenderingContext {} as any, {
      get(target, property, receiver) {
        if (Reflect.has(target, property)) {
          return Reflect.get(target, property, receiver);
        }
        return 0;
      },
    }),
    configurable: true,
  });
}

const { SpatiallyIndexedSkeletonLayer } = await import(
  "#src/skeleton/frontend.js"
);

describe("resolveSpatiallyIndexedSkeletonSegmentPick", () => {
  it("returns the node segment id for direct node picks", () => {
    const chunk = {
      indices: new Uint32Array([0, 1, 1, 2]),
      numVertices: 3,
    };
    const segmentIds = new BigUint64Array([11n, 13n, 17n]);

    expect(
      resolveSpatiallyIndexedSkeletonSegmentPick(chunk, segmentIds, 1, "node"),
    ).toBe(13n);
  });

  it("returns the first valid endpoint segment id for direct edge picks", () => {
    const chunk = {
      indices: new Uint32Array([0, 1, 1, 2]),
      numVertices: 3,
    };
    const segmentIds = new BigUint64Array([0n, 19n, 23n]);

    expect(
      resolveSpatiallyIndexedSkeletonSegmentPick(chunk, segmentIds, 0, "edge"),
    ).toBe(19n);
    expect(
      resolveSpatiallyIndexedSkeletonSegmentPick(chunk, segmentIds, 1, "edge"),
    ).toBe(19n);
  });

  it("returns undefined for out-of-range direct picks", () => {
    const chunk = {
      indices: new Uint32Array([0, 1]),
      numVertices: 2,
    };
    const segmentIds = new BigUint64Array([5n, 7n]);

    expect(
      resolveSpatiallyIndexedSkeletonSegmentPick(chunk, segmentIds, 4, "node"),
    ).toBeUndefined();
    expect(
      resolveSpatiallyIndexedSkeletonSegmentPick(chunk, segmentIds, 2, "edge"),
    ).toBeUndefined();
  });
});

describe("SpatiallyIndexedSkeletonLayer browse node picks", () => {
  it("resolves browse node picks with node id and source state", () => {
    const positions = new Float32Array([1, 2, 3, 4, 5, 6]);
    const segmentIds = new BigUint64Array([11n, 17n]);
    const vertexBytes = new Uint8Array(
      positions.byteLength + segmentIds.byteLength,
    );
    vertexBytes.set(new Uint8Array(positions.buffer), 0);
    vertexBytes.set(new Uint8Array(segmentIds.buffer), positions.byteLength);
    const chunk = {
      vertexAttributes: vertexBytes,
      vertexAttributeOffsets: new Uint32Array([0, positions.byteLength]),
      numVertices: 2,
      indices: new Uint32Array([0, 1]),
      nodeIds: new BigUint64Array([101n, 202n]),
      nodeSourceStates: [
        { revisionToken: "2026-03-29T11:50:00Z" },
        { revisionToken: "2026-03-29T11:51:00Z" },
      ],
    };
    const layer = Object.create(SpatiallyIndexedSkeletonLayer.prototype);

    expect((layer as any).resolveNodePickFromChunk(chunk, 1)).toEqual({
      nodeId: 202n,
      segmentId: 17n,
      position: new Float32Array([4, 5, 6]),
      sourceState: { revisionToken: "2026-03-29T11:51:00Z" },
    });
  });
});

describe("SpatiallyIndexedSkeletonLayer getNodes", () => {
  it("matches safe bigint segment filters without rounding through an unsafe number", () => {
    const layer = Object.assign(
      Object.create(SpatiallyIndexedSkeletonLayer.prototype),
      {
        inspectionState: {
          getCachedSegmentNodes: (segmentId: bigint) =>
            segmentId === 17n
              ? [
                  {
                    nodeId: 101n,
                    segmentId: 17n,
                    position: new Float32Array([1, 2, 3]),
                  },
                ]
              : undefined,
        },
        getCachedNodeInfo: (nodeId: bigint) =>
          nodeId === 101n
            ? {
                nodeId: 101n,
                segmentId: 17n,
                position: new Float32Array([1, 2, 3]),
              }
            : undefined,
        getPendingNodePositionOverride: undefined,
      },
    );

    expect(layer.getNodes({ segmentId: 17n })).toEqual([
      {
        nodeId: 101n,
        segmentId: 17n,
        position: new Float32Array([1, 2, 3]),
      },
    ]);
  });

  it("matches large bigint segment filters exactly", () => {
    const largeSegmentId = 9007199254740993n;
    const layer = Object.assign(
      Object.create(SpatiallyIndexedSkeletonLayer.prototype),
      {
        inspectionState: {
          getCachedSegmentNodes: (segmentId: bigint) =>
            segmentId === largeSegmentId
              ? [
                  {
                    nodeId: largeSegmentId + 1n,
                    segmentId: largeSegmentId,
                    position: new Float32Array([1, 2, 3]),
                  },
                ]
              : undefined,
        },
        getCachedNodeInfo: (nodeId: bigint) =>
          nodeId === largeSegmentId + 1n
            ? {
                nodeId: largeSegmentId + 1n,
                segmentId: largeSegmentId,
                position: new Float32Array([1, 2, 3]),
              }
            : undefined,
        getPendingNodePositionOverride: undefined,
      },
    );

    expect(layer.getNodes({ segmentId: largeSegmentId })).toEqual([
      {
        nodeId: largeSegmentId + 1n,
        segmentId: largeSegmentId,
        position: new Float32Array([1, 2, 3]),
      },
    ]);
  });
});

describe("spatiallyIndexedSkeletonTextureAttributeSpecs", () => {
  it("keeps the browse path upload layout to position plus segment", () => {
    expect(spatiallyIndexedSkeletonTextureAttributeSpecs).toEqual([
      { name: "position", dataType: DataType.FLOAT32, numComponents: 3 },
      { name: "segment", dataType: DataType.UINT64, numComponents: 1 },
    ]);
  });
});

describe("SpatiallyIndexedSkeletonLayer browse exclusions", () => {
  it("includes suppressed browse segments even when no overlay segment is loaded", () => {
    const layer = Object.assign(
      Object.create(SpatiallyIndexedSkeletonLayer.prototype),
      {
        suppressedBrowseSegmentIds: new Set<bigint>(),
        browseExcludedSegments: new Uint64Set(),
        browseExcludedSegmentsKey: undefined,
        redrawNeeded: { dispatch: vi.fn() },
        getLoadedOverlaySegmentIds: () => [],
      },
    );

    expect(layer.suppressBrowseSegment(29n)).toBe(true);
    expect(layer.redrawNeeded.dispatch).toHaveBeenCalledTimes(1);

    const excludedSegments = (layer as any).getBrowsePassExcludedSegments();
    expect(excludedSegments).toBeInstanceOf(Uint64Set);
    expect([...excludedSegments]).toEqual([29n]);
  });
});
