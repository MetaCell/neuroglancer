import { describe, expect, it, vi } from "vitest";

import { resolveSpatiallyIndexedSkeletonSegmentPick } from "#src/skeleton/picking.js";
import { spatiallyIndexedSkeletonTextureAttributeSpecs } from "#src/skeleton/spatial_attribute_layout.js";
import { WatchableValue } from "#src/trackable_value.js";
import { Uint64Set } from "#src/uint64_set.js";
import { DataType } from "#src/util/data_type.js";
import { getObjectId } from "#src/util/object_id.js";

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
    const segmentIds = new Uint32Array([11, 13, 17]);

    expect(
      resolveSpatiallyIndexedSkeletonSegmentPick(chunk, segmentIds, 1, "node"),
    ).toBe(13);
  });

  it("returns the first valid endpoint segment id for direct edge picks", () => {
    const chunk = {
      indices: new Uint32Array([0, 1, 1, 2]),
      numVertices: 3,
    };
    const segmentIds = new Uint32Array([0, 19, 23]);

    expect(
      resolveSpatiallyIndexedSkeletonSegmentPick(chunk, segmentIds, 0, "edge"),
    ).toBe(19);
    expect(
      resolveSpatiallyIndexedSkeletonSegmentPick(chunk, segmentIds, 1, "edge"),
    ).toBe(19);
  });

  it("returns undefined for out-of-range direct picks", () => {
    const chunk = {
      indices: new Uint32Array([0, 1]),
      numVertices: 2,
    };
    const segmentIds = new Uint32Array([5, 7]);

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
    const segmentIds = new Uint32Array([11, 17]);
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
      nodeIds: new Int32Array([101, 202]),
      nodeSourceStates: [
        { revisionToken: "2026-03-29T11:50:00Z" },
        { revisionToken: "2026-03-29T11:51:00Z" },
      ],
    };
    const layer = Object.create(SpatiallyIndexedSkeletonLayer.prototype);

    expect((layer as any).resolveNodePickFromChunk(chunk, 1)).toEqual({
      nodeId: 202,
      segmentId: 17,
      position: new Float32Array([4, 5, 6]),
      sourceState: { revisionToken: "2026-03-29T11:51:00Z" },
    });
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

describe("SpatiallyIndexedSkeletonLayer browse exclusions", () => {
  it("includes suppressed browse segments even when no overlay segment is loaded", () => {
    const layer = Object.assign(
      Object.create(SpatiallyIndexedSkeletonLayer.prototype),
      {
        suppressedBrowseSegmentIds: new Set<number>(),
        browseExcludedSegments: new Uint64Set(),
        browseExcludedSegmentsKey: undefined,
        redrawNeeded: { dispatch: vi.fn() },
        getLoadedOverlaySegmentIds: () => [],
      },
    );

    expect(layer.suppressBrowseSegment(29)).toBe(true);
    expect(layer.redrawNeeded.dispatch).toHaveBeenCalledTimes(1);

    const excludedSegments = (layer as any).getBrowsePassExcludedSegments();
    expect(excludedSegments).toBeInstanceOf(Uint64Set);
    expect([...excludedSegments]).toEqual([29n]);
  });
});

describe("SpatiallyIndexedSkeletonLayer chunk stats", () => {
  it("dedupes 2d chunk stats across rendered views for the selected grid source", () => {
    const coarseSource = { parameters: { gridIndex: 0 } };
    const fineSource = { parameters: { gridIndex: 1 } };
    const coarseSourceId = getObjectId(coarseSource);
    const fineSourceId = getObjectId(fineSource);
    const layer = Object.assign(
      Object.create(SpatiallyIndexedSkeletonLayer.prototype),
      {
        displayState: {
          spatialSkeletonGridLevel2d: { value: 1 },
          spatialSkeletonGridLevel3d: { value: 0 },
          spatialSkeletonGridChunkStats2d: new WatchableValue({
            presentCount: 0,
            totalCount: 0,
          }),
          spatialSkeletonGridChunkStats3d: new WatchableValue({
            presentCount: 0,
            totalCount: 0,
          }),
        },
        sources: [],
        sources2d: [
          {
            chunkSource: coarseSource,
            chunkToMultiscaleTransform: {},
          },
          {
            chunkSource: fineSource,
            chunkToMultiscaleTransform: {},
          },
        ],
        visibleChunkKeysByRenderedView: new Map(),
      },
    );

    (layer as any).setVisibleChunkKeysForRenderedView(
      "2d",
      11,
      new Map([
        [
          coarseSourceId,
          {
            presentChunkKeys: new Set(["ignored"]),
            totalChunkKeys: new Set(["ignored"]),
          },
        ],
        [
          fineSourceId,
          {
            presentChunkKeys: new Set(["a"]),
            totalChunkKeys: new Set(["a", "b"]),
          },
        ],
      ]),
    );
    expect(layer.displayState.spatialSkeletonGridChunkStats2d.value).toEqual({
      presentCount: 1,
      totalCount: 2,
    });

    (layer as any).setVisibleChunkKeysForRenderedView(
      "2d",
      22,
      new Map([
        [
          fineSourceId,
          {
            presentChunkKeys: new Set(["c"]),
            totalChunkKeys: new Set(["b", "c"]),
          },
        ],
      ]),
    );
    expect(layer.displayState.spatialSkeletonGridChunkStats2d.value).toEqual({
      presentCount: 2,
      totalCount: 3,
    });

    (layer as any).clearVisibleChunkKeysForRenderedView("2d", 11);
    expect(layer.displayState.spatialSkeletonGridChunkStats2d.value).toEqual({
      presentCount: 1,
      totalCount: 2,
    });
  });
});
