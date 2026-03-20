import { describe, expect, it, vi } from "vitest";

import { WatchableValue } from "#src/trackable_value.js";

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

const {
  SegmentationUserLayer,
} = await import("#src/layer/segmentation/index.js");

const {
  PerspectiveViewSpatiallyIndexedSkeletonLayer,
  SliceViewPanelSpatiallyIndexedSkeletonLayer,
  SliceViewSpatiallyIndexedSkeletonLayer,
  MultiscaleSliceViewSpatiallyIndexedSkeletonLayer,
} = await import("#src/skeleton/frontend.js");

describe("layer/segmentation spatial skeleton chunk stats", () => {
  it("tracks combined chunk load state from the loading render layers only", () => {
    const perspectiveLayer = Object.assign(
      Object.create(PerspectiveViewSpatiallyIndexedSkeletonLayer.prototype),
      {
        layerChunkProgressInfo: {
          numVisibleChunksNeeded: 5,
          numVisibleChunksAvailable: 3,
        },
      },
    );
    const sliceLayer = Object.assign(
      Object.create(SliceViewSpatiallyIndexedSkeletonLayer.prototype),
      {
        layerChunkProgressInfo: {
          numVisibleChunksNeeded: 4,
          numVisibleChunksAvailable: 2,
        },
      },
    );
    const multiscaleSliceLayer = Object.assign(
      Object.create(
        MultiscaleSliceViewSpatiallyIndexedSkeletonLayer.prototype,
      ),
      {
        layerChunkProgressInfo: {
          numVisibleChunksNeeded: 6,
          numVisibleChunksAvailable: 5,
        },
      },
    );
    const slicePanelLayer = Object.assign(
      Object.create(SliceViewPanelSpatiallyIndexedSkeletonLayer.prototype),
      {
        layerChunkProgressInfo: {
          numVisibleChunksNeeded: 100,
          numVisibleChunksAvailable: 100,
        },
      },
    );

    const layer = Object.assign(Object.create(SegmentationUserLayer.prototype), {
      renderLayers: [
        perspectiveLayer,
        sliceLayer,
        multiscaleSliceLayer,
        slicePanelLayer,
      ],
      displayState: {
        spatialSkeletonGridChunkStats2d: new WatchableValue({
          presentCount: 0,
          totalCount: 0,
        }),
        spatialSkeletonGridChunkStats3d: new WatchableValue({
          presentCount: 0,
          totalCount: 0,
        }),
      },
      spatialSkeletonState: {
        updateChunkLoadState: vi.fn(),
      },
      updateSpatialSkeletonSourceState: vi.fn(),
    });

    layer.updateSpatialSkeletonChunkLoadState();

    expect(layer.spatialSkeletonState.updateChunkLoadState).toHaveBeenCalledWith(
      15,
      10,
    );
  });
});
