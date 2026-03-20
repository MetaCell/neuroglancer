import { SkeletonRenderMode } from "#src/skeleton/render_mode.js";

export interface SkeletonMode3dLayerLike {
  displayState: {
    skeletonRenderingOptions: {
      params3d: {
        mode: {
          value: SkeletonRenderMode;
        };
      };
    };
  };
}

export function setSpatialSkeletonMode3dToLinesAndPoints(
  layer: SkeletonMode3dLayerLike,
) {
  layer.displayState.skeletonRenderingOptions.params3d.mode.value =
    SkeletonRenderMode.LINES_AND_POINTS;
}
