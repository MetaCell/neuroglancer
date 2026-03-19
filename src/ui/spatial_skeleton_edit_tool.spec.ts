import { describe, expect, it } from "vitest";

import { setSpatialSkeletonMode3dToLinesAndPoints } from "#src/skeleton/edit_mode_rendering.js";
import { SkeletonRenderMode } from "#src/skeleton/render_mode.js";

describe("spatial_skeleton_edit_tool", () => {
  it("switches 3d skeleton rendering to lines and points", () => {
    const layer = {
      displayState: {
        skeletonRenderingOptions: {
          params2d: { mode: { value: SkeletonRenderMode.LINES } },
          params3d: { mode: { value: SkeletonRenderMode.LINES } },
        },
      },
    } as any;

    setSpatialSkeletonMode3dToLinesAndPoints(layer);

    expect(
      layer.displayState.skeletonRenderingOptions.params3d.mode.value,
    ).toBe(SkeletonRenderMode.LINES_AND_POINTS);
    expect(
      layer.displayState.skeletonRenderingOptions.params2d.mode.value,
    ).toBe(SkeletonRenderMode.LINES);
  });
});
