/**
 * @license
 * Copyright 2026 Google Inc.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

import { describe, expect, it } from "vitest";
import { numRenderScaleHistogramBins } from "#src/render_scale_statistics.js";
import {
  RenderScaleWidget,
  SpatialSkeletonGridRenderScaleWidget,
} from "#src/widget/render_scale_widget.js";

type RenderScaleWidgetMapping = {
  canvasXToHistogramOffset(canvasX: number, width: number): number;
  histogramOffsetToCanvasX(offset: number, width: number): number;
  getWheelMoveValue(event: WheelEvent): number;
};

describe("RenderScaleWidget", () => {
  it("keeps the default left-to-right axis mapping", () => {
    const widget = Object.create(
      RenderScaleWidget.prototype,
    ) as RenderScaleWidgetMapping;

    expect(widget.canvasXToHistogramOffset(0, 400)).toBe(0);
    expect(widget.canvasXToHistogramOffset(200, 400)).toBe(
      numRenderScaleHistogramBins / 2,
    );
    expect(widget.canvasXToHistogramOffset(400, 400)).toBe(
      numRenderScaleHistogramBins,
    );

    expect(widget.histogramOffsetToCanvasX(0, 400)).toBe(0);
    expect(
      widget.histogramOffsetToCanvasX(numRenderScaleHistogramBins / 2, 400),
    ).toBe(200);
    expect(
      widget.histogramOffsetToCanvasX(numRenderScaleHistogramBins, 400),
    ).toBe(400);
  });
});

describe("SpatialSkeletonGridRenderScaleWidget", () => {
  it("reverses the axis so smaller cell sizes are on the right", () => {
    const widget = Object.create(
      SpatialSkeletonGridRenderScaleWidget.prototype,
    ) as RenderScaleWidgetMapping;

    expect(widget.canvasXToHistogramOffset(0, 400)).toBe(
      numRenderScaleHistogramBins,
    );
    expect(widget.canvasXToHistogramOffset(200, 400)).toBe(
      numRenderScaleHistogramBins / 2,
    );
    expect(widget.canvasXToHistogramOffset(400, 400)).toBe(0);

    expect(widget.histogramOffsetToCanvasX(0, 400)).toBe(400);
    expect(
      widget.histogramOffsetToCanvasX(numRenderScaleHistogramBins / 2, 400),
    ).toBe(200);
    expect(
      widget.histogramOffsetToCanvasX(numRenderScaleHistogramBins, 400),
    ).toBe(0);
  });

  it("reverses wheel direction to match the reversed axis", () => {
    const widget = Object.create(
      SpatialSkeletonGridRenderScaleWidget.prototype,
    ) as RenderScaleWidgetMapping;

    expect(
      widget.getWheelMoveValue(new WheelEvent("wheel", { deltaY: 6 })),
    ).toBe(-6);
  });
});
