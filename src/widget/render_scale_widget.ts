/**
 * @license
 * Copyright 2019 Google Inc.
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

import "#src/widget/render_scale_widget.css";

import { debounce, throttle } from "lodash-es";
import type { UserLayer } from "#src/layer/index.js";
import type { RenderScaleHistogram } from "#src/render_scale_statistics.js";
import {
  getRenderScaleFromHistogramOffset,
  getRenderScaleHistogramOffset,
  numRenderScaleHistogramBins,
  renderScaleHistogramBinSize,
  renderScaleHistogramOrigin,
} from "#src/render_scale_statistics.js";
import { TrackableBooleanCheckbox } from "#src/trackable_boolean.js";
import type { TrackableValueInterface, WatchableValueInterface } from "#src/trackable_value.js";
import { WatchableValue } from "#src/trackable_value.js";
import { serializeColor } from "#src/util/color.js";
import { hsvToRgb } from "#src/util/colorspace.js";
import { RefCounted } from "#src/util/disposable.js";
import type { ActionEvent } from "#src/util/event_action_map.js";
import {
  EventActionMap,
  registerActionListener,
} from "#src/util/event_action_map.js";
import { vec3 } from "#src/util/geom.js";
import { clampToInterval } from "#src/util/lerp.js";
import { MouseEventBinder } from "#src/util/mouse_bindings.js";
import { numberToStringFixed } from "#src/util/number_to_string.js";
import { formatScaleWithUnitAsString } from "#src/util/si_units.js";
import type { LayerControlFactory } from "#src/widget/layer_control.js";

const updateInterval = 200;

const inputEventMap = EventActionMap.fromObject({
  mousedown0: { action: "set" },
  wheel: { action: "adjust-via-wheel" },
  dblclick0: { action: "reset" },
});

function formatPixelNumber(x: number) {
  if (x < 1 || x > 1024) {
    const exponent = Math.log2(x) | 0;
    const coeff = x / 2 ** exponent;
    return `${numberToStringFixed(coeff, 1)}p${exponent}`;
  }
  return Math.round(x) + "";
}

export interface RenderScaleWidgetOptions {
  histogram: RenderScaleHistogram;
  target: TrackableValueInterface<number>;
}

type SpatialSkeletonGridLevel = {
  size: { x: number; y: number; z: number };
  lod: number;
};

export interface SpatialSkeletonGridResolutionWidgetOptions {
  levels: WatchableValueInterface<SpatialSkeletonGridLevel[]>;
  target: TrackableValueInterface<number>;
  relative: WatchableValueInterface<boolean>;
  pixelSize: WatchableValueInterface<number>;
  relativeLabel?: string;
  relativeTooltip?: string;
}

export class RenderScaleWidget extends RefCounted {
  label = document.createElement("div");
  element = document.createElement("div");
  canvas = document.createElement("canvas");
  legend = document.createElement("div");
  legendRenderScale = document.createElement("div");
  legendSpatialScale = document.createElement("div");
  legendChunks = document.createElement("div");
  protected logScaleOrigin = renderScaleHistogramOrigin;
  protected unitOfTarget = "px";
  private ctx = this.canvas.getContext("2d")!;
  hoverTarget = new WatchableValue<[number, number] | undefined>(undefined);
  private throttledUpdateView = this.registerCancellable(
    throttle(() => this.debouncedUpdateView(), updateInterval, {
      leading: true,
      trailing: true,
    }),
  );
  private debouncedUpdateView = this.registerCancellable(
    debounce(() => this.updateView(), 0),
  );

  adjustViaWheel(event: WheelEvent) {
    const deltaY = this.getWheelMoveValue(event);
    if (deltaY === 0) {
      return;
    }
    this.hoverTarget.value = undefined;
    const logScaleMax = Math.round(
      this.logScaleOrigin +
        numRenderScaleHistogramBins * renderScaleHistogramBinSize,
    );
    const targetValue = clampToInterval(
      [2 ** this.logScaleOrigin, 2 ** (logScaleMax - 1)],
      this.target.value * 2 ** Math.sign(deltaY),
    ) as number;
    this.target.value = targetValue;
    event.preventDefault();
  }

  constructor(
    public histogram: RenderScaleHistogram,
    public target: TrackableValueInterface<number>,
  ) {
    super();
    const {
      canvas,
      label,
      element,
      legend,
      legendRenderScale,
      legendSpatialScale,
      legendChunks,
    } = this;
    label.className = "neuroglancer-render-scale-widget-prompt";
    element.className = "neuroglancer-render-scale-widget";
    element.title = inputEventMap.describe();
    legend.className = "neuroglancer-render-scale-widget-legend";
    element.appendChild(label);
    element.appendChild(canvas);
    element.appendChild(legend);
    legendRenderScale.title = "Target resolution of data in screen pixels";
    legendChunks.title = "Number of chunks rendered";
    legend.appendChild(legendRenderScale);
    legend.appendChild(legendChunks);
    legend.appendChild(legendSpatialScale);
    this.registerDisposer(histogram.changed.add(this.throttledUpdateView));
    this.registerDisposer(
      histogram.visibility.changed.add(this.debouncedUpdateView),
    );
    this.registerDisposer(target.changed.add(this.debouncedUpdateView));
    this.registerDisposer(new MouseEventBinder(canvas, inputEventMap));
    this.registerDisposer(target.changed.add(this.debouncedUpdateView));
    this.registerDisposer(
      this.hoverTarget.changed.add(this.debouncedUpdateView),
    );

    const getTargetValue = (event: MouseEvent) => {
      const position =
        (event.offsetX / canvas.width) * numRenderScaleHistogramBins;
      return getRenderScaleFromHistogramOffset(position, this.logScaleOrigin);
    };
    this.registerEventListener(canvas, "pointermove", (event: MouseEvent) => {
      this.hoverTarget.value = [getTargetValue(event), event.offsetY];
    });

    this.registerEventListener(canvas, "pointerleave", () => {
      this.hoverTarget.value = undefined;
    });

    this.registerDisposer(
      registerActionListener<MouseEvent>(canvas, "set", (actionEvent) => {
        this.target.value = getTargetValue(actionEvent.detail);
      }),
    );

    this.registerDisposer(
      registerActionListener<WheelEvent>(
        canvas,
        "adjust-via-wheel",
        (actionEvent) => {
          this.adjustViaWheel(actionEvent.detail);
        },
      ),
    );

    this.registerDisposer(
      registerActionListener(canvas, "reset", (event) => {
        this.reset();
        event.preventDefault();
      }),
    );
    const resizeObserver = new ResizeObserver(() => this.debouncedUpdateView());
    resizeObserver.observe(canvas);
    this.registerDisposer(() => resizeObserver.disconnect());
    this.updateView();
  }

  getWheelMoveValue(event: WheelEvent) {
    return event.deltaY;
  }

  reset() {
    this.hoverTarget.value = undefined;
    this.target.reset();
  }

  updateView() {
    const { ctx } = this;
    const { canvas } = this;
    const width = (canvas.width = canvas.offsetWidth);
    const height = (canvas.height = canvas.offsetHeight);
    const targetValue = this.target.value;
    const hoverValue = this.hoverTarget.value;

    {
      const { legendRenderScale } = this;
      const value = hoverValue === undefined ? targetValue : hoverValue[0];
      const valueString = formatPixelNumber(value);
      legendRenderScale.textContent = valueString + " " + this.unitOfTarget;
    }

    function binToCanvasX(bin: number) {
      return (bin * width) / numRenderScaleHistogramBins;
    }

    ctx.clearRect(0, 0, width, height);

    const { histogram } = this;
    // histogram.begin(this.frameNumberCounter.frameNumber);
    const { value: histogramData, spatialScales } = histogram;

    if (!histogram.visibility.visible) {
      histogramData.fill(0);
    }

    const sortedSpatialScales = Array.from(spatialScales.keys());
    sortedSpatialScales.sort();

    const tempColor = vec3.create();

    let maxCount = 1;
    const numRows = spatialScales.size;
    let totalPresent = 0;
    let totalNotPresent = 0;
    for (let bin = 0; bin < numRenderScaleHistogramBins; ++bin) {
      let count = 0;
      for (let row = 0; row < numRows; ++row) {
        const index = row * numRenderScaleHistogramBins * 2 + bin;
        const presentCount = histogramData[index];
        const notPresentCount =
          histogramData[index + numRenderScaleHistogramBins];
        totalPresent += presentCount;
        totalNotPresent += notPresentCount;
        count += presentCount + notPresentCount;
      }
      maxCount = Math.max(count, maxCount);
    }
    totalNotPresent -= histogram.fakeChunkCount;

    const maxBarHeight = height;

    const yScale = maxBarHeight / Math.log(1 + maxCount);

    function countToCanvasY(count: number) {
      return height - Math.log(1 + count) * yScale;
    }

    let hoverSpatialScale: number | undefined = undefined;
    if (hoverValue !== undefined) {
      const i = Math.floor(
        getRenderScaleHistogramOffset(hoverValue[0], this.logScaleOrigin),
      );
      if (i >= 0 && i < numRenderScaleHistogramBins) {
        let sum = 0;
        const hoverY = hoverValue[1];
        for (
          let spatialScaleIndex = numRows - 1;
          spatialScaleIndex >= 0;
          --spatialScaleIndex
        ) {
          const spatialScale = sortedSpatialScales[spatialScaleIndex];
          const row = spatialScales.get(spatialScale)!;
          const index = 2 * row * numRenderScaleHistogramBins + i;
          const count =
            histogramData[index] +
            histogramData[index + numRenderScaleHistogramBins];
          if (count === 0) continue;
          const yStart = Math.round(countToCanvasY(sum));
          sum += count;
          const yEnd = Math.round(countToCanvasY(sum));
          if (yEnd <= hoverY && hoverY <= yStart) {
            hoverSpatialScale = spatialScale;
            break;
          }
        }
      }
    }
    if (hoverSpatialScale !== undefined) {
      totalPresent = 0;
      totalNotPresent = 0;
      const row = spatialScales.get(hoverSpatialScale)!;
      const baseIndex = 2 * row * numRenderScaleHistogramBins;
      for (let bin = 0; bin < numRenderScaleHistogramBins; ++bin) {
        const index = baseIndex + bin;
        totalPresent += histogramData[index];
        totalNotPresent += histogramData[index + numRenderScaleHistogramBins];
      }
      if (Number.isFinite(hoverSpatialScale)) {
        this.legendSpatialScale.textContent = formatScaleWithUnitAsString(
          hoverSpatialScale,
          "m",
          { precision: 2, elide1: false },
        );
      } else {
        this.legendSpatialScale.textContent = "unknown";
      }
    } else {
      this.legendSpatialScale.textContent = "";
    }

    this.legendChunks.textContent = `${totalPresent}/${
      totalPresent + totalNotPresent
    }`;

    const spatialScaleColors = sortedSpatialScales.map((spatialScale) => {
      const saturation = spatialScale === hoverSpatialScale ? 0.5 : 1;
      let hue;
      if (Number.isFinite(spatialScale)) {
        hue = (((Math.log2(spatialScale) * 0.1) % 1) + 1) % 1;
      } else {
        hue = 0;
      }
      hsvToRgb(tempColor, hue, saturation, 1);
      const presentColor = serializeColor(tempColor);
      hsvToRgb(tempColor, hue, saturation, 0.5);
      const notPresentColor = serializeColor(tempColor);
      return [presentColor, notPresentColor];
    });

    for (let i = 0; i < numRenderScaleHistogramBins; ++i) {
      let sum = 0;
      for (
        let spatialScaleIndex = numRows - 1;
        spatialScaleIndex >= 0;
        --spatialScaleIndex
      ) {
        const spatialScale = sortedSpatialScales[spatialScaleIndex];
        const row = spatialScales.get(spatialScale)!;
        const index = row * numRenderScaleHistogramBins * 2 + i;
        const presentCount = histogramData[index];
        const notPresentCount =
          histogramData[index + numRenderScaleHistogramBins];
        const count = presentCount + notPresentCount;
        if (count === 0) continue;
        const xStart = Math.round(binToCanvasX(i));
        const xEnd = Math.round(binToCanvasX(i + 1));
        const yStart = Math.round(countToCanvasY(sum));
        sum += count;
        const yEnd = Math.round(countToCanvasY(sum));
        const ySplit = (presentCount * yEnd + notPresentCount * yStart) / count;
        ctx.fillStyle = spatialScaleColors[spatialScaleIndex][1];
        ctx.fillRect(xStart, yEnd, xEnd - xStart, ySplit - yEnd);
        ctx.fillStyle = spatialScaleColors[spatialScaleIndex][0];
        ctx.fillRect(xStart, ySplit, xEnd - xStart, yStart - ySplit);
      }
    }

    {
      const value = targetValue;
      ctx.fillStyle = "#fff";
      const startOffset = binToCanvasX(
        getRenderScaleHistogramOffset(value, this.logScaleOrigin),
      );
      const lineWidth = 1;
      ctx.fillRect(Math.floor(startOffset), 0, lineWidth, height);
    }

    if (hoverValue !== undefined) {
      const value = hoverValue[0];
      ctx.fillStyle = "#888";
      const startOffset = binToCanvasX(
        getRenderScaleHistogramOffset(value, this.logScaleOrigin),
      );
      const lineWidth = 1;
      ctx.fillRect(Math.floor(startOffset), 0, lineWidth, height);
    }
  }
}

export class VolumeRenderingRenderScaleWidget extends RenderScaleWidget {
  protected unitOfTarget = "samples";
  protected logScaleOrigin = 1;

  getWheelMoveValue(event: WheelEvent) {
    return -event.deltaY;
  }
}

const gridInputEventMap = EventActionMap.fromObject({
  mousedown0: { action: "set" },
  wheel: { action: "adjust-via-wheel" },
  dblclick0: { action: "reset" },
});

export class SpatialSkeletonGridResolutionWidget extends RefCounted {
  label = document.createElement("div");
  element = document.createElement("div");
  canvas = document.createElement("canvas");
  legend = document.createElement("div");
  legendTarget = document.createElement("div");
  legendGrid = document.createElement("div");
  legendLod = document.createElement("div");
  hoverSpacing = new WatchableValue<number | undefined>(undefined);
  private ctx = this.canvas.getContext("2d")!;
  private throttledUpdateView = this.registerCancellable(
    throttle(() => this.debouncedUpdateView(), updateInterval, {
      leading: true,
      trailing: true,
    }),
  );
  private debouncedUpdateView = this.registerCancellable(
    debounce(() => this.updateView(), 0),
  );

  constructor(
    public levels: WatchableValueInterface<SpatialSkeletonGridLevel[]>,
    public target: TrackableValueInterface<number>,
    public relative: WatchableValueInterface<boolean>,
    public pixelSize: WatchableValueInterface<number>,
    options: {
      relativeLabel?: string;
      relativeTooltip?: string;
    } = {},
  ) {
    super();
    const {
      canvas,
      label,
      element,
      legend,
      legendTarget,
      legendGrid,
      legendLod,
    } = this;
    label.className = "neuroglancer-render-scale-widget-prompt";
    element.className = "neuroglancer-render-scale-widget";
    element.classList.add("neuroglancer-render-scale-widget-grid");
    element.title = gridInputEventMap.describe();
    legend.className = "neuroglancer-render-scale-widget-legend";
    element.appendChild(label);
    element.appendChild(canvas);
    element.appendChild(legend);
    const relativeTooltip =
      options.relativeTooltip ??
      "Interpret the skeleton grid resolution target as relative to zoom";
    label.classList.add("neuroglancer-render-scale-widget-relative");
    label.title = relativeTooltip;
    const relativeCheckbox = this.registerDisposer(
      new TrackableBooleanCheckbox(relative, {
        enabledTitle: relativeTooltip,
        disabledTitle: relativeTooltip,
      }),
    );
    relativeCheckbox.element.classList.add(
      "neuroglancer-render-scale-widget-relative-checkbox",
    );
    label.appendChild(relativeCheckbox.element);
    const relativeLabel = document.createElement("span");
    relativeLabel.textContent = options.relativeLabel ?? "Rel";
    label.appendChild(relativeLabel);
    legend.appendChild(legendTarget);
    legend.appendChild(legendGrid);
    legend.appendChild(legendLod);
    this.registerDisposer(levels.changed.add(this.throttledUpdateView));
    this.registerDisposer(target.changed.add(this.debouncedUpdateView));
    this.registerDisposer(relative.changed.add(this.debouncedUpdateView));
    this.registerDisposer(pixelSize.changed.add(this.debouncedUpdateView));
    this.registerDisposer(
      this.hoverSpacing.changed.add(this.debouncedUpdateView),
    );
    this.registerDisposer(new MouseEventBinder(canvas, gridInputEventMap));
    this.registerEventListener(element, "click", (event: MouseEvent) => {
      if (event.target === relativeCheckbox.element) {
        return;
      }
      // Prevent the layer-control <label> from toggling the relative checkbox
      // when interacting with the widget outside the checkbox itself.
      event.preventDefault();
    });

    const getSpacingFromEvent = (event: MouseEvent) => {
      const position =
        (event.offsetX / canvas.width) * numRenderScaleHistogramBins;
      return getRenderScaleFromHistogramOffset(
        position,
        renderScaleHistogramOrigin,
      );
    };

    this.registerEventListener(canvas, "pointermove", (event: MouseEvent) => {
      if (this.levels.value.length === 0) {
        this.hoverSpacing.value = undefined;
        return;
      }
      this.hoverSpacing.value = getSpacingFromEvent(event);
    });

    this.registerEventListener(canvas, "pointerleave", () => {
      this.hoverSpacing.value = undefined;
    });

    this.registerDisposer(
      registerActionListener<MouseEvent>(canvas, "set", (actionEvent) => {
        const spacing = getSpacingFromEvent(actionEvent.detail);
        const pixelSize = Math.max(this.pixelSize.value, 1e-6);
        const value = this.relative.value ? spacing / pixelSize : spacing;
        this.target.value = value;
      }),
    );

    this.registerDisposer(
      registerActionListener<WheelEvent>(
        canvas,
        "adjust-via-wheel",
        (actionEvent) => {
          this.adjustViaWheel(actionEvent.detail);
        },
      ),
    );

    this.registerDisposer(
      registerActionListener(canvas, "reset", (event) => {
        this.reset();
        event.preventDefault();
      }),
    );

    const resizeObserver = new ResizeObserver(() => this.debouncedUpdateView());
    resizeObserver.observe(canvas);
    this.registerDisposer(() => resizeObserver.disconnect());
    this.updateView();
  }

  adjustViaWheel(event: WheelEvent) {
    const delta = Math.sign(event.deltaY);
    if (delta === 0) return;
    const current = this.target.value;
    const next = Math.max(current * 2 ** delta, 1e-6);
    this.target.value = next;
    event.preventDefault();
  }

  reset() {
    this.hoverSpacing.value = undefined;
    const target = this.target as TrackableValueInterface<number> & {
      reset?: () => void;
    };
    if (typeof target.reset === "function") {
      target.reset();
    } else {
      this.target.value = 0;
    }
  }

  updateView() {
    const { ctx, canvas } = this;
    const width = (canvas.width = canvas.offsetWidth);
    const height = (canvas.height = canvas.offsetHeight);
    ctx.clearRect(0, 0, width, height);

    const levels = this.levels.value;
    const levelCount = levels.length;
    if (levelCount === 0) {
      this.legendTarget.textContent = "target -";
      this.legendGrid.textContent = "grid -";
      this.legendLod.textContent = "lod -";
      this.legendGrid.title = "";
      return;
    }

    const pixelSize = Math.max(this.pixelSize.value, 1e-6);
    const targetValue = this.target.value;
    const effectiveTargetSpacing = Math.max(
      this.relative.value ? targetValue * pixelSize : targetValue,
      1e-6,
    );
    const displaySpacing = this.hoverSpacing.value ?? effectiveTargetSpacing;
    let nearestIndex = 0;
    let nearestDistance = Number.POSITIVE_INFINITY;
    for (let i = 0; i < levelCount; ++i) {
      const spacing = Math.min(
        levels[i].size.x,
        levels[i].size.y,
        levels[i].size.z,
      );
      const distance = Math.abs(spacing - displaySpacing);
      if (distance < nearestDistance) {
        nearestDistance = distance;
        nearestIndex = i;
      }
    }
    const level = levels[nearestIndex];
    const unitLabel = this.relative.value ? "px" : "vx";
    this.legendTarget.textContent = `target ${numberToStringFixed(
      targetValue,
      2,
    )} ${unitLabel}`;
    this.legendGrid.textContent = `grid ${nearestIndex + 1}/${levelCount}`;
    const roundedSize = [
      Math.round(level.size.x),
      Math.round(level.size.y),
      Math.round(level.size.z),
    ];
    const sizeLabel =
      roundedSize[0] === roundedSize[1] && roundedSize[1] === roundedSize[2]
        ? `${roundedSize[0]}^3`
        : `${roundedSize[0]}x${roundedSize[1]}x${roundedSize[2]}`;
    this.legendGrid.title = `grid size ${sizeLabel}`;
    this.legendLod.textContent = `lod ${level.lod.toFixed(2)}`;

    const tempColor = vec3.create();
    const binToCanvasX = (bin: number) =>
      (bin * width) / numRenderScaleHistogramBins;
    const barTop = Math.round(height * 0.1);
    const barHeight = Math.max(4, height - barTop * 2);

    const entries = levels
      .map((level, index) => {
        const spacing = Math.max(
          Math.min(level.size.x, level.size.y, level.size.z),
          1e-6,
        );
        const offset = getRenderScaleHistogramOffset(
          spacing,
          renderScaleHistogramOrigin,
        );
        return { index, spacing, offset };
      })
      .sort((a, b) => a.offset - b.offset);
    const boundaries = new Array(entries.length + 1);
    boundaries[0] = 0;
    for (let i = 1; i < entries.length; ++i) {
      boundaries[i] = (entries[i - 1].offset + entries[i].offset) / 2;
    }
    boundaries[entries.length] = numRenderScaleHistogramBins;

    const hoverSpacing = this.hoverSpacing.value;
    let hoverIndex: number | undefined = undefined;
    if (hoverSpacing !== undefined) {
      let bestDistance = Number.POSITIVE_INFINITY;
      for (let i = 0; i < levelCount; ++i) {
        const spacing = Math.min(
          levels[i].size.x,
          levels[i].size.y,
          levels[i].size.z,
        );
        const distance = Math.abs(spacing - hoverSpacing);
        if (distance < bestDistance) {
          bestDistance = distance;
          hoverIndex = i;
        }
      }
    }
    for (let i = 0; i < entries.length; ++i) {
      const { index, spacing } = entries[i];
      const saturation = hoverIndex !== undefined && index === hoverIndex ? 0.5 : 1;
      let hue = 0;
      if (Number.isFinite(spacing)) {
        hue = (((Math.log2(spacing) * 0.1) % 1) + 1) % 1;
      }
      hsvToRgb(tempColor, hue, saturation, 1);
      ctx.fillStyle = serializeColor(tempColor);
      const xStart = Math.round(binToCanvasX(boundaries[i]));
      const xEnd = Math.round(binToCanvasX(boundaries[i + 1]));
      ctx.fillRect(xStart, barTop, xEnd - xStart, barHeight);
    }

    const targetOffset = getRenderScaleHistogramOffset(
      effectiveTargetSpacing,
      renderScaleHistogramOrigin,
    );
    {
      const x = Math.round(binToCanvasX(targetOffset));
      ctx.fillStyle = "#fff";
      ctx.fillRect(x, 0, 1, height);
    }
    if (hoverSpacing !== undefined) {
      const hoverOffset = getRenderScaleHistogramOffset(
        hoverSpacing,
        renderScaleHistogramOrigin,
      );
      const x = Math.round(binToCanvasX(hoverOffset));
      ctx.fillStyle = "#888";
      ctx.fillRect(x, 0, 1, height);
    }
  }
}

export class SpatialSkeletonGridRenderScaleWidget extends RenderScaleWidget {
  protected unitOfTarget = "grid";
}

const TOOL_INPUT_EVENT_MAP = EventActionMap.fromObject({
  "at:shift+wheel": { action: "adjust-via-wheel" },
  "at:shift+dblclick0": { action: "reset" },
});

export function renderScaleLayerControl<
  LayerType extends UserLayer,
  WidgetType extends RenderScaleWidget,
>(
  getter: (layer: LayerType) => RenderScaleWidgetOptions,
  widgetClass: new (
    histogram: RenderScaleHistogram,
    target: TrackableValueInterface<number>,
  ) => WidgetType = RenderScaleWidget as new (
    histogram: RenderScaleHistogram,
    target: TrackableValueInterface<number>,
  ) => WidgetType,
): LayerControlFactory<LayerType, RenderScaleWidget> {
  return {
    makeControl: (layer, context) => {
      const { histogram, target } = getter(layer);
      const control = context.registerDisposer(
        new widgetClass(histogram, target),
      );
      return { control, controlElement: control.element };
    },
    activateTool: (activation, control) => {
      activation.bindInputEventMap(TOOL_INPUT_EVENT_MAP);
      activation.bindAction(
        "adjust-via-wheel",
        (event: ActionEvent<WheelEvent>) => {
          event.stopPropagation();
          event.preventDefault();
          control.adjustViaWheel(event.detail);
        },
      );
      activation.bindAction("reset", (event: ActionEvent<WheelEvent>) => {
        event.stopPropagation();
        event.preventDefault();
        control.reset();
      });
    },
  };
}

export function spatialSkeletonGridResolutionLayerControl<
  LayerType extends UserLayer,
>(
  getter: (layer: LayerType) => SpatialSkeletonGridResolutionWidgetOptions,
): LayerControlFactory<LayerType, SpatialSkeletonGridResolutionWidget> {
  return {
    makeControl: (layer, context) => {
      const { levels, target, relative, pixelSize, relativeLabel, relativeTooltip } =
        getter(layer);
      const control = context.registerDisposer(
        new SpatialSkeletonGridResolutionWidget(
          levels,
          target,
          relative,
          pixelSize,
          { relativeLabel, relativeTooltip },
        ),
      );
      return { control, controlElement: control.element };
    },
    activateTool: (activation, control) => {
      activation.bindInputEventMap(TOOL_INPUT_EVENT_MAP);
      activation.bindAction(
        "adjust-via-wheel",
        (event: ActionEvent<WheelEvent>) => {
          event.stopPropagation();
          event.preventDefault();
          control.adjustViaWheel(event.detail);
        },
      );
      activation.bindAction("reset", (event: ActionEvent<WheelEvent>) => {
        event.stopPropagation();
        event.preventDefault();
        control.reset();
      });
    },
  };
}
