/**
 * @license
 * Copyright 2016 Google Inc.
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

import "#src/widget/segmentation_color_mode.css";

import svg_rotate from "ikonate/icons/rotate.svg?raw";
import type { SegmentationUserLayer } from "#src/layer/segmentation/index.js";
import { observeWatchable } from "#src/trackable_value.js";
import svg_format_color_fill from "#src/ui/images/format_color_fill.svg?raw";
import svg_gradient from "#src/ui/images/gradient.svg?raw";
import { RefCounted } from "#src/util/disposable.js";
import { vec3 } from "#src/util/geom.js";
import type { ColorWidget } from "#src/widget/color.js";
import { makeIcon } from "#src/widget/icon.js";
import type { LayerControlFactory } from "#src/widget/layer_control.js";
import { colorLayerControl } from "#src/widget/layer_control_color.js";
import { TextInputWidget } from "#src/widget/text_input.js";

const layerFixedColors = new WeakMap<SegmentationUserLayer, vec3>();

// @metacell, the old always went to red on fixed color,
// so we keep a ref to the last fixed color used for each layer
// and restore it when switching back to fixed color mode
function restoreLayerFixedColor(layer: SegmentationUserLayer) {
  const savedColor = layerFixedColors.get(layer);
  if (savedColor) {
    layer.displayState.segmentDefaultColor.value = savedColor;
    return true;
  }
  return false;
}

function chooseColorMode(layer: SegmentationUserLayer, useFixedColor: boolean) {
  if (!useFixedColor) {
    layer.displayState.segmentDefaultColor.value = undefined;
  } else {
    if (restoreLayerFixedColor(layer)) return;
    layer.displayState.segmentDefaultColor.value = vec3.fromValues(1, 0, 0);
  }
}

export class SegmentationColorModeWidget extends RefCounted {
  element = document.createElement("div");
  colorText = document.createElement("span");
  buttonContainer = document.createElement("div");
  header = document.createElement("div");
  randomColorButton = document.createElement("button");
  fixedColorButton = document.createElement("button");
  colorControlsContainer = document.createElement("div");
  seedControlContainer: HTMLElement | null = null;
  fixedColorControlContainer: HTMLElement | null = null;
  constructor(public layer: SegmentationUserLayer) {
    super();
    const {
      element,
      colorText,
      buttonContainer,
      header,
      randomColorButton,
      fixedColorButton,
      colorControlsContainer,
    } = this;
    element.classList.add("neuroglancer-segmentation-color-tab");
    colorText.textContent = "Colours";
    buttonContainer.classList.add(
      "neuroglancer-segmentation-color-tab-buttons",
    );
    randomColorButton.classList.add(
      "neuroglancer-segmentation-color-tab-button",
    );
    header.classList.add("neuroglancer-segmentation-color-tab-header");
    randomColorButton.classList.add("active");
    randomColorButton.innerHTML = svg_gradient;
    randomColorButton.title = "Seeded random colours";
    fixedColorButton.classList.add(
      "neuroglancer-segmentation-color-tab-button",
    );
    fixedColorButton.innerHTML = svg_format_color_fill;
    fixedColorButton.title = "Fixed color";
    colorControlsContainer.classList.add(
      "neuroglancer-segmentation-color-controls",
    );

    header.appendChild(colorText);
    header.appendChild(buttonContainer);
    buttonContainer.appendChild(fixedColorButton);
    buttonContainer.appendChild(randomColorButton);
    element.appendChild(header);
    element.appendChild(colorControlsContainer);

    this.registerEventListener(randomColorButton, "click", () => {
      chooseColorMode(layer, false /* fixed color */);
    });
    this.registerEventListener(fixedColorButton, "click", () => {
      chooseColorMode(layer, true /* fixed color */);
    });
    this.registerDisposer(
      observeWatchable((value) => {
        const isRandomColor = value === undefined;
        this.updateTabState(isRandomColor);

        if (!isRandomColor) {
          layerFixedColors.set(layer, vec3.clone(value));
        }
      }, layer.displayState.segmentDefaultColor),
    );
  }

  registerColorControl(element: HTMLElement, toolJson: string) {
    if (toolJson === "colorSeed") {
      this.seedControlContainer = element;
    } else if (toolJson === "segmentDefaultColor") {
      this.fixedColorControlContainer = element;
    } else {
      return;
    }
    this.colorControlsContainer.appendChild(element);
    this.updateTabState();
  }

  updateTabState(isRandomColor?: boolean) {
    if (isRandomColor === undefined) {
      isRandomColor =
        this.layer.displayState.segmentDefaultColor.value === undefined;
    }
    this.randomColorButton.classList.toggle("active", isRandomColor);
    this.fixedColorButton.classList.toggle("active", !isRandomColor);

    // Control visibility of the entire control containers
    if (this.seedControlContainer) {
      this.seedControlContainer.style.display = isRandomColor ? "" : "none";
    }
    if (this.fixedColorControlContainer) {
      this.fixedColorControlContainer.style.display = isRandomColor
        ? "none"
        : "";
    }
  }
}

// @metacell, no longer watches the layer values, instead SegmentationColorModeWidget does that
export function colorSeedLayerControl(): LayerControlFactory<SegmentationUserLayer> {
  const randomize = (layer: SegmentationUserLayer) => {
    layer.displayState.segmentationColorGroupState.value.segmentColorHash.randomize();
  };
  return {
    makeControl: (layer, context) => {
      const controlElement = document.createElement("div");
      controlElement.classList.add(
        "neuroglancer-segmentation-color-seed-control",
      );
      const widget = context.registerDisposer(
        new TextInputWidget(layer.displayState.segmentColorHash),
      );
      controlElement.appendChild(widget.element);
      const randomizeButton = makeIcon({
        svg: svg_rotate,
        title: "Randomize",
        onClick: () => randomize(layer),
        className: "ikonate",
      });
      controlElement.appendChild(randomizeButton);
      return { controlElement, control: widget };
    },
    activateTool: (activation) => {
      const { layer } = activation.tool;
      chooseColorMode(layer, false);
      randomize(layer);
    },
  };
}

// @metacell, no longer watches the layer values, instead SegmentationColorModeWidget does that
export function fixedColorLayerControl(): LayerControlFactory<
  SegmentationUserLayer,
  ColorWidget<vec3 | undefined>
> {
  const options = colorLayerControl(
    (layer: SegmentationUserLayer) => layer.displayState.segmentDefaultColor,
  );
  return {
    ...options,
    makeControl: (layer, context, labelElements) => {
      const result = options.makeControl(layer, context, labelElements);
      return result;
    },
    activateTool: (activation, control) => {
      chooseColorMode(activation.tool.layer, true);
      options.activateTool(activation, control);
    },
  };
}
