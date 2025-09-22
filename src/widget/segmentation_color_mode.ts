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
import type { RefCounted } from "#src/util/disposable.js";
import { vec3 } from "#src/util/geom.js";
import type { ColorWidget } from "#src/widget/color.js";
import { makeIcon } from "#src/widget/icon.js";
import type { LayerControlFactory } from "#src/widget/layer_control.js";
import { colorLayerControl } from "#src/widget/layer_control_color.js";
import { TextInputWidget } from "#src/widget/text_input.js";

// @metacell, the old always went to red on fixed color,
// so we keep a weak ref to the last fixed color used for each layer
// and restore it when switching back to fixed color mode
function chooseColorMode(layer: SegmentationUserLayer, useFixedColor: boolean) {
  if (!useFixedColor) {
    const currentColor = layer.displayState.segmentDefaultColor.value;
    if (currentColor !== undefined) {
      layerFixedColors.set(layer, vec3.clone(currentColor));
    }
    layer.displayState.segmentDefaultColor.value = undefined;
  } else {
    const savedColor = layerFixedColors.get(layer);
    if (savedColor) {
      layer.displayState.segmentDefaultColor.value = vec3.clone(savedColor);
    } else if (layer.displayState.segmentDefaultColor.value === undefined) {
      layer.displayState.segmentDefaultColor.value = vec3.fromValues(1, 0, 0);
    }
  }
}

const layerFixedColors = new WeakMap<SegmentationUserLayer, vec3>();

const createColorModeTabContainer = () => {
  const colorTab = document.createElement("div");
  colorTab.classList.add("neuroglancer-segmentation-color-tab");

  const colorText = document.createElement("span");
  colorText.textContent = "Colours";
  colorTab.appendChild(colorText);

  const buttonContainer = document.createElement("div");
  buttonContainer.classList.add("neuroglancer-segmentation-color-tab-buttons");
  colorTab.appendChild(buttonContainer);

  const randomColorButton = document.createElement("button");
  randomColorButton.classList.add("neuroglancer-segmentation-color-tab-button");
  randomColorButton.classList.add("active");
  randomColorButton.innerHTML = svg_gradient;
  randomColorButton.title = "Seeded random colours";
  buttonContainer.appendChild(randomColorButton);

  const fixedColorButton = document.createElement("button");
  fixedColorButton.classList.add("neuroglancer-segmentation-color-tab-button");
  fixedColorButton.innerHTML = svg_format_color_fill;
  fixedColorButton.title = "Fixed color";
  buttonContainer.appendChild(fixedColorButton);

  const colorControlsContainer = document.createElement("div");
  colorControlsContainer.classList.add(
    "neuroglancer-segmentation-color-controls",
  );

  return {
    colorTab,
    randomColorButton,
    fixedColorButton,
    colorControlsContainer,
  };
};

export function createColorModeTabsWithControls(
  layer: SegmentationUserLayer,
  context: RefCounted,
) {
  const {
    colorTab,
    randomColorButton,
    fixedColorButton,
    colorControlsContainer,
  } = createColorModeTabContainer();

  let seedControlContainer: HTMLElement | null = null;
  let fixedColorControlContainer: HTMLElement | null = null;

  const updateTabState = (isRandomColor: boolean) => {
    randomColorButton.classList.toggle("active", isRandomColor);
    fixedColorButton.classList.toggle("active", !isRandomColor);

    // Control visibility of the entire control containers
    if (seedControlContainer) {
      seedControlContainer.style.display = isRandomColor ? "" : "none";
    }
    if (fixedColorControlContainer) {
      fixedColorControlContainer.style.display = isRandomColor ? "none" : "";
    }
  };

  const showControls = (isRandomColor: boolean) => {
    chooseColorMode(layer, !isRandomColor);
  };

  randomColorButton.addEventListener("click", () => showControls(true));
  fixedColorButton.addEventListener("click", () => showControls(false));

  context.registerDisposer(
    observeWatchable((value) => {
      const isRandomColor = value === undefined;
      updateTabState(isRandomColor);

      if (!isRandomColor && value !== undefined) {
        layerFixedColors.set(layer, vec3.clone(value));
      }
    }, layer.displayState.segmentDefaultColor),
  );

  const initialIsRandomColor =
    layer.displayState.segmentDefaultColor.value === undefined;
  updateTabState(initialIsRandomColor);

  function registerColorControl(element: HTMLElement, toolJson: string) {
    if (toolJson === "colorSeed") {
      seedControlContainer = element;
    } else if (toolJson === "segmentDefaultColor") {
      fixedColorControlContainer = element;
    } else {
      return;
    }
    colorControlsContainer.appendChild(element);
    updateTabState(layer.displayState.segmentDefaultColor.value === undefined);
  }

  return {
    colorTab,
    colorControlsContainer,
    registerColorControl,
  };
}

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
      // Remove individual visibility control - now handled by tabs
      return { controlElement, control: widget };
    },
    activateTool: (activation) => {
      const { layer } = activation.tool;
      chooseColorMode(layer, false);
      randomize(layer);
    },
  };
}

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
