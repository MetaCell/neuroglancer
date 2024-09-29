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

import type { SegmentationUserLayer } from "#src/layer/segmentation/index.js";
import { SKELETON_RENDERING_SHADER_CONTROL_TOOL_ID } from "#src/layer/segmentation/json_keys.js";
import { VISIBILITY_LAYER_CONTROLS, APPEARANCE_LAYER_CONTROLS, SLICE2D_LAYER_CONTROLS, MESH_LAYER_CONTROLS, SKELETONS_LAYER_CONTROLS } from "#src/layer/segmentation/layer_controls.js";
import { Overlay } from "#src/overlay.js";
import type { AccordionItem } from "#src/widget/accordion.js";
import { Accordion } from "#src/widget/accordion.js";
import { DependentViewWidget } from "#src/widget/dependent_view_widget.js";
import { makeHelpButton } from "#src/widget/help_button.js";
import { addLayerControlToOptionsTab } from "#src/widget/layer_control.js";
import { LinkedLayerGroupWidget } from "#src/widget/linked_layer.js";
import { makeMaximizeButton } from "#src/widget/maximize_button.js";
import { ShaderCodeWidget } from "#src/widget/shader_code_widget.js";
import { ShaderControls } from "#src/widget/shader_controls.js";
import { Tab } from "#src/widget/tab_view.js";

function makeSkeletonShaderCodeWidget(layer: SegmentationUserLayer) {
  return new ShaderCodeWidget({
    fragmentMain: layer.displayState.skeletonRenderingOptions.shader,
    shaderError: layer.displayState.shaderError,
    shaderControlState:
      layer.displayState.skeletonRenderingOptions.shaderControlState,
  });
}

export class DisplayOptionsTab extends Tab {
  constructor(public layer: SegmentationUserLayer) {
    super();
    const { element } = this;
    element.classList.add("neuroglancer-segmentation-rendering-tab")

    // Create the accordion and add the items
    const visibilityAccordion = this.createVisibilityAccordion();
    const appearanceAccordion = this.createAppearanceAccordion();
    const slice2DAccordion = this.createSlice2DAccordion();
    const mesh3DAccordion = this.createMesh3DAccordion();
    const channelsAccordion = this.createChannelsAccordion();
    const skeletonAccordion = this.createSkeletonAccordion();

    const accordion = new Accordion([
      visibilityAccordion,
      skeletonAccordion,
      appearanceAccordion,
      slice2DAccordion,
      mesh3DAccordion,
      channelsAccordion,
    ]);

    // Append the accordion to the element
    element.appendChild(accordion.getElement());
  }

  // Create Slice 2D Accordion
  private createVisibilityAccordion(): AccordionItem {
    const containerDiv = document.createElement("div");
    containerDiv.className = "visibility-container";

    for (const control of VISIBILITY_LAYER_CONTROLS) {
      containerDiv.appendChild(
        addLayerControlToOptionsTab(this, this.layer, this.visibility, control),
      );
    }
    
    const widget = this.registerDisposer(
      new LinkedLayerGroupWidget(this.layer.displayState.linkedSegmentationGroup),
    );
    widget.label.textContent = "Linked to: ";
    containerDiv.appendChild(widget.element);
    
    return {
      title: "Visibility",
      content: containerDiv,
    };
  }

  // Create the Appearance Accordion section
  private createAppearanceAccordion(): AccordionItem {
    const containerDiv = document.createElement("div");
    containerDiv.className = "appearance-container";

    for (const control of APPEARANCE_LAYER_CONTROLS) {
      containerDiv.appendChild(
        addLayerControlToOptionsTab(this, this.layer, this.visibility, control),
      );
    }
    // Linked segmentation control
    const widget = this.registerDisposer(
      new LinkedLayerGroupWidget(
        this.layer.displayState.linkedSegmentationColorGroup,
      ),
    );
    widget.label.textContent = "Colors linked to: ";
    containerDiv.appendChild(widget.element);
    
    return {
      title: "Appearance",
      content: containerDiv,
    };
  }

  // Create the Slice 2D Accordion section
  private createSlice2DAccordion(): AccordionItem {
    const containerDiv = document.createElement("div");
    containerDiv.className = "slice2d-container";

    for (const control of SLICE2D_LAYER_CONTROLS) {
      containerDiv.appendChild(
        addLayerControlToOptionsTab(this, this.layer, this.visibility, control),
      );
    }

    return {
      title: "Slice 2D",
      content: containerDiv,
    };
  }

  // Create the Mesh 3D Accordion section
  private createMesh3DAccordion(): AccordionItem {
    const containerDiv = document.createElement("div");
    containerDiv.className = "mesh3d-container";

    for (const control of MESH_LAYER_CONTROLS) {
      containerDiv.appendChild(
        addLayerControlToOptionsTab(this, this.layer, this.visibility, control),
      );
    }

    return {
      title: "Mesh 3D",
      content: containerDiv,
    };
  }

  // Create the Skeleton 3D Accordion section
  private createSkeletonAccordion(): AccordionItem {
    const containerDiv = document.createElement("div");
    containerDiv.className = "skeleton-container";

    for (const control of SKELETONS_LAYER_CONTROLS) {
      containerDiv.appendChild(
        addLayerControlToOptionsTab(this, this.layer, this.visibility, control),
      );
    }

    return {
      title: "Skeletons",
      content: containerDiv,
    };
  }

  private createChannelsAccordion(): AccordionItem {
    const containerDiv = document.createElement("div");
    containerDiv.className = "shader-container";
    const skeletonControls = this.registerDisposer(
      new DependentViewWidget(
        this.layer.hasSkeletonsLayer,
        (hasSkeletonsLayer, parent, refCounted) => {
          if (!hasSkeletonsLayer) return;
          const topRow = document.createElement("div");
          topRow.className =
            "neuroglancer-segmentation-dropdown-skeleton-shader-header";
          const label = document.createElement("div");
          label.style.flex = "1";
          label.textContent = "Skeleton shader:";
          topRow.appendChild(label);
          topRow.appendChild(
            makeMaximizeButton({
              title: "Show larger editor view",
              onClick: () => {
                new ShaderCodeOverlay(this.layer);
              },
            }),
          );
          topRow.appendChild(
            makeHelpButton({
              title: "Documentation on skeleton rendering",
              href: "https://github.com/google/neuroglancer/blob/master/src/sliceview/image_layer_rendering.md",
            }),
          );
          parent.appendChild(topRow);

          const codeWidget = refCounted.registerDisposer(
            makeSkeletonShaderCodeWidget(this.layer),
          );
          parent.appendChild(codeWidget.element);
          parent.appendChild(
            refCounted.registerDisposer(
              new ShaderControls(
                this.layer.displayState.skeletonRenderingOptions.shaderControlState,
                this.layer.manager.root.display,
                this.layer,
                {
                  visibility: this.visibility,
                  toolId: SKELETON_RENDERING_SHADER_CONTROL_TOOL_ID,
                },
              ),
            ).element,
          );
          codeWidget.textEditor.refresh();
        },
        this.visibility,
      ),
    );
    containerDiv.appendChild(skeletonControls.element)
    return {
      title: "Channels",
      content: containerDiv,
    };
  }
}

class ShaderCodeOverlay extends Overlay {
  codeWidget = this.registerDisposer(makeSkeletonShaderCodeWidget(this.layer));
  constructor(public layer: SegmentationUserLayer) {
    super();
    this.content.classList.add(
      "neuroglancer-segmentation-layer-skeleton-shader-overlay",
    );
    this.content.appendChild(this.codeWidget.element);
    this.codeWidget.textEditor.refresh();
  }
}
