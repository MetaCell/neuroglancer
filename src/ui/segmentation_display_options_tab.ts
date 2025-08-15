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

import { type SegmentationUserLayer } from "#src/layer/segmentation/index.js";
import { SKELETON_RENDERING_SHADER_CONTROL_TOOL_ID } from "#src/layer/segmentation/json_keys.js";
import {
  APPEARANCE_SECTION_JSON_KEY,
  LAYER_CONTROLS,
  VISIBILITY_SECTION_JSON_KEY,
  SKELETON_SECTION_JSON_KEY,
} from "#src/layer/segmentation/layer_controls.js";
import { AccordionTab } from "#src/widget/accordion.js";
import { DependentViewWidget } from "#src/widget/dependent_view_widget.js";
import { addLayerControlToOptionsTab } from "#src/widget/layer_control.js";
import { LinkedLayerGroupWidget } from "#src/widget/linked_layer.js";
import {
  makeShaderCodeWidgetTopRow,
  ShaderCodeWidget,
} from "#src/widget/shader_code_widget.js";
import { ShaderControls } from "#src/widget/shader_controls.js";

function makeSkeletonShaderCodeWidget(layer: SegmentationUserLayer) {
  return new ShaderCodeWidget({
    fragmentMain: layer.displayState.skeletonRenderingOptions.shader,
    shaderError: layer.displayState.shaderError,
    shaderControlState:
      layer.displayState.skeletonRenderingOptions.shaderControlState,
  });
}

export class DisplayOptionsTab extends AccordionTab {
  constructor(public layer: SegmentationUserLayer) {
    super(layer.renderingAccordionState);
    const { element } = this;
    element.classList.add("neuroglancer-segmentation-rendering-tab");

    // Linked segmentation control
    {
      const widget = this.registerDisposer(
        new LinkedLayerGroupWidget(layer.displayState.linkedSegmentationGroup),
      );
      widget.label.textContent = "Linked to: ";
      widget.label.classList.add("neuroglancer-label-linkedto");
      this.appendChild(widget.element, VISIBILITY_SECTION_JSON_KEY);
    }

    // Linked segmentation control
    {
      const widget = this.registerDisposer(
        new LinkedLayerGroupWidget(
          layer.displayState.linkedSegmentationColorGroup,
        ),
      );
      widget.label.textContent = "Colors linked to: ";
      widget.label.classList.add("neuroglancer-label-linkedto");
      this.appendChild(widget.element, APPEARANCE_SECTION_JSON_KEY);
    }

    for (const control of LAYER_CONTROLS) {
      const e = addLayerControlToOptionsTab(
        this,
        layer,
        this.visibility,
        control,
      );
      e.id = control.toolJson;
      this.appendChild(e, control.sectionKey);
    }

    const skeletonControls = this.registerDisposer(
      new DependentViewWidget(
        layer.hasSkeletonsLayer,
        (hasSkeletonsLayer, parent, refCounted) => {
          if (!hasSkeletonsLayer) return;
          const codeWidget = refCounted.registerDisposer(
            makeSkeletonShaderCodeWidget(this.layer),
          );
          parent.appendChild(
            makeShaderCodeWidgetTopRow(
              this.layer,
              codeWidget.element,
              makeSkeletonShaderCodeWidget,
              {
                title: "Documentation on skeleton rendering",
                href: "https://github.com/google/neuroglancer/blob/master/src/sliceview/image_layer_rendering.md",
                type: "Skeleton",
              },
            ),
          );
          parent.appendChild(codeWidget.element);
          parent.appendChild(
            refCounted.registerDisposer(
              new ShaderControls(
                layer.displayState.skeletonRenderingOptions.shaderControlState,
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
    this.appendChild(
      skeletonControls.element,
      SKELETON_SECTION_JSON_KEY,
      !this.layer.hasSkeletonsLayer.value,
    );
    this.registerDisposer(
      this.layer.hasSkeletonsLayer.changed.add(() => {
        this.setSectionHidden(
          SKELETON_SECTION_JSON_KEY,
          !this.layer.hasSkeletonsLayer.value,
        );
      }),
    );
  }
}
