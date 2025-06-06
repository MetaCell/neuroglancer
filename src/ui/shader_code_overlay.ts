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

import type { AnnotationUserLayer } from "#src/layer/annotation/index.js";
import type { ImageUserLayer } from "#src/layer/image/index.js";
import type {
  SingleMeshUserLayer,
  VertexAttributeWidget,
} from "#src/layer/single_mesh/index.js";
import { Overlay } from "#src/overlay.js";
import svg_close from "ikonate/icons/close.svg?raw";
import { makeIcon } from "#src/widget/icon.js";
import type { ShaderCodeWidget } from "#src/widget/shader_code_widget.js";
import { TrackableBoolean } from "#src/trackable_boolean.js";
import { SegmentationUserLayer } from "#src/layer/segmentation/index.js";

type UserLayer =
  | AnnotationUserLayer
  | ImageUserLayer
  | SingleMeshUserLayer
  | SegmentationUserLayer;

interface ShaderCodeOverlayOptions {
  additionalClass?: string;
  title?: string;
}

export type UserLayerWithCodeEditor = UserLayer & {
  codeVisible: TrackableBoolean;
};

export type ShaderCodeOverlayConstructor<T extends Overlay> = new (
  layer: UserLayerWithCodeEditor,
  makeShaderCodeWidget: (layer: UserLayerWithCodeEditor) => ShaderCodeWidget,
) => T;

export function makeFooterBtnGroup(onClose: () => void) {
  const buttonApply = document.createElement("button");
  buttonApply.textContent = "Close";
  buttonApply.classList.add("cancel-button");

  buttonApply.addEventListener("click", () => {
    onClose();
  });

  return buttonApply;
}

export class OverlayWithCodeEditor extends Overlay {
  header: HTMLDivElement;
  body: HTMLDivElement;
  footer: HTMLDivElement;
  createCloseMenuButton() {
    const button = document.createElement("button");
    const icon = makeIcon({ svg: svg_close });
    button.appendChild(icon);
    button.addEventListener("click", () => this.close());
    return button;
  }
  constructor(title: string = "Code editor") {
    super();
    this.content.classList.add("neuroglancer-code-editor-overlay");

    const header = (this.header = document.createElement("div"));
    const closeMenuButton = this.createCloseMenuButton();
    const titleText = document.createElement("p");
    titleText.textContent = title;
    header.classList.add("neuroglancer-code-editor-overlay-header");
    header.appendChild(titleText);
    header.appendChild(closeMenuButton);
    this.content.appendChild(header);

    const body = (this.body = document.createElement("div"));
    body.classList.add("neuroglancer-code-editor-overlay-body");
    this.content.appendChild(body);

    const footer = (this.footer = document.createElement("div"));
    footer.classList.add("neuroglancer-code-editor-overlay-footer");
    this.content.appendChild(this.footer);
  }
}

export class ShaderCodeOverlay extends OverlayWithCodeEditor {
  footerActionsBtnContainer: HTMLDivElement;
  footerBtnsWrapper: HTMLDivElement;
  constructor(
    public layer: UserLayer,
    private makeShaderCodeWidget: (layer: UserLayer) => ShaderCodeWidget,
    options: ShaderCodeOverlayOptions = {},
    makeVertexAttributeWidget?: (layer: UserLayer) => VertexAttributeWidget,
  ) {
    const { additionalClass, title = "Shader editor" } = options;
    super(title);

    if (additionalClass) {
      this.content.classList.add(additionalClass);
    }

    const codeWidget = this.registerDisposer(
      this.makeShaderCodeWidget(this.layer),
    );
    if (makeVertexAttributeWidget) {
      const attributeWidget = this.registerDisposer(
        makeVertexAttributeWidget(this.layer),
      );
      this.body.appendChild(attributeWidget.element);
    }
    this.body.appendChild(codeWidget.element);
    codeWidget.textEditor.refresh();
  }
}
