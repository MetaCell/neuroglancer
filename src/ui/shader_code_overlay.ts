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

type OverlayWithCodeWidget = Overlay & {
  footerActionsBtnContainer: HTMLDivElement;
  footerBtnsWrapper: HTMLDivElement;
  createButton: (
    text: string | null,
    onClick: () => void,
    cssClass?: string,
    svgUrl?: string | null,
  ) => HTMLButtonElement;
};

export function commonShaderCodeOverlaySetup<T extends OverlayWithCodeWidget>(
  overlay: T,
  textEditor: HTMLElement,
  attributeDisplayElement?: HTMLElement,
  title: string = "Shader editor",
) {
  overlay.content.classList.add("modal-lg");
  const titleText = document.createElement("p");
  titleText.textContent = title;
  const closeMenuButton = overlay.createButton(
    null,
    () => overlay.close(),
    "",
    svg_close,
  );
  const closeAndHelpContainer = document.createElement("div");
  closeAndHelpContainer.classList.add("overlay-content-header");

  closeAndHelpContainer.appendChild(titleText);
  closeAndHelpContainer.appendChild(closeMenuButton);

  overlay.content.appendChild(closeAndHelpContainer);

  const mainBody = document.createElement("div");
  mainBody.classList.add("overlay-content-body");
  overlay.content.appendChild(mainBody);
  if (attributeDisplayElement) {
    mainBody.appendChild(attributeDisplayElement);
  }
  mainBody.appendChild(textEditor);

  overlay.footerActionsBtnContainer = document.createElement("div");
  overlay.footerActionsBtnContainer.classList.add("overlay-content-footer");
  overlay.footerBtnsWrapper = document.createElement("div");
  overlay.footerBtnsWrapper.classList.add("button-wrapper");
  overlay.content.appendChild(overlay.footerActionsBtnContainer);
}

export class ShaderCodeOverlay extends Overlay {
  attributeWidget?: VertexAttributeWidget;
  footerActionsBtnContainer: HTMLDivElement;
  footerBtnsWrapper: HTMLDivElement;
  constructor(
    public layer: UserLayer,
    private makeShaderCodeWidget: (layer: UserLayer) => ShaderCodeWidget,
    options: ShaderCodeOverlayOptions = {},
    makeVertexAttributeWidget?: (layer: UserLayer) => VertexAttributeWidget,
  ) {
    super();
    const { additionalClass, title = "Shader editor" } = options;

    if (additionalClass) {
      this.content.classList.add(additionalClass);
    }

    const codeWidget = this.registerDisposer(
      this.makeShaderCodeWidget(this.layer),
    );
    if (makeVertexAttributeWidget) {
      this.attributeWidget = this.registerDisposer(
        makeVertexAttributeWidget(this.layer),
      );
    }
    commonShaderCodeOverlaySetup(
      this,
      codeWidget.element,
      this.attributeWidget?.element,
      title,
    );
    codeWidget.textEditor.refresh();
  }

  public createButton(
    text: string | null,
    onClick: () => void,
    cssClass: string = "",
    svgUrl: string | null = null,
  ): HTMLButtonElement {
    const button = document.createElement("button");
    if (svgUrl) {
      const icon = makeIcon({ svg: svgUrl });
      button.appendChild(icon);
    } else if (text) {
      button.textContent = text;
    }
    if (cssClass) button.classList.add(cssClass);
    button.addEventListener("click", onClick);
    return button;
  }
}
