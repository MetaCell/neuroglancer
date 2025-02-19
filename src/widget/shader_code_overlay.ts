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

import type { AnnotationUserLayer } from "#src/layer/annotation/index.ts";
import { makeFooterBtnGroup } from "#src/layer/image/shader_overlay_footer.js";
import { Overlay } from "#src/overlay.js";
import svg_close from "#src/ui/images/metacell/close.svg?raw";
import { makeIcon } from "#src/widget/icon.js";
import type { ShaderCodeWidget } from "#src/widget/shader_code_widget.js";

interface ShaderCodeOverlayOptions {
  additionalClass?: string;
  title?: string;
}

export class ShaderCodeOverlay extends Overlay {
  closeMenuButton: HTMLButtonElement;
  codeWidget: ShaderCodeWidget;
  footerActionsBtnContainer: HTMLDivElement;
  footerBtnsWrapper: HTMLDivElement;
  constructor(public layer: AnnotationUserLayer, makeShaderCodeWidget: (layer: AnnotationUserLayer) => ShaderCodeWidget, options: ShaderCodeOverlayOptions = {}) {
    super();
    const { additionalClass = "", title = "Shader editor" } = options;

    this.content.classList.add("modal-lg", ...additionalClass);

    const titleText = document.createElement("p");
    titleText.textContent = title;

    this.closeMenuButton = this.createButton(null, () => this.close(), "", svg_close);

    this.codeWidget = this.registerDisposer(makeShaderCodeWidget(this.layer));

    const closeAndHelpContainer = document.createElement("div");
    closeAndHelpContainer.classList.add("overlay-content-header");

    closeAndHelpContainer.appendChild(titleText);
    closeAndHelpContainer.appendChild(this.closeMenuButton);

    this.content.appendChild(closeAndHelpContainer);

    const mainBody = document.createElement("div");
    mainBody.classList.add("overlay-content-body");
    mainBody.appendChild(this.codeWidget.element);
    this.content.appendChild(mainBody);
    this.codeWidget.textEditor.refresh();

    this.footerActionsBtnContainer = document.createElement("div");
    this.footerActionsBtnContainer.classList.add("overlay-content-footer");
    this.footerActionsBtnContainer.appendChild(
      makeFooterBtnGroup(() => this.close()),
    );
    this.content.appendChild(this.footerActionsBtnContainer);
    this.codeWidget.textEditor.refresh();
  }
  
  private createButton(
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