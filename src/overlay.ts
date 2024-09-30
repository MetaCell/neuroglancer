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
import close from "ikonate/icons/close.svg?raw";
import svg_plus from "#src/ui/images/expandIcon.svg?raw";
import { AutomaticallyFocusedElement } from "#src/util/automatic_focus.js";
import { RefCounted } from "#src/util/disposable.js";
import {
  EventActionMap,
  KeyboardEventBinder,
} from "#src/util/keyboard_bindings.js";
import "#src/overlay.css";
import { makeIcon } from "#src/widget/icon.js";

export const overlayKeyboardHandlerPriority = 100;

export let overlaysOpen = 0;

export const defaultEventMap = EventActionMap.fromObject({
  escape: { action: "close" },
});

export class Overlay extends RefCounted {
  container: HTMLDivElement;
  content: HTMLDivElement;
  keyMap = new EventActionMap();
  constructor() {
    super();
    this.keyMap.addParent(defaultEventMap, Number.NEGATIVE_INFINITY);
    ++overlaysOpen;
    const container = (this.container = document.createElement("div"));
    container.className = "overlay";
    container.className = "metacell-theme";
    const content = (this.content = document.createElement("div"));
    this.registerDisposer(new AutomaticallyFocusedElement(content));
    content.className = "overlay-content";
    content.classList.add("neuroglancer-state-editor")
    container.appendChild(content);

    const wrapper = document.createElement("div");
    wrapper.className = "overlay-header";

    const heading = document.createElement("p");
    heading.className = "overlay-heading";
    heading.textContent = "Code editor";

    wrapper.appendChild(heading);

    wrapper.appendChild(
      makeIcon({
        svg: svg_plus,
        onClick: () => {
        },
      }),
    );

    wrapper.appendChild(
      makeIcon({
        svg: close,
        onClick: () => {
        },
      }),
    );
    this.content.appendChild(wrapper)
    document.body.appendChild(container);
    this.registerDisposer(new KeyboardEventBinder(this.container, this.keyMap));
    this.registerEventListener(container, "action:close", () => {
      this.dispose();
    });
    content.focus();
  }

  disposed() {
    --overlaysOpen;
    document.body.removeChild(this.container);
    super.disposed();
  }
}
