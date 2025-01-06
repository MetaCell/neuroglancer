/**
 * @license
 * Copyright 2020 Google Inc.
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

import "#src/widget/checkbox_icon.css";

import type { WatchableValueInterface } from "#src/trackable_value.js";
import { RefCounted } from "#src/util/disposable.js";
import type { MakeIconOptions } from "#src/widget/icon.js";
import { makeIcon } from "#src/widget/icon.js";

export interface MakeCheckboxIconOptions
  extends Omit<MakeIconOptions, "onClick" | "title"> {
  enableTitle?: string;
  disableTitle?: string;
  backgroundScheme?: "light" | "dark";
  enableSvg?: string;
  disableSvg?: string;
}

export class CheckboxIcon extends RefCounted {
  readonly element: HTMLElement;
  constructor(
    model: WatchableValueInterface<boolean>,
    options: MakeCheckboxIconOptions,
  ) {
    super();
    this.element = makeIcon({
      ...options,
      onClick: () => {
        model.value = !model.value;
      },
    });
    this.element.classList.add("neuroglancer-checkbox-icon");
    this.element.classList.add(
      options.backgroundScheme === "dark"
        ? "dark-background"
        : "light-background",
    );
    const updateView = () => {
      const value = model.value;
      this.element.dataset.checked = value ? "true" : "false";
      this.element.title =
        (value ? options.disableTitle : options.enableTitle) || "";
      this.element.innerHTML = (value ? options.disableSvg : options.enableSvg) || "";
    };
    this.registerDisposer(model.changed.add(updateView));
    updateView();
  }
}
