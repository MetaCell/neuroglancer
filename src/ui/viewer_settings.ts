/**
 * @license
 * Copyright 2021 Google Inc.
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

import "#src/ui/viewer_settings.css";

import { TrackableBooleanCheckbox } from "#src/trackable_boolean.js";
import type {
  TrackableValue,
  WatchableValueInterface,
} from "#src/trackable_value.js";
import type { SidePanelManager } from "#src/ui/side_panel.js";
import { SidePanel } from "#src/ui/side_panel.js";
import type { SidePanelLocation } from "#src/ui/side_panel_location.js";
import {
  DEFAULT_SIDE_PANEL_LOCATION,
  TrackableSidePanelLocation,
} from "#src/ui/side_panel_location.js";
import type { vec3 } from "#src/util/geom.js";
import { emptyToUndefined } from "#src/util/json.js";
import type { Viewer } from "#src/viewer.js";
import { ColorWidget } from "#src/widget/color.js";
import { NumberInputWidget } from "#src/widget/number_input_widget.js";
import { TextInputWidget } from "#src/widget/text_input.js";

const DEFAULT_SETTINGS_PANEL_LOCATION: SidePanelLocation = {
  ...DEFAULT_SIDE_PANEL_LOCATION,
  side: "left",
  row: 2,
};

export class ViewerSettingsPanelState {
  location = new TrackableSidePanelLocation(DEFAULT_SETTINGS_PANEL_LOCATION);
  get changed() {
    return this.location.changed;
  }
  toJSON() {
    return emptyToUndefined(this.location.toJSON());
  }
  reset() {
    this.location.reset();
  }
  restoreState(obj: unknown) {
    this.location.restoreState(obj);
  }
}

export class ViewerSettingsPanel extends SidePanel {
  constructor(
    sidePanelManager: SidePanelManager,
    state: ViewerSettingsPanelState,
    viewer: Viewer,
  ) {
    super(sidePanelManager, state.location);
    this.addTitleBar({ title: "Settings" });

    const body = document.createElement("div");
    body.classList.add("neuroglancer-settings-body");

    const scroll = document.createElement("div");
    scroll.classList.add("neuroglancer-settings-scroll-container");
    body.appendChild(scroll);
    this.addBody(body);

    {
      const titleWidget = this.registerDisposer(
        new TextInputWidget(viewer.title),
      );
      titleWidget.element.placeholder = "Title";
      titleWidget.element.title =
        "A name for this view. It becomes the browser tab title.";
      titleWidget.element.classList.add("neuroglancer-settings-title");
      scroll.appendChild(titleWidget.element);
    }

    const addLimitWidget = (
      label: string,
      limit: TrackableValue<number>,
      description: string,
    ) => {
      const widget = this.registerDisposer(
        new NumberInputWidget(limit, { label }),
      );
      widget.element.classList.add("neuroglancer-settings-limit-widget");
      widget.element.title = description;
      scroll.appendChild(widget.element);
    };
    addLimitWidget(
      "GPU memory limit",
      viewer.chunkQueueManager.capacities.gpuMemory.sizeLimit,
      "Max GPU memory to spend on loaded chunks, in bytes, e.g. 1000000000 for 1GB.",
    );
    addLimitWidget(
      "System memory limit",
      viewer.chunkQueueManager.capacities.systemMemory.sizeLimit,
      "Max system (CPU) memory to spend on loaded chunks, in bytes, e.g. 2000000000 for 2GB.",
    );
    addLimitWidget(
      "Concurrent chunk requests",
      viewer.chunkQueueManager.capacities.download.itemLimit,
      "How many chunk downloads can be in flight at once. Higher can be faster but can cause chunks to appear in a less prioritised order.",
    );

    const addCheckbox = (
      label: string,
      value: WatchableValueInterface<boolean>,
      description: string,
    ) => {
      const labelElement = document.createElement("label");
      labelElement.textContent = label;
      labelElement.title = description;
      const checkbox = this.registerDisposer(
        new TrackableBooleanCheckbox(value),
      );
      labelElement.appendChild(checkbox.element);
      scroll.appendChild(labelElement);
    };
    addCheckbox(
      "Show axis lines",
      viewer.showAxisLines,
      "Draws the red, green and blue lines marking the x, y and z axes through the current position.",
    );
    addCheckbox(
      "Show scale bar",
      viewer.showScaleBar,
      "Shows a scale bar in the corner of each 2D panel, and 3D panels that use an ortographic projection",
    );
    addCheckbox(
      "Show picking indicator",
      viewer.showPickingIndicator,
      "Shows a white ring where the cursor is picking, and at that same point in the other panels. In 3D perspective projection the ring is depth modulated: larger for closer points, smaller for further ones.",
    );
    addCheckbox(
      "Show cross sections in 3-d",
      viewer.showPerspectiveSliceViews,
      "Draws the 2D cross sections as planes inside the 3D view, in layouts that have both 2D and 3D panels.",
    );
    addCheckbox(
      "Hide sections background 3-d",
      viewer.hideCrossSectionBackground3D,
      "Leaves the cross-section planes in the 3D view transparent where there is no data, instead of filling them with the cross-section background color.",
    );
    addCheckbox(
      "Show default annotations",
      viewer.showDefaultAnnotations,
      "Shows the bounding box of each layer's data source.",
    );
    addCheckbox(
      "Show chunk statistics",
      viewer.statisticsDisplayState.location.watchableVisible,
      "Opens a panel with live download and memory statistics for each layer's chunks.",
    );
    addCheckbox(
      "Wire frame rendering",
      viewer.wireFrame,
      "Draws chunk outlines instead of the data. Mostly useful for debugging what is loaded and at which resolution.",
    );
    addCheckbox(
      "Enable prefetching",
      viewer.chunkQueueManager.enablePrefetch,
      "Loads chunks just outside the current view before you get to them, so panning and scrolling through slices feel smoother at the cost of some extra bandwidth.",
    );
    addCheckbox(
      "Enable adaptive downsampling",
      viewer.enableAdaptiveDownsampling,
      "Renders volume rendering at a lower resolution while the 3D camera is moving, then renders at full resolution once it stops.",
    );

    const addColor = (
      label: string,
      value: WatchableValueInterface<vec3>,
      description: string,
    ) => {
      const labelElement = document.createElement("label");
      labelElement.textContent = label;
      labelElement.title = description;
      const widget = this.registerDisposer(new ColorWidget(value));
      labelElement.appendChild(widget.element);
      scroll.appendChild(labelElement);
    };

    addColor(
      "Cross-section background",
      viewer.crossSectionBackgroundColor,
      "Background color behind the 2D cross sections, wherever there is no data.",
    );
    addColor(
      "Projection background",
      viewer.perspectiveViewBackgroundColor,
      "Background color of the 3D view.",
    );
  }
}
