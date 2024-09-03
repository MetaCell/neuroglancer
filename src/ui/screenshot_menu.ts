/**
 * @license
 * Copyright 2024 Google Inc.
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

import { debounce } from "lodash-es";
import { Overlay } from "#src/overlay.js";
import "#src/ui/screenshot_menu.css";

import { ScreenshotMode } from "#src/util/trackable_screenshot_mode.js";
import type { Viewer } from "#src/viewer.js";

const friendlyNameMap = {
  chunkUsageDescription: "Number of loaded chunks",
  gpuMemoryUsageDescription: "Visible chunk GPU memory usage",
  downloadSpeedDescription: "Number of downloading chunks",
};

export class ScreenshotDialog extends Overlay {
  private nameInput: HTMLInputElement;
  private saveButton: HTMLButtonElement;
  private closeButton: HTMLButtonElement;
  private forceScreenshotButton: HTMLButtonElement;
  private statisticsTable: HTMLTableElement;
  private statisticsContainer: HTMLDivElement;
  private scaleSelectContainer: HTMLDivElement;
  private filenameAndButtonsContainer: HTMLDivElement;
  private screenshotMode: ScreenshotMode;
  private statisticsKeyToCellMap: Map<string, HTMLTableCellElement> = new Map();
  constructor(public viewer: Viewer) {
    super();
    this.screenshotMode = this.viewer.display.screenshotMode.value;

    this.initializeUI();
    this.setupEventListeners();
  }

  private initializeUI() {
    this.content.classList.add("neuroglancer-screenshot-dialog");

    this.closeButton = this.createButton(
      "Close",
      () => this.dispose(),
      "neuroglancer-screenshot-close-button",
    );
    this.saveButton = this.createButton("Take screenshot", () =>
      this.screenshot(),
    );
    this.forceScreenshotButton = this.createButton("Force screenshot", () =>
      this.forceScreenshot(),
    );
    this.forceScreenshotButton.title =
      "Force a screenshot of the current view without waiting for all data to be loaded and rendered";
    this.filenameAndButtonsContainer = document.createElement("div");
    this.filenameAndButtonsContainer.classList.add(
      "neuroglancer-screenshot-filename-and-buttons",
    );
    this.filenameAndButtonsContainer.appendChild(this.createNameInput());
    this.filenameAndButtonsContainer.appendChild(this.saveButton);

    this.content.appendChild(this.closeButton);
    this.content.appendChild(this.filenameAndButtonsContainer);
    this.content.appendChild(this.createScaleRadioButtons());
    this.content.appendChild(this.createStatisticsTable());
    this.updateSetupUIVisibility();
  }

  private setupEventListeners() {
    this.registerDisposer(
      this.viewer.screenshotActionHandler.sendScreenshotRequested.add(() => {
        this.debouncedUpdateUIElements();
        this.dispose();
      }),
    );
    this.registerDisposer(
      this.viewer.screenshotActionHandler.sendStatisticsRequested.add(() => {
        this.populateStatistics();
      }),
    );
  }

  private createNameInput(): HTMLInputElement {
    const nameInput = document.createElement("input");
    nameInput.type = "text";
    nameInput.placeholder = "Enter optional filename...";
    nameInput.classList.add("neuroglancer-screenshot-name-input");
    return (this.nameInput = nameInput);
  }

  private updateStatisticsTableDisplayBasedOnMode() {
    if (this.screenshotMode === ScreenshotMode.OFF) {
      this.statisticsContainer.style.display = "none";
    } else {
      this.statisticsContainer.style.display = "block";
    }
  }

  private createButton(
    text: string,
    onClick: () => void,
    cssClass: string = "",
  ): HTMLButtonElement {
    const button = document.createElement("button");
    button.textContent = text;
    button.classList.add("neuroglancer-screenshot-button");
    if (cssClass) button.classList.add(cssClass);
    button.addEventListener("click", onClick);
    return button;
  }

  private createScaleRadioButtons() {
    const scaleMenu = (this.scaleSelectContainer =
      document.createElement("div"));
    scaleMenu.classList.add("neuroglancer-screenshot-scale-menu");

    const scaleLabel = document.createElement("label");
    scaleLabel.textContent = "Screenshot scale factor:";
    scaleMenu.appendChild(scaleLabel);

    const scales = [1, 2, 4];
    scales.forEach((scale) => {
      const label = document.createElement("label");
      const input = document.createElement("input");

      input.type = "radio";
      input.name = "screenshot-scale";
      input.value = scale.toString();
      input.checked = scale === this.screenshotHandler.screenshotScale;
      input.classList.add("neuroglancer-screenshot-scale-radio");

      label.appendChild(input);
      label.appendChild(document.createTextNode(`${scale}x`));

      scaleMenu.appendChild(label);

      input.addEventListener("change", () => {
        this.screenshotHandler.screenshotScale = scale;
      });
    });
    return scaleMenu;
  }

  private createStatisticsTable() {
    this.statisticsContainer = document.createElement("div");
    this.statisticsContainer.classList.add(
      "neuroglancer-screenshot-statistics-title",
    );
    this.statisticsContainer.appendChild(this.forceScreenshotButton);

    this.statisticsTable = document.createElement("table");
    this.statisticsTable.classList.add(
      "neuroglancer-screenshot-statistics-table",
    );
    this.statisticsTable.title = "Screenshot statistics";

    const headerRow = this.statisticsTable.createTHead().insertRow();
    const keyHeader = document.createElement("th");
    keyHeader.textContent = "Screenshot in progress";
    headerRow.appendChild(keyHeader);
    const valueHeader = document.createElement("th");
    valueHeader.textContent = "";
    headerRow.appendChild(valueHeader);

    // Populate inital table elements with placeholder text
    const statsRow = this.screenshotHandler.screenshotStatistics;
    for (const key in statsRow) {
      if (key === "timeElapsedString") {
        continue;
      }
      const row = this.statisticsTable.insertRow();
      const keyCell = row.insertCell();
      const valueCell = row.insertCell();
      keyCell.textContent =
        friendlyNameMap[key as keyof typeof friendlyNameMap];
      valueCell.textContent = "Loading...";
      this.statisticsKeyToCellMap.set(key, valueCell);
    }

    this.populateStatistics();
    this.updateStatisticsTableDisplayBasedOnMode();
    this.statisticsContainer.appendChild(this.statisticsTable);
    return this.statisticsContainer;
  }

  private forceScreenshot() {
    this.screenshotHandler.forceScreenshot();
    this.debouncedUpdateUIElements();
  }

  private screenshot() {
    const filename = this.nameInput.value;
    this.screenshotHandler.takeScreenshot(filename);
    this.debouncedUpdateUIElements();
  }

  private populateStatistics() {
    const statsRow = this.screenshotHandler.screenshotStatistics;

    for (const key in statsRow) {
      if (key === "timeElapsedString") {
        const headerRow = this.statisticsTable.rows[0];
        const keyHeader = headerRow.cells[0];
        const time = statsRow[key];
        if (time === null) {
          keyHeader.textContent = "Screenshot in progress (statistics loading)";
        } else {
          keyHeader.textContent = `Screenshot in progress for ${statsRow[key]}s`;
        }
        continue;
      }
      this.statisticsKeyToCellMap.get(key)!.textContent = String(
        statsRow[key as keyof typeof statsRow],
      );
    }
  }

  private debouncedUpdateUIElements = debounce(() => {
    this.updateSetupUIVisibility();
    this.updateStatisticsTableDisplayBasedOnMode();
  }, 200);

  private updateSetupUIVisibility() {
    this.screenshotMode = this.viewer.display.screenshotMode.value;
    if (this.screenshotMode === ScreenshotMode.OFF) {
      this.forceScreenshotButton.style.display = "none";
      this.filenameAndButtonsContainer.style.display = "block";
      this.scaleSelectContainer.style.display = "block";
    } else {
      this.forceScreenshotButton.style.display = "block";
      this.filenameAndButtonsContainer.style.display = "none";
      this.scaleSelectContainer.style.display = "none";
    }
  }

  get screenshotHandler() {
    return this.viewer.screenshotHandler;
  }
}
