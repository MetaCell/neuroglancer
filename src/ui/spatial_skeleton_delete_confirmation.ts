/**
 * @license
 * Copyright 2026 Google Inc.
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

import svg_close from "ikonate/icons/close.svg?raw";
import { Overlay } from "#src/overlay.js";
import { makeIcon } from "#src/widget/icon.js";
import "#src/ui/spatial_skeleton_delete_confirmation.css";

export interface SpatialSkeletonDeleteConfirmationOptions {
  kind: "skeleton" | "subtree";
  segmentId: number;
  nodeId?: number;
}

class SpatialSkeletonDeleteConfirmationDialog extends Overlay {
  private resolved = false;

  constructor(
    options: SpatialSkeletonDeleteConfirmationOptions,
    private resolve: (confirmed: boolean) => void,
  ) {
    super();
    const { content } = this;
    content.classList.add(
      "neuroglancer-spatial-skeleton-delete-confirmation",
    );

    const header = document.createElement("div");
    header.className =
      "neuroglancer-spatial-skeleton-delete-confirmation-header";
    const title = document.createElement("h2");
    title.textContent =
      options.kind === "skeleton"
        ? "Confirm delete skeleton"
        : "Confirm delete subtree";
    header.appendChild(title);

    const closeButton = document.createElement("button");
    closeButton.type = "button";
    closeButton.className =
      "neuroglancer-spatial-skeleton-delete-confirmation-close";
    closeButton.title = "Cancel";
    closeButton.appendChild(
      makeIcon({ svg: svg_close, title: "Cancel", clickable: false }),
    );
    closeButton.addEventListener("click", () => this.finish(false));
    header.appendChild(closeButton);
    content.appendChild(header);

    const body = document.createElement("div");
    body.className = "neuroglancer-spatial-skeleton-delete-confirmation-body";

    const description = document.createElement("p");
    if (options.kind === "skeleton") {
      description.textContent = "This will delete the selected skeleton.";
    } else {
      description.textContent =
        options.nodeId === undefined
          ? "This will delete the selected subtree."
          : `This will delete the subtree starting at node ${options.nodeId}.`;
    }
    body.appendChild(description);
    content.appendChild(body);

    const buttons = document.createElement("div");
    buttons.className =
      "neuroglancer-spatial-skeleton-delete-confirmation-buttons";

    const deleteButton = document.createElement("button");
    deleteButton.type = "button";
    deleteButton.className =
      "neuroglancer-spatial-skeleton-delete-confirmation-delete";
    deleteButton.textContent = "Confirm";
    deleteButton.addEventListener("click", () => this.finish(true));
    buttons.appendChild(deleteButton);

    content.appendChild(buttons);
    deleteButton.focus();
  }

  private finish(confirmed: boolean) {
    if (this.resolved) return;
    this.resolved = true;
    this.resolve(confirmed);
    this.dispose();
  }

  disposed() {
    if (!this.resolved) {
      this.resolved = true;
      this.resolve(false);
    }
    super.disposed();
  }
}

export function confirmSpatialSkeletonDeletion(
  options: SpatialSkeletonDeleteConfirmationOptions,
) {
  return new Promise<boolean>((resolve) => {
    new SpatialSkeletonDeleteConfirmationDialog(options, resolve);
  });
}
