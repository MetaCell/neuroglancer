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
import "#src/ui/spatial_skeleton_delete_confirmation.css";
import { makeIcon } from "#src/widget/icon.js";

export type SpatialSkeletonSoftDeleteConflictChoice = "restore" | "hide";

class SpatialSkeletonSoftDeleteConflictDialog extends Overlay {
  private resolved = false;

  constructor(
    private skeletonId: number,
    private resolve: (choice: SpatialSkeletonSoftDeleteConflictChoice) => void,
  ) {
    super();
    const { content } = this;
    content.classList.add("neuroglancer-spatial-skeleton-delete-confirmation");

    const header = document.createElement("div");
    header.className =
      "neuroglancer-spatial-skeleton-delete-confirmation-header";
    const title = document.createElement("h2");
    title.textContent = "Skeleton was deleted";
    header.appendChild(title);

    const closeButton = document.createElement("button");
    closeButton.type = "button";
    closeButton.className =
      "neuroglancer-spatial-skeleton-delete-confirmation-close";
    closeButton.title = "Cancel and hide";
    closeButton.appendChild(
      makeIcon({
        svg: svg_close,
        title: "Cancel and hide",
        clickable: false,
      }),
    );
    closeButton.addEventListener("click", () => this.finish("hide"));
    header.appendChild(closeButton);
    content.appendChild(header);

    const body = document.createElement("div");
    body.className = "neuroglancer-spatial-skeleton-delete-confirmation-body";
    const description = document.createElement("p");
    description.textContent = `Skeleton ${this.skeletonId} was marked deleted in CATMAID by another session.`;
    body.appendChild(description);
    content.appendChild(body);

    const buttons = document.createElement("div");
    buttons.className =
      "neuroglancer-spatial-skeleton-delete-confirmation-buttons";

    const restoreButton = document.createElement("button");
    restoreButton.type = "button";
    restoreButton.className =
      "neuroglancer-spatial-skeleton-delete-confirmation-delete";
    restoreButton.textContent = "Restore and proceed";
    restoreButton.addEventListener("click", () => this.finish("restore"));
    buttons.appendChild(restoreButton);

    const hideButton = document.createElement("button");
    hideButton.type = "button";
    hideButton.textContent = "Cancel and hide";
    hideButton.addEventListener("click", () => this.finish("hide"));
    buttons.appendChild(hideButton);

    content.appendChild(buttons);
    restoreButton.focus();
  }

  private finish(choice: SpatialSkeletonSoftDeleteConflictChoice) {
    if (this.resolved) return;
    this.resolved = true;
    this.resolve(choice);
    this.dispose();
  }

  disposed() {
    if (!this.resolved) {
      this.resolved = true;
      this.resolve("hide");
    }
    super.disposed();
  }
}

export function confirmSpatialSkeletonSoftDeleteConflict(skeletonId: number) {
  return new Promise<SpatialSkeletonSoftDeleteConflictChoice>((resolve) => {
    new SpatialSkeletonSoftDeleteConflictDialog(skeletonId, resolve);
  });
}
