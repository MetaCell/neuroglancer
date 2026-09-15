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

import "#src/picking_indicator_overlay.css";

import type { MouseSelectionState } from "#src/layer/index.js";
import type {
  PanelOverlayContext,
  PanelOverlaySource,
} from "#src/panel_overlay.js";
import type { WatchableValueInterface } from "#src/trackable_value.js";
import type { NullarySignal } from "#src/util/signal.js";

const PICKING_INDICATOR_DIAMETER = 14;
const PICKING_INDICATOR_MIN_DEPTH_SCALE = 0.6;
const PICKING_INDICATOR_MAX_DEPTH_SCALE = 1.7;

export class PickingIndicatorOverlay implements PanelOverlaySource {
  readonly overlayPriority = 100;
  private readonly rings = new WeakMap<HTMLElement, HTMLElement>();

  constructor(
    private readonly mouseState: MouseSelectionState,
    readonly overlayVisible: WatchableValueInterface<boolean>,
  ) {}

  get overlayUpdateNeeded(): NullarySignal {
    return this.mouseState.changed;
  }

  private ringFor(container: HTMLElement): HTMLElement {
    let ring = this.rings.get(container);
    if (ring === undefined) {
      ring = document.createElement("div");
      ring.className = "neuroglancer-picking-indicator";
      container.appendChild(ring);
      this.rings.set(container, ring);
    }
    return ring;
  }

  updatePanelOverlays(context: PanelOverlayContext): void {
    const { mouseState } = this;
    const ring = this.ringFor(context.container);
    const projected = mouseState.active
      ? context.project(mouseState.position, mouseState.coordinateSpace)
      : undefined;
    ring.hidden = projected === undefined;
    if (projected === undefined) return;
    const scale = Math.min(
      PICKING_INDICATOR_MAX_DEPTH_SCALE,
      Math.max(PICKING_INDICATOR_MIN_DEPTH_SCALE, projected.scale ?? 1),
    );
    const size = PICKING_INDICATOR_DIAMETER * scale;
    const { style } = ring;
    style.width = `${size}px`;
    style.height = `${size}px`;
    style.opacity = `${projected.opacity ?? 1}`;
    style.transform = `translate(${projected.x - size / 2}px, ${projected.y - size / 2}px)`;
  }
}
