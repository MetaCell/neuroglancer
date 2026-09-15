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

import "#src/panel_overlay.css";

import type { CoordinateSpace } from "#src/coordinate_transform.js";
import type { DisplayContext } from "#src/display_context.js";
import type { WatchableValueInterface } from "#src/trackable_value.js";
import { RefCounted } from "#src/util/disposable.js";
import type { NullarySignal } from "#src/util/signal.js";

/**
 * Logical CSS pixel position within a panel.  `scale` conveys depth in
 * perspective views; `opacity` is the cross-section fade in slice views.  Both
 * default to 1.
 */
export interface ProjectedPosition {
  readonly x: number;
  readonly y: number;
  readonly scale?: number;
  readonly opacity?: number;
}

export interface PanelOverlayContext {
  project(
    position: Float32Array,
    coordinateSpace: CoordinateSpace,
  ): ProjectedPosition | undefined;
  readonly container: HTMLElement;
}

export interface PanelOverlaySource {
  /** Higher draws on top of lower.  Default 0. */
  readonly overlayPriority?: number;
  readonly overlayUpdateNeeded: NullarySignal;
  readonly overlayVisible?: WatchableValueInterface<boolean>;
  updatePanelOverlays(context: PanelOverlayContext): void;
}

export interface PanelOverlayHost {
  readonly element: HTMLElement;
  projectPosition(
    position: Float32Array,
    coordinateSpace: CoordinateSpace,
  ): ProjectedPosition | undefined;
}

interface Binding {
  readonly element: HTMLElement;
  readonly owner: RefCounted;
}

export class PanelOverlayManager extends RefCounted {
  private readonly container = document.createElement("div");
  private readonly bindings = new Map<PanelOverlaySource, Binding>();
  private readonly requestUpdate = () => this.context.scheduleOverlayUpdate();

  constructor(
    private readonly host: PanelOverlayHost,
    private readonly context: DisplayContext,
  ) {
    super();
    this.container.className = "neuroglancer-panel-overlay-container";
    host.element.appendChild(this.container);
    this.registerDisposer(() => this.container.remove());
    this.registerDisposer(
      context.panelOverlaysChanged.add(() => this.syncSources()),
    );
    this.registerDisposer(() => {
      for (const { owner } of this.bindings.values()) owner.dispose();
      this.bindings.clear();
    });
    this.syncSources();
  }

  private syncSources() {
    const { panelOverlays } = this.context;
    for (const [source, binding] of this.bindings) {
      if (!panelOverlays.has(source)) {
        binding.owner.dispose();
        this.bindings.delete(source);
      }
    }
    for (const source of panelOverlays) {
      if (!this.bindings.has(source)) {
        this.bindings.set(source, this.bind(source));
      }
    }
    this.requestUpdate();
  }

  private bind(source: PanelOverlaySource): Binding {
    const element = document.createElement("div");
    element.className = "neuroglancer-panel-overlay-source";
    element.style.zIndex = `${source.overlayPriority ?? 0}`;
    this.container.appendChild(element);
    const owner = new RefCounted();
    owner.registerDisposer(() => element.remove());
    owner.registerDisposer(source.overlayUpdateNeeded.add(this.requestUpdate));
    const { overlayVisible } = source;
    if (overlayVisible !== undefined) {
      owner.registerDisposer(overlayVisible.changed.add(this.requestUpdate));
    }
    return { element, owner };
  }

  update() {
    const { host } = this;
    const project = (
      position: Float32Array,
      coordinateSpace: CoordinateSpace,
    ) => host.projectPosition(position, coordinateSpace);
    for (const [source, { element }] of this.bindings) {
      element.hidden = source.overlayVisible?.value === false;
      if (element.hidden) continue;
      source.updatePanelOverlays({ project, container: element });
    }
  }

  clear() {
    for (const { element } of this.bindings.values()) element.hidden = true;
  }
}
