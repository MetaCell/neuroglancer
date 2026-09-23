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

import "#src/panel_overlay.css";

import type { CoordinateSpace } from "#src/coordinate_transform.js";
import type { DisplayContext } from "#src/display_context.js";
import type { MouseSelectionState } from "#src/layer/index.js";
import type { ProjectionParameters } from "#src/projection_parameters.js";
import type { WatchableValueInterface } from "#src/trackable_value.js";
import type { Disposable } from "#src/util/disposable.js";
import { RefCounted } from "#src/util/disposable.js";
import { getViewFrustumDepthRange, vec4 } from "#src/util/geom.js";
import type { NullarySignal } from "#src/util/signal.js";

export interface ViewportPoint {
  readonly viewportLeft: number;
  readonly viewportTop: number;
  readonly focalPlaneDepthFraction: number;
  readonly perspectiveDivideFactor: number;
}

const tempClip = vec4.create();

export function projectToViewport(
  parameters: ProjectionParameters,
  position: ArrayLike<number>,
): ViewportPoint | undefined {
  const {
    projectionMat,
    viewProjectionMat,
    logicalWidth,
    logicalHeight,
    displayDimensionRenderInfo: { displayDimensionIndices },
  } = parameters;
  const clip = tempClip;
  for (let i = 0; i < 3; ++i) {
    const index = displayDimensionIndices[i];
    clip[i] = index >= 0 ? position[index] : 0;
  }
  clip[3] = 1;
  vec4.transformMat4(clip, clip, viewProjectionMat);
  const w = clip[3];
  if (w <= 0) return undefined;
  const normalizedDeviceZ = clip[2] / w;
  if (normalizedDeviceZ < -1 || normalizedDeviceZ > 1) return undefined;
  const orthographic = projectionMat[15] === 1;
  return {
    viewportLeft: ((clip[0] / w) * 0.5 + 0.5) * logicalWidth,
    viewportTop: (0.5 - (clip[1] / w) * 0.5) * logicalHeight,
    focalPlaneDepthFraction: orthographic
      ? normalizedDeviceZ
      : (w - 1) / (getViewFrustumDepthRange(projectionMat) / 2),
    perspectiveDivideFactor: 1 / w,
  };
}

export type ProjectOverlayPosition = (
  position: Float32Array,
  coordinateSpace: CoordinateSpace,
) => ViewportPoint | undefined;

export interface PanelOverlay extends Disposable {
  update(): void;
}

export interface PanelOverlaySource {
  /** Higher draws on top of lower. */
  readonly overlayPriority?: number;
  readonly updateNeeded: NullarySignal;
  readonly visible?: WatchableValueInterface<boolean>;
  createPanelOverlay(
    container: HTMLElement,
    project: ProjectOverlayPosition,
  ): PanelOverlay;
}

interface Binding {
  readonly element: HTMLElement;
  readonly overlay: PanelOverlay;
  readonly owner: RefCounted;
}

export class PanelOverlayManager extends RefCounted {
  private readonly container = document.createElement("div");
  private readonly bindings = new Map<PanelOverlaySource, Binding>();
  private readonly requestUpdate = () => this.context.scheduleOverlayUpdate();

  constructor(
    panelElement: HTMLElement,
    private readonly context: DisplayContext,
    private readonly project: ProjectOverlayPosition,
  ) {
    super();
    this.container.className = "neuroglancer-panel-overlay-container";
    panelElement.appendChild(this.container);
    this.registerDisposer(() => this.container.remove());
    this.registerDisposer(
      context.panelOverlays.changed.add((source, added) => {
        if (source === null) return;
        if (added) this.bind(source);
        else this.unbind(source);
        this.requestUpdate();
      }),
    );
    this.registerDisposer(() => {
      for (const { owner } of this.bindings.values()) owner.dispose();
      this.bindings.clear();
    });
    for (const source of context.panelOverlays) this.bind(source);
  }

  private bind(source: PanelOverlaySource) {
    const element = document.createElement("div");
    element.className = "neuroglancer-panel-overlay-source";
    element.style.zIndex = `${source.overlayPriority ?? 0}`;
    this.container.appendChild(element);
    const overlay = source.createPanelOverlay(element, this.project);
    const owner = new RefCounted();
    owner.registerDisposer(overlay);
    owner.registerDisposer(() => element.remove());
    owner.registerDisposer(source.updateNeeded.add(this.requestUpdate));
    const { visible } = source;
    if (visible !== undefined) {
      owner.registerDisposer(visible.changed.add(this.requestUpdate));
    }
    this.bindings.set(source, { element, overlay, owner });
  }

  private unbind(source: PanelOverlaySource) {
    const binding = this.bindings.get(source);
    if (binding === undefined) return;
    this.bindings.delete(source);
    binding.owner.dispose();
  }

  update() {
    for (const [source, { element, overlay }] of this.bindings) {
      element.hidden = source.visible?.value === false;
      if (element.hidden) continue;
      overlay.update();
    }
  }

  hide() {
    for (const { element } of this.bindings.values()) element.hidden = true;
  }
}

const PICKING_INDICATOR_DIAMETER = 14;
const PICKING_INDICATOR_MIN_SCALE = 0.6;
const PICKING_INDICATOR_MAX_SCALE = 1.7;

class PickingIndicator implements PanelOverlay {
  private readonly ring = document.createElement("div");

  constructor(
    container: HTMLElement,
    private readonly project: ProjectOverlayPosition,
    private readonly mouseState: MouseSelectionState,
  ) {
    this.ring.className = "neuroglancer-picking-indicator";
    container.appendChild(this.ring);
  }

  update() {
    const { mouseState, ring } = this;
    const point = mouseState.active
      ? this.project(mouseState.position, mouseState.coordinateSpace)
      : undefined;
    ring.hidden = point === undefined;
    if (point === undefined) return;
    const scale = Math.min(
      PICKING_INDICATOR_MAX_SCALE,
      Math.max(PICKING_INDICATOR_MIN_SCALE, point.perspectiveDivideFactor),
    );
    const size = PICKING_INDICATOR_DIAMETER * scale;
    const { style } = ring;
    style.width = `${size}px`;
    style.height = `${size}px`;
    style.opacity = `${1 - Math.abs(point.focalPlaneDepthFraction)}`;
    style.transform = `translate(${point.viewportLeft - size / 2}px, ${point.viewportTop - size / 2}px)`;
  }

  dispose() {
    this.ring.remove();
  }
}

export class PickingIndicatorOverlay implements PanelOverlaySource {
  readonly overlayPriority = 100;

  constructor(
    private readonly mouseState: MouseSelectionState,
    readonly visible: WatchableValueInterface<boolean>,
  ) {}

  get updateNeeded(): NullarySignal {
    return this.mouseState.changed;
  }

  createPanelOverlay(
    container: HTMLElement,
    project: ProjectOverlayPosition,
  ): PanelOverlay {
    return new PickingIndicator(container, project, this.mouseState);
  }
}
