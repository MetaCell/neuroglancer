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

import { describe, expect, it } from "vitest";
import { emptyInvalidCoordinateSpace } from "#src/coordinate_transform.js";
import type { DisplayContext } from "#src/display_context.js";
import type { MouseSelectionState } from "#src/layer/index.js";
import type {
  PanelOverlayContext,
  PanelOverlaySource,
} from "#src/panel_overlay.js";
import { PanelOverlayManager } from "#src/panel_overlay.js";
import { PickingIndicatorOverlay } from "#src/picking_indicator_overlay.js";
import { WatchableValue } from "#src/trackable_value.js";
import { NullarySignal } from "#src/util/signal.js";

class RecordingSource implements PanelOverlaySource {
  readonly overlayUpdateNeeded = new NullarySignal();
  readonly overlayVisible = new WatchableValue(true);
  updateCount = 0;
  updatePanelOverlays(_context: PanelOverlayContext) {
    ++this.updateCount;
  }
}

function makeFixture() {
  const panelOverlays = new Set<PanelOverlaySource>();
  const panelOverlaysChanged = new NullarySignal();
  let updateRequests = 0;
  const context = {
    panelOverlays,
    panelOverlaysChanged,
    scheduleOverlayUpdate: () => {
      ++updateRequests;
    },
  } as unknown as DisplayContext;
  const panelElement = document.createElement("div");
  const manager = new PanelOverlayManager(panelElement, context);
  const setRegistered = (source: PanelOverlaySource, registered: boolean) => {
    if (registered) panelOverlays.add(source);
    else panelOverlays.delete(source);
    panelOverlaysChanged.dispatch();
  };
  const sourceElements = () =>
    Array.from(panelElement.firstElementChild!.children) as HTMLElement[];
  return {
    manager,
    setRegistered,
    sourceElements,
    get updateRequests() {
      return updateRequests;
    },
  };
}

describe("PanelOverlayManager", () => {
  it("binds registered sources and requests updates when they signal", () => {
    const fixture = makeFixture();
    const source = new RecordingSource();
    fixture.setRegistered(source, true);
    expect(fixture.sourceElements()).toHaveLength(1);
    fixture.manager.update(() => undefined);
    expect(source.updateCount).toBe(1);

    const before = fixture.updateRequests;
    source.overlayUpdateNeeded.dispatch();
    expect(fixture.updateRequests).toBe(before + 1);

    fixture.setRegistered(source, false);
    expect(fixture.sourceElements()).toHaveLength(0);
    fixture.manager.update(() => undefined);
    expect(source.updateCount).toBe(1);
    fixture.manager.dispose();
  });

  it("hides a source whose overlayVisible is false without updating it", () => {
    const fixture = makeFixture();
    const source = new RecordingSource();
    fixture.setRegistered(source, true);
    const [sourceElement] = fixture.sourceElements();
    source.overlayVisible.value = false;
    fixture.manager.update(() => undefined);
    expect(source.updateCount).toBe(0);
    expect(sourceElement.hidden).toBe(true);
    source.overlayVisible.value = true;
    fixture.manager.update(() => undefined);
    expect(source.updateCount).toBe(1);
    expect(sourceElement.hidden).toBe(false);
    fixture.manager.dispose();
  });
});

describe("PickingIndicatorOverlay", () => {
  it("draws one ring per panel at the projected mouse position", () => {
    const mouseState = {
      active: true,
      position: new Float32Array([10, 20, 30]),
      coordinateSpace: emptyInvalidCoordinateSpace,
      changed: new NullarySignal(),
    } as unknown as MouseSelectionState;
    const overlay = new PickingIndicatorOverlay(
      mouseState,
      new WatchableValue(true),
    );
    const container = document.createElement("div");
    const context: PanelOverlayContext = {
      project: () => ({ x: 100, y: 50, scale: 1.5 }),
      container,
    };
    overlay.updatePanelOverlays(context);
    overlay.updatePanelOverlays(context);
    expect(container.childElementCount).toBe(1);
    const ring = container.firstElementChild as HTMLElement;
    expect(ring.hidden).toBe(false);
    expect(ring.style.width).toBe("21px");
    expect(ring.style.transform).toBe("translate(89.5px, 39.5px)");

    mouseState.active = false;
    overlay.updatePanelOverlays(context);
    expect(ring.hidden).toBe(true);
  });
});
