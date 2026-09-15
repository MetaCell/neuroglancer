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

import { describe, expect, it } from "vitest";
import type { CoordinateSpace } from "#src/coordinate_transform.js";
import { emptyInvalidCoordinateSpace } from "#src/coordinate_transform.js";
import type { DisplayContext } from "#src/display_context.js";
import type { MouseSelectionState } from "#src/layer/index.js";
import type {
  PanelOverlayContext,
  PanelOverlaySource,
  ProjectedPosition,
} from "#src/panel_overlay.js";
import { PanelOverlayManager } from "#src/panel_overlay.js";
import { PickingIndicatorOverlay } from "#src/picking_indicator_overlay.js";
import { WatchableValue } from "#src/trackable_value.js";
import { NullarySignal } from "#src/util/signal.js";

class RecordingSource implements PanelOverlaySource {
  readonly overlayUpdateNeeded = new NullarySignal();
  readonly overlayVisible = new WatchableValue(true);
  updateCount = 0;
  lastContext: PanelOverlayContext | undefined;
  constructor(readonly overlayPriority: number = 0) {}
  updatePanelOverlays(context: PanelOverlayContext) {
    ++this.updateCount;
    this.lastContext = context;
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
  const element = document.createElement("div");
  const manager = new PanelOverlayManager(
    {
      element,
      projectPosition: (position: Float32Array) => ({
        x: position[0],
        y: position[1],
      }),
    },
    context,
  );
  const register = (source: PanelOverlaySource) => {
    panelOverlays.add(source);
    panelOverlaysChanged.dispatch();
  };
  const unregister = (source: PanelOverlaySource) => {
    panelOverlays.delete(source);
    panelOverlaysChanged.dispatch();
  };
  const sourceElements = () =>
    Array.from(element.firstElementChild!.children) as HTMLElement[];
  return {
    manager,
    register,
    unregister,
    sourceElements,
    get updateRequests() {
      return updateRequests;
    },
  };
}

describe("PanelOverlayManager", () => {
  it("binds registered sources into prioritized elements and updates them", () => {
    const fixture = makeFixture();
    const source = new RecordingSource(7);
    fixture.register(source);
    const [sourceElement] = fixture.sourceElements();
    expect(sourceElement.style.zIndex).toBe("7");

    fixture.manager.update();
    expect(source.updateCount).toBe(1);
    expect(source.lastContext!.container).toBe(sourceElement);
    expect(
      source.lastContext!.project(
        new Float32Array([3, 4]),
        emptyInvalidCoordinateSpace,
      ),
    ).toEqual({ x: 3, y: 4 });

    fixture.unregister(source);
    expect(fixture.sourceElements()).toHaveLength(0);
    fixture.manager.update();
    expect(source.updateCount).toBe(1);
    fixture.manager.dispose();
  });

  it("requests an update when a source signals or changes visibility", () => {
    const fixture = makeFixture();
    const source = new RecordingSource();
    fixture.register(source);
    const afterRegister = fixture.updateRequests;
    source.overlayUpdateNeeded.dispatch();
    expect(fixture.updateRequests).toBe(afterRegister + 1);
    source.overlayVisible.value = false;
    expect(fixture.updateRequests).toBe(afterRegister + 2);
    fixture.manager.dispose();
    source.overlayUpdateNeeded.dispatch();
    expect(fixture.updateRequests).toBe(afterRegister + 2);
  });

  it("hides a source whose overlayVisible is false without updating it", () => {
    const fixture = makeFixture();
    const source = new RecordingSource();
    fixture.register(source);
    const [sourceElement] = fixture.sourceElements();

    source.overlayVisible.value = false;
    fixture.manager.update();
    expect(source.updateCount).toBe(0);
    expect(sourceElement.hidden).toBe(true);

    source.overlayVisible.value = true;
    fixture.manager.update();
    expect(source.updateCount).toBe(1);
    expect(sourceElement.hidden).toBe(false);

    fixture.manager.clear();
    expect(sourceElement.hidden).toBe(true);
    fixture.manager.dispose();
  });
});

describe("PickingIndicatorOverlay", () => {
  function makeContext(projected: ProjectedPosition | undefined) {
    const container = document.createElement("div");
    const projectCalls: CoordinateSpace[] = [];
    const context: PanelOverlayContext = {
      project: (_position, coordinateSpace) => {
        projectCalls.push(coordinateSpace);
        return projected;
      },
      container,
    };
    return { container, context, projectCalls };
  }

  function makeMouseState(active: boolean) {
    return {
      active,
      position: new Float32Array([10, 20, 30]),
      coordinateSpace: emptyInvalidCoordinateSpace,
      changed: new NullarySignal(),
    } as unknown as MouseSelectionState;
  }

  it("positions one ring per container at the projected position", () => {
    const mouseState = makeMouseState(true);
    const overlay = new PickingIndicatorOverlay(
      mouseState,
      new WatchableValue(true),
    );
    const { container, context, projectCalls } = makeContext({
      x: 100,
      y: 50,
      scale: 1.5,
    });
    overlay.updatePanelOverlays(context);
    overlay.updatePanelOverlays(context);
    expect(container.childElementCount).toBe(1);
    expect(projectCalls).toEqual([
      emptyInvalidCoordinateSpace,
      emptyInvalidCoordinateSpace,
    ]);
    const ring = container.firstElementChild as HTMLElement;
    expect(ring.className).toBe("neuroglancer-picking-indicator");
    expect(ring.hidden).toBe(false);
    expect(ring.style.width).toBe("21px");
    expect(ring.style.transform).toBe("translate(89.5px, 39.5px)");

    mouseState.active = false;
    overlay.updatePanelOverlays(context);
    expect(ring.hidden).toBe(true);
  });

  it("clamps the depth scale", () => {
    const overlay = new PickingIndicatorOverlay(
      makeMouseState(true),
      new WatchableValue(true),
    );
    const large = makeContext({ x: 0, y: 0, scale: 10 });
    overlay.updatePanelOverlays(large.context);
    expect((large.container.firstElementChild as HTMLElement).style.width).toBe(
      "23.8px",
    );
    const small = makeContext({ x: 0, y: 0, scale: 0.01 });
    overlay.updatePanelOverlays(small.context);
    expect((small.container.firstElementChild as HTMLElement).style.width).toBe(
      "8.4px",
    );
  });

  it("hides the ring when the position does not project", () => {
    const overlay = new PickingIndicatorOverlay(
      makeMouseState(true),
      new WatchableValue(true),
    );
    const { container, context } = makeContext(undefined);
    overlay.updatePanelOverlays(context);
    expect((container.firstElementChild as HTMLElement).hidden).toBe(true);
  });
});
