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

import { afterEach, describe, expect, it, vi } from "vitest";
import {
  EventActionMap,
  friendlyEventIdentifier,
  registerActionListener,
} from "#src/util/event_action_map.js";
import { KeyboardEventBinder } from "#src/util/keyboard_bindings.js";

describe("friendlyEventIdentifier", () => {
  it("strips the phase along with its colon", () => {
    expect(friendlyEventIdentifier("at:control+mousedown0")).toBe(
      "control+mousedown0",
    );
  });
});

describe("mac control bindings", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  function pressKey(platform: string, init: KeyboardEventInit): string[] {
    vi.stubGlobal("navigator", { platform });
    const element = document.createElement("div");
    document.body.appendChild(element);
    const binder = new KeyboardEventBinder(
      element,
      EventActionMap.fromObject({ "control+keya": "do-thing" }),
    );
    const dispatched: string[] = [];
    const listener = registerActionListener(element, "do-thing", () => {
      dispatched.push("do-thing");
    });
    try {
      element.dispatchEvent(
        new KeyboardEvent("keydown", { bubbles: true, ...init }),
      );
    } finally {
      listener();
      binder.dispose();
      element.remove();
    }
    return dispatched;
  }

  it("binds a control stroke to Command on Mac", () => {
    expect(pressKey("MacIntel", { code: "KeyA", metaKey: true })).toEqual([
      "do-thing",
    ]);
  });

  it("leaves Control free for the system secondary click on Mac", () => {
    expect(pressKey("MacIntel", { code: "KeyA", ctrlKey: true })).toEqual([]);
  });

  it("binds a control stroke to Control off Mac", () => {
    expect(pressKey("Win32", { code: "KeyA", ctrlKey: true })).toEqual([
      "do-thing",
    ]);
    expect(pressKey("Win32", { code: "KeyA", metaKey: true })).toEqual([]);
  });
});
