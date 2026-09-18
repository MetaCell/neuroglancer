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
import { formatKeyStroke } from "#src/ui/command.js";

describe("formatKeyStroke", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  function format(platform: string, stroke: string) {
    vi.stubGlobal("navigator", { platform });
    return formatKeyStroke(stroke);
  }

  it("joins modifier names off Mac", () => {
    expect(format("Win32", "control+alt+keyp")).toBe("control+alt+p");
  });

  it("uses platform symbols on Mac", () => {
    expect(format("MacIntel", "control+keyp")).toBe("⌘p");
    expect(format("MacIntel", "alt+shift+keya")).toBe("⌥⇧a");
    expect(format("MacIntel", "meta+keya")).toBe("⌘a");
  });

  it("keeps a separator before a multi-character key name", () => {
    expect(format("MacIntel", "control+mousedown0")).toBe("⌘+mousedown0");
  });
});
