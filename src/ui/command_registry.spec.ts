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
import { CommandRegistry } from "#src/ui/command_registry.js";
import { WatchableValue } from "#src/trackable_value.js";

describe("CommandRegistry", () => {
  it("registers and retrieves a command by id", () => {
    const registry = new CommandRegistry();
    registry.registerAction({
      id: "screenshot",
      label: "Screenshot",
      description: "Capture a screenshot.",
    });
    expect(registry.has("screenshot")).toBe(true);
    expect(registry.get("screenshot")?.label).toBe("Screenshot");
    registry.dispose();
  });

  it("stamps the command type explicitly per registrar", () => {
    const registry = new CommandRegistry();
    registry.registerAction({ id: "a", label: "A" });
    registry.registerCallback({ id: "c", label: "C", invoke: () => {} });
    expect(registry.get("a")?.type).toBe("action");
    expect(registry.get("c")?.type).toBe("callback");
    registry.dispose();
  });

  it("invokes a callback command's callback", () => {
    const registry = new CommandRegistry();
    let ran = false;
    registry.registerCallback({
      id: "c",
      label: "C",
      invoke: () => {
        ran = true;
      },
    });
    const command = registry.get("c");
    if (command?.type === "callback") command.invoke();
    expect(ran).toBe(true);
    registry.dispose();
  });

  it("enumerates commands independent of any binding", () => {
    const registry = new CommandRegistry();
    registry.registerAction({ id: "a", label: "A" });
    registry.registerAction({ id: "b", label: "B" });
    expect([...registry.values()].map((c) => c.id)).toStrictEqual(["a", "b"]);
    registry.dispose();
  });

  it("throws on duplicate id", () => {
    const registry = new CommandRegistry();
    registry.registerAction({ id: "dup", label: "First" });
    expect(() =>
      registry.registerAction({ id: "dup", label: "Second" }),
    ).toThrow(/already registered/);
    registry.dispose();
  });

  it("unregisters via the returned disposer", () => {
    const registry = new CommandRegistry();
    const dispose = registry.registerAction({ id: "temp", label: "Temp" });
    expect(registry.has("temp")).toBe(true);
    dispose();
    expect(registry.has("temp")).toBe(false);
    registry.dispose();
  });

  it("dispatches changed on register and unregister", () => {
    const registry = new CommandRegistry();
    let count = 0;
    registry.changed.add(() => ++count);
    const dispose = registry.registerAction({ id: "x", label: "X" });
    expect(count).toBe(1);
    dispose();
    expect(count).toBe(2);
    registry.dispose();
  });

  it("dispatches changed when a command's availability changes", () => {
    const registry = new CommandRegistry();
    const isAvailable = new WatchableValue(true);
    registry.registerAction({ id: "x", label: "X", isAvailable });
    let count = 0;
    registry.changed.add(() => ++count);
    isAvailable.value = false;
    expect(count).toBe(1);
    registry.dispose();
  });

  it("stops observing availability after unregister", () => {
    const registry = new CommandRegistry();
    const isAvailable = new WatchableValue(true);
    const dispose = registry.registerAction({
      id: "x",
      label: "X",
      isAvailable,
    });
    dispose();
    let count = 0;
    registry.changed.add(() => ++count);
    isAvailable.value = false;
    expect(count).toBe(0);
    registry.dispose();
  });
});
