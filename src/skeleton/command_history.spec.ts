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

import {
  SpatialSkeletonCommandHistory,
  type SpatialSkeletonCommand,
} from "#src/skeleton/command_history.js";

function deferred() {
  let resolve: (() => void) | undefined;
  const promise = new Promise<void>((innerResolve) => {
    resolve = innerResolve;
  });
  return {
    promise,
    resolve: () => resolve?.(),
  };
}

describe("skeleton/command_history", () => {
  it("serializes execution, updates labels, and truncates redo on a new edit", async () => {
    const history = new SpatialSkeletonCommandHistory();
    const events: string[] = [];
    const firstExecute = deferred();
    const firstCommand: SpatialSkeletonCommand = {
      label: "First command",
      execute: async () => {
        events.push("first:start");
        await firstExecute.promise;
        events.push("first:end");
      },
      undo: async () => {
        events.push("first:undo");
      },
    };
    const secondCommand: SpatialSkeletonCommand = {
      label: "Second command",
      execute: async () => {
        events.push("second:execute");
      },
      undo: async () => {
        events.push("second:undo");
      },
    };
    const thirdCommand: SpatialSkeletonCommand = {
      label: "Third command",
      execute: async () => {
        events.push("third:execute");
      },
      undo: async () => {
        events.push("third:undo");
      },
    };

    const firstPromise = history.execute(firstCommand);
    const secondPromise = history.execute(secondCommand);

    expect(history.isBusy.value).toBe(true);
    expect(events).toEqual(["first:start"]);

    firstExecute.resolve();
    await firstPromise;
    await secondPromise;

    expect(events).toEqual(["first:start", "first:end", "second:execute"]);
    expect(history.canUndo.value).toBe(true);
    expect(history.undoLabel.value).toBe("Second command");

    await history.undo();

    expect(events).toEqual([
      "first:start",
      "first:end",
      "second:execute",
      "second:undo",
    ]);
    expect(history.canRedo.value).toBe(true);
    expect(history.redoLabel.value).toBe("Second command");

    await history.execute(thirdCommand);

    expect(events).toEqual([
      "first:start",
      "first:end",
      "second:execute",
      "second:undo",
      "third:execute",
    ]);
    expect(history.canRedo.value).toBe(false);
    expect(history.redoLabel.value).toBeUndefined();
    expect(history.undoLabel.value).toBe("Third command");
  });

  it("keeps remapped node and segment ids across undo and redo", async () => {
    const history = new SpatialSkeletonCommandHistory();
    let nextNodeId = 100n;
    let nextSegmentId = 200n;
    const command: SpatialSkeletonCommand = {
      label: "Remap ids",
      execute: async ({ mappings }) => {
        mappings.remapNodeId(11n, nextNodeId++);
        mappings.remapSegmentId(21n, nextSegmentId++);
      },
      undo: async ({ mappings }) => {
        mappings.remapNodeId(11n, nextNodeId++);
        mappings.remapSegmentId(21n, nextSegmentId++);
      },
    };

    await history.execute(command);
    expect(history.mappings.resolveNodeId(11n)).toBe(100n);
    expect(history.mappings.resolveSegmentId(21n)).toBe(200n);
    expect(history.mappings.getStableNodeId(100n)).toBe(11n);
    expect(history.mappings.getStableSegmentId(200n)).toBe(21n);

    await history.undo();
    expect(history.mappings.resolveNodeId(11n)).toBe(101n);
    expect(history.mappings.resolveSegmentId(21n)).toBe(201n);

    await history.redo();
    expect(history.mappings.resolveNodeId(11n)).toBe(102n);
    expect(history.mappings.resolveSegmentId(21n)).toBe(202n);
  });

  it("restores mapping state if an operation fails", async () => {
    const history = new SpatialSkeletonCommandHistory();
    history.mappings.remapNodeId(11n, 99n);

    const failingCommand: SpatialSkeletonCommand = {
      label: "Failing command",
      execute: async ({ mappings }) => {
        mappings.remapNodeId(11n, 100n);
        throw new Error("boom");
      },
      undo: async () => {},
    };

    await expect(history.execute(failingCommand)).rejects.toThrow("boom");
    expect(history.mappings.resolveNodeId(11n)).toBe(99n);
    expect(history.canUndo.value).toBe(false);
  });
});
