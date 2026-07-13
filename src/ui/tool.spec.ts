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
  GlobalToolBinder,
  LocalToolBinder,
  getMatchingTools,
  registerTool,
  restoreTool,
  type InputEventMapBinder,
  type Tool,
} from "#src/ui/tool.js";
import type { MultiToolPaletteState } from "#src/ui/tool_palette.js";
import { INCLUDE_EVERYTHING_QUERY } from "#src/ui/tool_query.js";

// Mirrors the real situation where the same tool `type` string is
// registered against two distinct, unrelated context classes (e.g. the
// dimension tool is registered separately for `Viewer` and for every
// `LayerGroupViewer`)
class FakeContextA {}
class FakeContextB {}

const TOOL_TYPE = "fake-shared-tool";

registerTool(
  FakeContextA,
  TOOL_TYPE,
  (context) =>
    ({ dispose() {}, description: "from-A", context }) as unknown as Tool,
  () => [{ type: TOOL_TYPE, source: "A" }],
);
registerTool(
  FakeContextB,
  TOOL_TYPE,
  (context) =>
    ({ dispose() {}, description: "from-B", context }) as unknown as Tool,
  () => [{ type: TOOL_TYPE, source: "B" }],
);

function makeGlobalBinder() {
  const noopInputEventMapBinder: InputEventMapBinder = () => {};
  return new GlobalToolBinder(
    noopInputEventMapBinder,
    {} as unknown as MultiToolPaletteState,
  );
}

describe("getMatchingTools", () => {
  it("preserves the originating context for each match, per local binder", () => {
    const globalBinder = makeGlobalBinder();
    const contextA = new FakeContextA();
    const contextB = new FakeContextB();
    new LocalToolBinder(contextA, globalBinder);
    new LocalToolBinder(contextB, globalBinder);

    const matches = getMatchingTools(globalBinder, INCLUDE_EVERYTHING_QUERY);
    expect(matches.size).toBe(2);

    const contexts = Array.from(matches.values(), (match) => match.context);
    expect(contexts).toContain(contextA);
    expect(contexts).toContain(contextB);

    for (const match of matches.values()) {
      const tool = restoreTool(match.context, match.toolJson) as unknown as {
        description: string;
        context: unknown;
      };
      // Reconstructing with the match's own context must resolve back to
      // the same context's registration
      expect(tool.context).toBe(match.context);
      expect(tool.description).toBe(
        match.context === contextA ? "from-A" : "from-B",
      );
    }
  });
});
