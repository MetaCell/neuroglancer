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
import type { SwcSegments } from "#src/datasource/swc/segment_ids.js";
import { assignSegmentIds, fnv1a64 } from "#src/datasource/swc/segment_ids.js";

function getIdByFileName(segments: SwcSegments) {
  return new Map(
    Array.from(segments.ids, (segmentId, index) => [
      segments.fileNames[index],
      segmentId,
    ]),
  );
}

describe("datasource/swc/segment_ids", () => {
  it("gives a file the same ID in every release", () => {
    expect(assignSegmentIds(["cell_1.swc"], fnv1a64).ids).toEqual(
      BigUint64Array.of(6963899503464382429n),
    );
  });

  it("gives a file the same ID whatever order the folder is listed in", () => {
    expect(
      getIdByFileName(assignSegmentIds(["b.swc", "a.swc", "c.swc"], fnv1a64)),
    ).toEqual(
      getIdByFileName(assignSegmentIds(["c.swc", "a.swc", "b.swc"], fnv1a64)),
    );
  });

  it("keeps the ID of every existing file when a new file with a different hash is added", () => {
    const before = getIdByFileName(
      assignSegmentIds(["cell_1.swc", "cell_2.swc"], fnv1a64),
    );
    const after = getIdByFileName(
      assignSegmentIds(["cell_1.swc", "cell_2.swc", "cell_0.swc"], fnv1a64),
    );
    expect(after.get("cell_1.swc")).toBe(before.get("cell_1.swc"));
    expect(after.get("cell_2.swc")).toBe(before.get("cell_2.swc"));
  });

  it("keeps the ID of a file when the case of its .swc extension changes", () => {
    expect(assignSegmentIds(["cell_1.SWC"], fnv1a64).ids).toEqual(
      assignSegmentIds(["cell_1.swc"], fnv1a64).ids,
    );
  });

  it("gives files whose names hash alike different IDs, and the name that sorts first keeps its hash", () => {
    const collidingHash = (key: string) =>
      key === "a" || key === "b" ? 5n : fnv1a64(key);
    expect(
      getIdByFileName(assignSegmentIds(["b.swc", "a.swc"], collidingHash)),
    ).toEqual(
      new Map([
        ["a.swc", 5n],
        ["b.swc", fnv1a64("b\u00001")],
      ]),
    );
  });

  it("never gives a file segment ID 0", () => {
    const zeroHash = (key: string) => (key === "a" ? 0n : fnv1a64(key));
    expect(assignSegmentIds(["a.swc"], zeroHash).ids).toEqual(
      BigUint64Array.of(fnv1a64("a\u00001")),
    );
  });
});
