import { describe, expect, it } from "vitest";

import { selectSpatiallyIndexedSkeletonEntriesByGrid } from "#src/skeleton/source_selection.js";

describe("skeleton/source_selection", () => {
  it("returns the exact grid match when available", () => {
    const entries = [
      { id: "coarse", gridIndex: 0 },
      { id: "medium", gridIndex: 2 },
      { id: "fine", gridIndex: 4 },
    ];
    expect(
      selectSpatiallyIndexedSkeletonEntriesByGrid(
        entries,
        2,
        (entry) => entry.gridIndex,
      ),
    ).toEqual([entries[1]]);
  });

  it("returns the nearest grid match and keeps the first entry on ties", () => {
    const entries = [
      { id: "left", gridIndex: 0 },
      { id: "right", gridIndex: 4 },
    ];
    expect(
      selectSpatiallyIndexedSkeletonEntriesByGrid(
        entries,
        2,
        (entry) => entry.gridIndex,
      ),
    ).toEqual([entries[0]]);
  });

  it("returns all entries if any entry is missing a grid index", () => {
    const entries = [
      { id: "indexed", gridIndex: 0 },
      { id: "unindexed" },
      { id: "indexed-2", gridIndex: 2 },
    ];
    expect(
      selectSpatiallyIndexedSkeletonEntriesByGrid(
        entries,
        1,
        (entry) => entry.gridIndex,
      ),
    ).toEqual(entries);
  });
});
