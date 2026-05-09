import { describe, expect, it } from "vitest";

import { filterSpatiallyIndexedSkeletonEntriesByView } from "#src/skeleton/source_selection.js";

describe("skeleton/source_selection", () => {
  it("keeps entries for the requested view", () => {
    const entries = [
      { id: "slice", view: "2d" },
      { id: "perspective", view: "3d" },
      { id: "shared" },
    ];
    expect(
      filterSpatiallyIndexedSkeletonEntriesByView(
        entries,
        "2d",
        (entry) => entry.view,
      ),
    ).toEqual([entries[0], entries[2]]);
  });

  it("keeps shared entries for every view", () => {
    const entries = [{ id: "slice", view: "2d" }, { id: "shared" }];
    expect(
      filterSpatiallyIndexedSkeletonEntriesByView(
        entries,
        "3d",
        (entry) => entry.view,
      ),
    ).toEqual([entries[1]]);
  });
});
