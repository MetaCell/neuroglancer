import { describe, expect, it } from "vitest";

import {
  committedAddSourcesSatisfied,
  committedMoveSourcesSatisfied,
  computeSpatiallyIndexedOwnerChunkKey,
} from "#src/skeleton/spatial_reconciliation.js";

describe("skeleton/spatial_reconciliation", () => {
  it("computes the owner chunk key from chunk-space coordinates", () => {
    const transform = new Float32Array([
      1, 0, 0, 0,
      0, 1, 0, 0,
      0, 0, 1, 0,
      0, 0, 0, 1,
    ]);
    expect(
      computeSpatiallyIndexedOwnerChunkKey(transform, [1.9, 3.2, 5.99], 3, 4),
    ).toBe("1,3,5:4");
  });

  it("requires both old and new move owners to reconcile", () => {
    expect(
      committedMoveSourcesSatisfied([
        {
          oldOwnerSatisfied: true,
          newOwnerSatisfied: false,
        },
      ]),
    ).toBe(false);
    expect(
      committedMoveSourcesSatisfied([
        {
          oldOwnerSatisfied: true,
          newOwnerSatisfied: true,
        },
        {
          oldOwnerSatisfied: true,
          newOwnerSatisfied: true,
        },
      ]),
    ).toBe(true);
  });

  it("requires every add owner chunk to reconcile", () => {
    expect(
      committedAddSourcesSatisfied([
        {
          ownerSatisfied: true,
        },
        {
          ownerSatisfied: false,
        },
      ]),
    ).toBe(false);
    expect(
      committedAddSourcesSatisfied([
        {
          ownerSatisfied: true,
        },
      ]),
    ).toBe(true);
  });
});
