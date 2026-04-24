import { describe, expect, it } from "vitest";

import {
  DEFAULT_MAX_RETAINED_OVERLAY_SEGMENTS,
  mergeSpatiallyIndexedSkeletonOverlaySegmentIds,
  retainSpatiallyIndexedSkeletonOverlaySegment,
} from "#src/skeleton/overlay_segment_retention.js";

describe("mergeSpatiallyIndexedSkeletonOverlaySegmentIds", () => {
  it("dedupes and sorts active and retained segment ids", () => {
    expect(
      mergeSpatiallyIndexedSkeletonOverlaySegmentIds([7, 3, 7], [9, 3, 5]),
    ).toEqual([3, 5, 7, 9]);
  });

  it("ignores invalid segment ids", () => {
    expect(
      mergeSpatiallyIndexedSkeletonOverlaySegmentIds([1, 0, -2], [NaN, 4]),
    ).toEqual([1, 4]);
  });
});

describe("retainSpatiallyIndexedSkeletonOverlaySegment", () => {
  it("moves retained segments to the most recent position", () => {
    expect(retainSpatiallyIndexedSkeletonOverlaySegment([2, 4, 6], 4)).toEqual([
      2, 6, 4,
    ]);
  });

  it("keeps only the most recent retained segments", () => {
    const retained: number[] = [];
    for (
      let segmentId = 1;
      segmentId <= DEFAULT_MAX_RETAINED_OVERLAY_SEGMENTS + 2;
      ++segmentId
    ) {
      retained.splice(
        0,
        retained.length,
        ...retainSpatiallyIndexedSkeletonOverlaySegment(retained, segmentId),
      );
    }
    const firstRetainedSegmentId = 3;
    expect(retained).toEqual(
      Array.from(
        { length: DEFAULT_MAX_RETAINED_OVERLAY_SEGMENTS },
        (_, index) => firstRetainedSegmentId + index,
      ),
    );
  });
});
