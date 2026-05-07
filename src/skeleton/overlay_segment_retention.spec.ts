import { describe, expect, it } from "vitest";

import {
  DEFAULT_MAX_RETAINED_OVERLAY_SEGMENTS,
  mergeSpatiallyIndexedSkeletonOverlaySegmentIds,
  retainSpatiallyIndexedSkeletonOverlaySegment,
} from "#src/skeleton/overlay_segment_retention.js";

describe("mergeSpatiallyIndexedSkeletonOverlaySegmentIds", () => {
  it("dedupes and sorts active and retained segment ids", () => {
    expect(
      mergeSpatiallyIndexedSkeletonOverlaySegmentIds(
        [7n, 3n, 7n],
        [9n, 3n, 5n],
      ),
    ).toEqual([3n, 5n, 7n, 9n]);
  });

  it("ignores invalid segment ids", () => {
    expect(
      mergeSpatiallyIndexedSkeletonOverlaySegmentIds([1n, 0n], [4n]),
    ).toEqual([1n, 4n]);
  });
});

describe("retainSpatiallyIndexedSkeletonOverlaySegment", () => {
  it("moves retained segments to the most recent position", () => {
    expect(
      retainSpatiallyIndexedSkeletonOverlaySegment([2n, 4n, 6n], 4n),
    ).toEqual([2n, 6n, 4n]);
  });

  it("keeps only the most recent retained segments", () => {
    const retained: bigint[] = [];
    for (
      let segmentId = 1;
      segmentId <= DEFAULT_MAX_RETAINED_OVERLAY_SEGMENTS + 2;
      ++segmentId
    ) {
      retained.splice(
        0,
        retained.length,
        ...retainSpatiallyIndexedSkeletonOverlaySegment(
          retained,
          BigInt(segmentId),
        ),
      );
    }
    const firstRetainedSegmentId = 3n;
    expect(retained).toEqual(
      Array.from(
        { length: DEFAULT_MAX_RETAINED_OVERLAY_SEGMENTS },
        (_, index) => firstRetainedSegmentId + BigInt(index),
      ),
    );
  });
});
