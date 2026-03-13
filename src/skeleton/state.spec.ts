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

import { SpatialSkeletonState } from "#src/skeleton/state.js";

describe("skeleton/state", () => {
  it("clears the full skeleton cache before notifying node data listeners", () => {
    const state = new SpatialSkeletonState();
    const cachedSegmentId = 11;
    (state as any).fullSegmentNodeCache.set(cachedSegmentId, [
      {
        nodeId: 1,
        segmentId: cachedSegmentId,
        position: new Float32Array([1, 2, 3]),
      },
    ]);

    let cachePresentDuringNotification: boolean | undefined;
    state.nodeDataVersion.changed.add(() => {
      cachePresentDuringNotification = (state as any).fullSegmentNodeCache.has(
        cachedSegmentId,
      );
    });

    state.markNodeDataChanged();

    expect(cachePresentDuringNotification).toBe(false);
    expect((state as any).fullSegmentNodeCache.has(cachedSegmentId)).toBe(
      false,
    );
  });
});
