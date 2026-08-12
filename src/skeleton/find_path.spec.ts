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
  SkeletonDataSourceState,
  type SkeletonFindPathEndpoint,
  SkeletonFindPathState,
} from "#src/skeleton/find_path.js";

function endpoint(
  nodeId: bigint | number,
  segmentId: bigint | number = 100n,
  position: ArrayLike<number> = [
    Number(nodeId),
    Number(nodeId) + 1,
    Number(nodeId) + 2,
  ],
): SkeletonFindPathEndpoint {
  return {
    nodeId: BigInt(nodeId),
    segmentId: BigInt(segmentId),
    position: new Float32Array(position),
  };
}

describe("SkeletonFindPathState", () => {
  it("round-trips generic uint64 endpoint identities and ordered geometry", () => {
    const state = new SkeletonFindPathState();
    const largeSegmentId = 9_007_199_254_740_993n;
    state.setEndpoints(
      endpoint(1, largeSegmentId),
      endpoint(4, largeSegmentId),
    );
    const generation = state.beginRequest();
    expect(
      state.completeRequest(generation, [
        { nodeId: 1n, position: new Float32Array([1, 2, 3]) },
        { nodeId: 3n, position: new Float32Array([3, 4, 5]) },
        { nodeId: 4n, position: new Float32Array([4, 5, 6]) },
      ]),
    ).toBe(true);

    const json = state.toJSON();
    expect(json).toEqual({
      source: {
        nodeId: "1",
        segmentId: largeSegmentId.toString(),
        position: [1, 2, 3],
      },
      target: {
        nodeId: "4",
        segmentId: largeSegmentId.toString(),
        position: [4, 5, 6],
      },
      result: [
        { nodeId: "1", position: [1, 2, 3] },
        { nodeId: "3", position: [3, 4, 5] },
        { nodeId: "4", position: [4, 5, 6] },
      ],
    });

    const restored = new SkeletonFindPathState();
    restored.restoreState(json);
    expect(restored.toJSON()).toEqual(json);
    expect(restored.source?.position).toBeInstanceOf(Float32Array);
    expect(restored.result?.map((node) => node.nodeId)).toEqual([1n, 3n, 4n]);
  });

  it("clones inputs and excludes representation-specific runtime fields", () => {
    const state = new SkeletonFindPathState();
    const position = new Float32Array([1, 2, 3]);
    const source = {
      ...endpoint(1, 2, position),
      annotationReference: { id: "runtime-only" },
    };
    state.setSource(source);
    position[0] = 99;

    expect(state.source?.position[0]).toBe(1);
    expect(state.toJSON()).toEqual({
      source: { nodeId: "1", segmentId: "2", position: [1, 2, 3] },
    });
    expect("annotationReference" in state.source!).toBe(false);
  });

  it.each([
    null,
    [],
    { source: { nodeId: "0", segmentId: "1", position: [1, 2, 3] } },
    { source: { nodeId: "01", segmentId: "1", position: [1, 2, 3] } },
    { source: { nodeId: "1", segmentId: "-1", position: [1, 2, 3] } },
    {
      source: {
        nodeId: "18446744073709551616",
        segmentId: "1",
        position: [1, 2, 3],
      },
    },
    { source: { nodeId: "1", segmentId: "1", position: [1, 2] } },
    { source: { nodeId: "1", segmentId: "1", position: [1, NaN, 3] } },
    {
      source: { nodeId: "1", segmentId: "1", position: [1, 2, 3] },
      target: { nodeId: "1", segmentId: "1", position: [4, 5, 6] },
    },
    {
      source: { nodeId: "1", segmentId: "1", position: [1, 2, 3] },
      target: { nodeId: "2", segmentId: "2", position: [4, 5, 6] },
    },
    { result: {} },
    { result: [] },
    {
      source: { nodeId: "1", segmentId: "1", position: [1, 2, 3] },
      target: { nodeId: "2", segmentId: "1", position: [4, 5, 6] },
      result: [{ nodeId: "2", position: [4, 5, 6] }],
    },
  ])("rejects malformed serialized state %# atomically", (json) => {
    const state = new SkeletonFindPathState();
    state.setSource(endpoint(7));
    const before = state.toJSON();

    expect(() => state.restoreState(json)).toThrow();
    expect(state.toJSON()).toEqual(before);
  });

  it("endpoint changes clear results and invalidate pending completions", () => {
    const state = new SkeletonFindPathState();
    state.setEndpoints(endpoint(1), endpoint(2));
    const first = state.beginRequest();
    state.completeRequest(first, [endpoint(1), endpoint(2)]);
    expect(state.result).toHaveLength(2);

    const pending = state.beginRequest();
    state.setTarget(endpoint(3));
    expect(state.result).toBeUndefined();
    expect(state.isRequestCurrent(pending)).toBe(false);
    expect(state.completeRequest(pending, [endpoint(1), endpoint(3)])).toBe(
      false,
    );
  });

  it("uses latest-request-wins tokens", () => {
    const state = new SkeletonFindPathState();
    state.setEndpoints(endpoint(1), endpoint(2));
    const first = state.beginRequest();
    const second = state.beginRequest();

    expect(state.isRequestCurrent(first)).toBe(false);
    expect(state.failRequest(first)).toBe(false);
    expect(state.completeRequest(second, [endpoint(1), endpoint(2)])).toBe(
      true,
    );
    expect(state.result?.map((node) => node.nodeId)).toEqual([1n, 2n]);
  });

  it("invalidates only the result when topology changes", () => {
    const state = new SkeletonFindPathState();
    state.setEndpoints(endpoint(1), endpoint(2));
    state.completeRequest(state.beginRequest(), [endpoint(1), endpoint(2)]);

    expect(state.invalidateResult()).toBe(true);
    expect(state.source?.nodeId).toBe(1n);
    expect(state.target?.nodeId).toBe(2n);
    expect(state.result).toBeUndefined();
  });

  it("clear and reset return the state to its default", () => {
    const state = new SkeletonFindPathState();
    state.setEndpoints(endpoint(1), endpoint(2));
    const generation = state.beginRequest();

    expect(state.clear()).toBe(true);
    expect(state.isRequestCurrent(generation)).toBe(false);
    expect(state.toJSON()).toBeUndefined();
    expect(state.clear()).toBe(false);

    state.setSource(endpoint(3));
    state.reset();
    expect(state.toJSON()).toBeUndefined();
  });

  it("enforces endpoint and result relationships", () => {
    const state = new SkeletonFindPathState();
    expect(() => state.setEndpoints(endpoint(1), endpoint(1))).toThrow(
      /distinct/i,
    );
    expect(() => state.setEndpoints(endpoint(1, 10), endpoint(2, 11))).toThrow(
      /same segment/i,
    );

    state.setEndpoints(endpoint(1), endpoint(3));
    const generation = state.beginRequest();
    expect(() => state.completeRequest(generation, [])).toThrow(
      /at least one/i,
    );
    expect(() =>
      state.completeRequest(generation, [endpoint(2), endpoint(3)]),
    ).toThrow(/start at the source/i);
    expect(state.completeRequest(generation, [endpoint(1), endpoint(3)])).toBe(
      true,
    );
  });

  it("allows one endpoint to remain as a debugging marker", () => {
    const state = new SkeletonFindPathState();
    state.setEndpoints(endpoint(1), endpoint(2));
    state.setSource(undefined);

    const restored = new SkeletonFindPathState();
    restored.restoreState(state.toJSON());
    expect(restored.source).toBeUndefined();
    expect(restored.target?.nodeId).toBe(2n);
  });
});

describe("SkeletonDataSourceState", () => {
  it("round-trips Find Path under the datasource-owned findPath key", () => {
    const state = new SkeletonDataSourceState();
    state.findPathState.setEndpoints(endpoint(1, 7), endpoint(3, 7));

    expect(state.toJSON()).toEqual({
      findPath: {
        source: { nodeId: "1", segmentId: "7", position: [1, 2, 3] },
        target: { nodeId: "3", segmentId: "7", position: [3, 4, 5] },
      },
    });

    const restored = new SkeletonDataSourceState(state.toJSON());
    expect(restored.toJSON()).toEqual(state.toJSON());
  });

  it("forwards nested changes and never serializes runtime request state", () => {
    const state = new SkeletonDataSourceState();
    let changes = 0;
    state.changed.add(() => ++changes);
    state.findPathState.setEndpoints(endpoint(1), endpoint(2));
    state.findPathState.beginRequest();

    expect(changes).toBe(2);
    expect(state.toJSON()).toEqual({
      findPath: {
        source: { nodeId: "1", segmentId: "100", position: [1, 2, 3] },
        target: { nodeId: "2", segmentId: "100", position: [2, 3, 4] },
      },
    });
  });

  it("rejects malformed datasource state atomically", () => {
    const state = new SkeletonDataSourceState();
    state.findPathState.setSource(endpoint(7));
    const before = state.toJSON();

    expect(() =>
      state.restoreState({
        findPath: {
          source: { nodeId: "0", segmentId: "1", position: [1, 2, 3] },
        },
      }),
    ).toThrow(/nodeId/);
    expect(state.toJSON()).toEqual(before);
  });
});
