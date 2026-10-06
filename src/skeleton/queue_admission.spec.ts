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

import { describe, expect, it, vi } from "vitest";

import type {
  SpatialSkeletonEditCommand,
  SpatialSkeletonQueueInput,
  SpatialSkeletonQueueInputRequirement,
} from "#src/skeleton/command_protocol.js";
import { SpatialSkeletonActions } from "#src/skeleton/command_protocol.js";
import { SpatialSkeletonInspectionRequiredError } from "#src/skeleton/edit_errors.js";
import {
  SpatialSkeletonOptimisticReloadRequiredError,
  type SpatialSkeletonOptimisticFatalState,
} from "#src/skeleton/optimistic_edit/fatal.js";
import type { SpatialSkeletonOptimisticEditSettlement } from "#src/skeleton/optimistic_edit/lifecycle.js";
import type { SpatialSkeletonOptimisticEditExecution } from "#src/skeleton/optimistic_edit/types.js";
import { prepareAndSubmitSpatialSkeletonEdit } from "#src/skeleton/queue_admission.js";
import { createDeferred } from "#src/util/promise.js";

function optimisticExecution<T>(
  execution: Promise<T>,
  settled: Promise<SpatialSkeletonOptimisticEditSettlement> = Promise.resolve({
    outcome: "committed",
  }),
): SpatialSkeletonOptimisticEditExecution<T> {
  Object.defineProperties(execution, {
    acceptedByQueue: { configurable: true, value: Promise.resolve() },
    settled: { configurable: true, value: settled },
  });
  return execution as SpatialSkeletonOptimisticEditExecution<T>;
}

function makeHarness(initialSegments: readonly number[] = []) {
  const cachedSegments = new Set(initialSegments);
  const revisions = new Map(initialSegments.map((segmentId) => [segmentId, 1]));
  const snapshotHandles = new Map(
    initialSegments.map((segmentId) => [segmentId, { segmentId }]),
  );
  const releases: number[] = [];
  const getFullSegmentNodes = vi.fn(async (_layer, segmentId: number) => {
    cachedSegments.add(segmentId);
    return [];
  });
  let fatalState: SpatialSkeletonOptimisticFatalState | undefined;
  const state = {
    getOptimisticEditingIdentityService: () => ({}),
    tryAcquireInputReference(
      requirement: SpatialSkeletonQueueInputRequirement,
    ) {
      if (!cachedSegments.has(requirement.segmentId)) return undefined;
      const revision = revisions.get(requirement.segmentId) ?? 0;
      let handle = snapshotHandles.get(requirement.segmentId);
      if (handle === undefined) {
        handle = { segmentId: requirement.segmentId };
        snapshotHandles.set(requirement.segmentId, handle);
      }
      let released = false;
      return {
        segmentId: requirement.segmentId,
        snapshot: { handle, cacheRevision: revision },
        isCurrent: () =>
          !released &&
          cachedSegments.has(requirement.segmentId) &&
          (revisions.get(requirement.segmentId) ?? 0) === revision,
        release: () => {
          if (released) return;
          released = true;
          releases.push(requirement.segmentId);
        },
      };
    },
    acquireInputReference(requirement: SpatialSkeletonQueueInputRequirement) {
      const inputReference = this.tryAcquireInputReference(requirement);
      if (inputReference !== undefined) return inputReference;
      throw new SpatialSkeletonInspectionRequiredError(requirement);
    },
    getFullSegmentNodes,
    releaseFullSegmentNodeFetchOwner: vi.fn(),
    assertOptimisticEditingAllowed() {
      if (fatalState !== undefined) {
        throw new SpatialSkeletonOptimisticReloadRequiredError(fatalState);
      }
    },
  };
  const layer = {
    spatialSkeletonState: state,
    getSpatiallyIndexedSkeletonLayer: () => ({}),
  };
  const replaceCachedSegment = (segmentId: number) => {
    cachedSegments.add(segmentId);
    revisions.set(segmentId, (revisions.get(segmentId) ?? 0) + 1);
    snapshotHandles.set(segmentId, { segmentId });
  };
  return {
    cachedSegments,
    layer: layer as any,
    releases,
    replaceCachedSegment,
    setFatalState(value: SpatialSkeletonOptimisticFatalState) {
      fatalState = value;
    },
    state,
  };
}

function makeCommand(
  required: readonly SpatialSkeletonQueueInputRequirement[],
  loadable: readonly SpatialSkeletonQueueInputRequirement[] = [],
): SpatialSkeletonEditCommand {
  return {
    action: SpatialSkeletonActions.moveNodes,
    label: "Test edit",
    payload: Object.freeze({}),
    getQueueInputRequirements: () => ({ required, loadable }),
  };
}

describe("spatial skeleton queue admission", () => {
  it("admits a root-like transition without existing queue input or a read", async () => {
    const { layer, state } = makeHarness();
    const submitToQueue = vi.fn((queueInput: SpatialSkeletonQueueInput) => {
      expect(queueInput).toEqual({ segments: [] });
      expect(Object.isFrozen(queueInput)).toBe(true);
      expect(Object.isFrozen(queueInput.segments)).toBe(true);
      return optimisticExecution(Promise.resolve());
    });

    const execution = prepareAndSubmitSpatialSkeletonEdit(
      layer,
      makeCommand([]),
      submitToQueue,
    );

    expect(submitToQueue).toHaveBeenCalledOnce();
    expect(state.getFullSegmentNodes).not.toHaveBeenCalled();
    await expect(execution).resolves.toBeUndefined();
  });

  it("forwards a frozen bundle deduplicated in requirement order", async () => {
    const { layer } = makeHarness([11, 21]);
    const submitToQueue = vi.fn((queueInput: SpatialSkeletonQueueInput) => {
      expect(queueInput.segments.map(({ segmentId }) => segmentId)).toEqual([
        21, 11,
      ]);
      expect(Object.isFrozen(queueInput)).toBe(true);
      expect(Object.isFrozen(queueInput.segments)).toBe(true);
      expect(
        queueInput.segments.every((segment) => Object.isFrozen(segment)),
      ).toBe(true);
      expect(Object.isFrozen(queueInput.segments[0].snapshot)).toBe(false);
      return optimisticExecution(Promise.resolve());
    });

    await prepareAndSubmitSpatialSkeletonEdit(
      layer,
      makeCommand([
        { segmentId: 21, nodeId: 210 },
        { segmentId: 11 },
        { segmentId: 21, nodeId: 211 },
      ]),
      submitToQueue,
    );

    expect(submitToQueue).toHaveBeenCalledOnce();
  });

  it("rejects duplicate segment requirements that resolve differently", async () => {
    const { layer, state } = makeHarness([11]);
    const releases = [vi.fn(), vi.fn()];
    let call = 0;
    (state as any).acquireInputReference = vi.fn(
      (requirement: SpatialSkeletonQueueInputRequirement) => {
        const index = call++;
        return {
          segmentId: requirement.segmentId,
          snapshot: {
            handle: { version: index },
            cacheRevision: index + 1,
          },
          isCurrent: () => true,
          release: releases[index],
        };
      },
    );
    const submitToQueue = vi.fn(() => optimisticExecution(Promise.resolve()));

    const execution = prepareAndSubmitSpatialSkeletonEdit(
      layer,
      makeCommand([
        { segmentId: 11, nodeId: 110 },
        { segmentId: 11, nodeId: 111 },
      ]),
      submitToQueue,
    );

    await expect(execution.acceptedByQueue).rejects.toThrow(
      /segment 11.*different complete snapshots/,
    );
    await expect(execution).rejects.toThrow(
      /segment 11.*different complete snapshots/,
    );
    expect(submitToQueue).not.toHaveBeenCalled();
    expect(releases[0]).toHaveBeenCalledOnce();
    expect(releases[1]).toHaveBeenCalledOnce();
  });

  it("rejects Reload required before acquiring or loading queue input", async () => {
    const { layer, setFatalState, state } = makeHarness();
    setFatalState({
      reason: "authority-indeterminate",
      authority: "indeterminate",
    });
    const submitToQueue = vi.fn(() => optimisticExecution(Promise.resolve()));

    const execution = prepareAndSubmitSpatialSkeletonEdit(
      layer,
      makeCommand([], [{ segmentId: 21 }]),
      submitToQueue,
      SpatialSkeletonActions.mergeSkeletons,
    );

    await expect(execution.acceptedByQueue).rejects.toBeInstanceOf(
      SpatialSkeletonOptimisticReloadRequiredError,
    );
    await expect(execution).rejects.toBeInstanceOf(
      SpatialSkeletonOptimisticReloadRequiredError,
    );
    await expect(execution.settled).resolves.toMatchObject({
      outcome: "unchanged",
      reason: "not-started",
    });
    expect(state.getFullSegmentNodes).not.toHaveBeenCalled();
    expect(submitToQueue).not.toHaveBeenCalled();
  });

  it("rechecks Reload required after loading queue input", async () => {
    const { cachedSegments, layer, setFatalState, state } = makeHarness([11]);
    const targetRead = createDeferred<never[]>();
    state.getFullSegmentNodes.mockImplementationOnce(
      async (_layer: unknown, segmentId: number) => {
        await targetRead.promise;
        cachedSegments.add(segmentId);
        return [];
      },
    );
    const submitToQueue = vi.fn(() => optimisticExecution(Promise.resolve()));
    const execution = prepareAndSubmitSpatialSkeletonEdit(
      layer,
      makeCommand([{ segmentId: 11 }], [{ segmentId: 21 }, { segmentId: 31 }]),
      submitToQueue,
      SpatialSkeletonActions.mergeSkeletons,
    );

    setFatalState({
      reason: "committed-local-publication-failed",
      authority: "committed",
    });
    targetRead.resolve([]);

    await expect(execution.acceptedByQueue).rejects.toBeInstanceOf(
      SpatialSkeletonOptimisticReloadRequiredError,
    );
    await expect(execution).rejects.toBeInstanceOf(
      SpatialSkeletonOptimisticReloadRequiredError,
    );
    expect(submitToQueue).not.toHaveBeenCalled();
    expect(state.getFullSegmentNodes).toHaveBeenCalledTimes(1);
  });

  it("rejects a cold required skeleton before queue submission", async () => {
    const { layer, state } = makeHarness();
    const submitToQueue = vi.fn(() => optimisticExecution(Promise.resolve()));
    const execution = prepareAndSubmitSpatialSkeletonEdit(
      layer,
      makeCommand([{ segmentId: 11, nodeId: 12 }]),
      submitToQueue,
    );

    await expect(execution.acceptedByQueue).rejects.toBeInstanceOf(
      SpatialSkeletonInspectionRequiredError,
    );
    await expect(execution).rejects.toBeInstanceOf(
      SpatialSkeletonInspectionRequiredError,
    );
    await expect(execution.settled).resolves.toMatchObject({
      outcome: "unchanged",
      reason: "not-started",
    });
    expect(submitToQueue).not.toHaveBeenCalled();
    expect(state.getFullSegmentNodes).not.toHaveBeenCalled();
  });

  it("releases a cached input after the exact preview while forwarding authority settlement", async () => {
    const { layer, releases, state } = makeHarness([11, 21]);
    const exactPreview = createDeferred<void>();
    const authoritySettlement =
      createDeferred<SpatialSkeletonOptimisticEditSettlement>();
    const submitToQueue = vi.fn(() =>
      optimisticExecution(exactPreview.promise, authoritySettlement.promise),
    );
    const execution = prepareAndSubmitSpatialSkeletonEdit(
      layer,
      makeCommand([{ segmentId: 11, nodeId: 12 }], [{ segmentId: 21 }]),
      submitToQueue,
    );

    expect(submitToQueue).toHaveBeenCalledOnce();
    expect(state.getFullSegmentNodes).not.toHaveBeenCalled();
    expect(releases).toEqual([]);
    expect(execution.settled).toBe(authoritySettlement.promise);
    exactPreview.resolve();
    await execution;
    expect(releases).toEqual([11, 21]);
    let settled = false;
    void execution.settled.then(() => {
      settled = true;
    });
    await Promise.resolve();
    expect(settled).toBe(false);

    authoritySettlement.resolve({ outcome: "committed" });
    await expect(execution.settled).resolves.toEqual({
      outcome: "committed",
    });
  });

  it.each(["resolved", "rejected", "pending"] as const)(
    "preserves %s queue settlement after loading inputs and failing the preview",
    async (outcome) => {
      const { layer, releases, state } = makeHarness([11]);
      const preview = createDeferred<void>();
      const queueSettlement =
        createDeferred<SpatialSkeletonOptimisticEditSettlement>();
      const submitToQueue = vi.fn(() =>
        optimisticExecution(preview.promise, queueSettlement.promise),
      );
      const execution = prepareAndSubmitSpatialSkeletonEdit(
        layer,
        makeCommand([{ segmentId: 11 }], [{ segmentId: 21 }]),
        submitToQueue,
      );
      const onSettled = vi.fn();
      const onSettlementRejected = vi.fn();
      void execution.settled.then(onSettled, onSettlementRejected);

      expect(submitToQueue).not.toHaveBeenCalled();
      await execution.acceptedByQueue;
      expect(state.getFullSegmentNodes).toHaveBeenCalledOnce();
      expect(submitToQueue).toHaveBeenCalledOnce();

      const previewError = new Error("Preview preparation failed");
      preview.reject(previewError);
      await expect(execution).rejects.toBe(previewError);
      expect(releases).toEqual([11, 21]);
      expect(onSettled).not.toHaveBeenCalled();
      expect(onSettlementRejected).not.toHaveBeenCalled();

      if (outcome === "resolved") {
        const result: SpatialSkeletonOptimisticEditSettlement = {
          outcome: "unchanged",
          reason: "not-started",
          error: previewError,
        };
        queueSettlement.resolve(result);
        await expect(execution.settled).resolves.toBe(result);
      } else if (outcome === "rejected") {
        const settlementError = new Error("Queue settlement failed");
        queueSettlement.reject(settlementError);
        await expect(execution.settled).rejects.toBe(settlementError);
      }
    },
  );

  it("reports a submission failure after loading and releases the input references", async () => {
    const { layer, releases, state } = makeHarness([11]);
    const error = new Error("Queue submission failed");
    const submitToQueue = vi.fn(() => {
      throw error;
    });
    const execution = prepareAndSubmitSpatialSkeletonEdit(
      layer,
      makeCommand([{ segmentId: 11 }], [{ segmentId: 21 }]),
      submitToQueue,
    );

    await expect(execution.acceptedByQueue).rejects.toBe(error);
    await expect(execution).rejects.toBe(error);
    await expect(execution.settled).resolves.toEqual({
      outcome: "unchanged",
      reason: "not-started",
      error,
    });
    expect(state.getFullSegmentNodes).toHaveBeenCalledOnce();
    expect(submitToQueue).toHaveBeenCalledOnce();
    expect(releases).toEqual([11, 21]);
  });

  it("checks required inputs before loading any declared input", async () => {
    const { layer, state } = makeHarness();
    const submitToQueue = vi.fn(() => optimisticExecution(Promise.resolve()));
    const execution = prepareAndSubmitSpatialSkeletonEdit(
      layer,
      makeCommand(
        [{ segmentId: 11, nodeId: 12 }],
        [
          {
            segmentId: 21,
            nodeId: 22,
          },
        ],
      ),
      submitToQueue,
      SpatialSkeletonActions.mergeSkeletons,
    );

    await expect(execution).rejects.toBeInstanceOf(
      SpatialSkeletonInspectionRequiredError,
    );
    expect(state.getFullSegmentNodes).not.toHaveBeenCalled();
    expect(submitToQueue).not.toHaveBeenCalled();
  });

  it("loads a declared input before Execute admission", async () => {
    const { cachedSegments, layer, state } = makeHarness([11]);
    const targetRead = createDeferred<never[]>();
    state.getFullSegmentNodes.mockImplementationOnce(
      async (_layer: unknown, segmentId: number) => {
        expect(segmentId).toBe(21);
        await targetRead.promise;
        cachedSegments.add(segmentId);
        return [];
      },
    );
    const submitToQueue = vi.fn(() => optimisticExecution(Promise.resolve()));
    const command = makeCommand(
      [{ segmentId: 11, nodeId: 12 }],
      [
        {
          segmentId: 21,
          nodeId: 22,
        },
      ],
    );

    const execution = prepareAndSubmitSpatialSkeletonEdit(
      layer,
      command,
      submitToQueue,
      SpatialSkeletonActions.mergeSkeletons,
    );
    expect(submitToQueue).not.toHaveBeenCalled();
    expect(state.getFullSegmentNodes).toHaveBeenCalledTimes(1);
    targetRead.resolve([]);
    await execution.acceptedByQueue;
    await execution;
    expect(submitToQueue).toHaveBeenCalledOnce();
  });

  it("keeps the source input available and rejects if it is replaced during a target read", async () => {
    const { cachedSegments, layer, releases, replaceCachedSegment, state } =
      makeHarness([11]);
    const targetRead = createDeferred<never[]>();
    state.getFullSegmentNodes.mockImplementationOnce(
      async (_layer: unknown, segmentId: number) => {
        expect(segmentId).toBe(21);
        await targetRead.promise;
        cachedSegments.add(segmentId);
        return [];
      },
    );
    const submitToQueue = vi.fn(() => optimisticExecution(Promise.resolve()));
    const execution = prepareAndSubmitSpatialSkeletonEdit(
      layer,
      makeCommand(
        [{ segmentId: 11, nodeId: 12 }],
        [
          {
            segmentId: 21,
            nodeId: 22,
          },
        ],
      ),
      submitToQueue,
      SpatialSkeletonActions.mergeSkeletons,
    );

    expect(releases).toEqual([]);
    replaceCachedSegment(11);
    expect(releases).toEqual([]);
    targetRead.resolve([]);

    await expect(execution.acceptedByQueue).rejects.toMatchObject({
      name: "SpatialSkeletonInspectionRequiredError",
      reason: "snapshot-changed",
      segmentId: 11,
    });
    await expect(execution).rejects.toBeInstanceOf(
      SpatialSkeletonInspectionRequiredError,
    );
    expect(submitToQueue).not.toHaveBeenCalled();
    expect(state.getFullSegmentNodes).toHaveBeenCalledTimes(1);
    expect(releases).toEqual([11]);
  });

  it("rejects endpoint remapping during a target read without loading the new target", async () => {
    const { cachedSegments, layer, releases, state } = makeHarness([11]);
    const targetRead = createDeferred<never[]>();
    state.getFullSegmentNodes.mockImplementationOnce(
      async (_layer: unknown, segmentId: number) => {
        expect(segmentId).toBe(21);
        await targetRead.promise;
        cachedSegments.add(segmentId);
        return [];
      },
    );
    let targetSegmentId = 21;
    const command: SpatialSkeletonEditCommand = {
      action: SpatialSkeletonActions.mergeSkeletons,
      label: "Mapped merge",
      payload: Object.freeze({}),
      getQueueInputRequirements: () => ({
        required: [{ segmentId: 11, nodeId: 12 }],
        loadable: [
          {
            segmentId: targetSegmentId,
            nodeId: 22,
          },
        ],
      }),
    };
    const submitToQueue = vi.fn(() => optimisticExecution(Promise.resolve()));
    const execution = prepareAndSubmitSpatialSkeletonEdit(
      layer,
      command,
      submitToQueue,
      SpatialSkeletonActions.mergeSkeletons,
    );

    targetSegmentId = 31;
    targetRead.resolve([]);

    await expect(execution.acceptedByQueue).rejects.toMatchObject({
      name: "SpatialSkeletonInspectionRequiredError",
      reason: "requirements-changed",
      segmentId: 21,
    });
    await expect(execution).rejects.toBeInstanceOf(
      SpatialSkeletonInspectionRequiredError,
    );
    expect(submitToQueue).not.toHaveBeenCalled();
    expect(state.getFullSegmentNodes).toHaveBeenCalledTimes(1);
    expect(state.getFullSegmentNodes).toHaveBeenCalledWith(
      expect.anything(),
      21,
      expect.anything(),
    );
    expect(releases).toEqual([11]);
  });

  it("loads multiple inputs for a non-Merge action and preserves requirement order", async () => {
    const { cachedSegments, layer, releases, state } = makeHarness([11, 31]);
    const firstRead = createDeferred<void>();
    const secondRead = createDeferred<void>();
    const secondStarted = createDeferred<void>();
    state.getFullSegmentNodes.mockImplementation(async (_layer, segmentId) => {
      if (segmentId === 21) await firstRead.promise;
      if (segmentId === 41) {
        secondStarted.resolve();
        await secondRead.promise;
      }
      cachedSegments.add(segmentId);
      return [];
    });
    const submitToQueue = vi.fn((input: SpatialSkeletonQueueInput) => {
      expect(input.segments.map(({ segmentId }) => segmentId)).toEqual([
        11, 21, 31, 41,
      ]);
      return optimisticExecution(Promise.resolve());
    });
    const execution = prepareAndSubmitSpatialSkeletonEdit(
      layer,
      makeCommand(
        [{ segmentId: 11 }],
        [
          { segmentId: 21, nodeId: 210 },
          { segmentId: 31 },
          { segmentId: 21, nodeId: 211 },
          { segmentId: 41 },
        ],
      ),
      submitToQueue,
    );

    expect(submitToQueue).not.toHaveBeenCalled();
    expect(state.getFullSegmentNodes).toHaveBeenCalledTimes(1);
    firstRead.resolve();
    await secondStarted.promise;
    expect(submitToQueue).not.toHaveBeenCalled();
    expect(releases).toEqual([]);
    secondRead.resolve();
    await execution.acceptedByQueue;
    await execution;

    expect(submitToQueue).toHaveBeenCalledOnce();
    expect(state.getFullSegmentNodes.mock.calls.map(([, id]) => id)).toEqual([
      21, 41,
    ]);
    expect(state.releaseFullSegmentNodeFetchOwner).toHaveBeenCalledTimes(2);
    expect(releases).toEqual([11, 21, 31, 21, 41]);
  });

  it.each(["read failure", 11, 21, 31] as const)(
    "releases every input reference when a later read fails or invalidates input %s",
    async (failure) => {
      const { cachedSegments, layer, releases, replaceCachedSegment, state } =
        makeHarness([11, 31]);
      const lastRead = createDeferred<void>();
      const lastStarted = createDeferred<void>();
      const readError = new Error("input unavailable");
      state.getFullSegmentNodes.mockImplementation(
        async (_layer, segmentId) => {
          if (segmentId === 41) {
            lastStarted.resolve();
            await lastRead.promise;
            if (failure === "read failure") throw readError;
          }
          cachedSegments.add(segmentId);
          return [];
        },
      );
      const submitToQueue = vi.fn(() => optimisticExecution(Promise.resolve()));
      const execution = prepareAndSubmitSpatialSkeletonEdit(
        layer,
        makeCommand(
          [{ segmentId: 11 }],
          [{ segmentId: 21 }, { segmentId: 31 }, { segmentId: 41 }],
        ),
        submitToQueue,
      );
      await lastStarted.promise;
      expect(releases).toEqual([]);
      if (typeof failure === "number") replaceCachedSegment(failure);
      lastRead.resolve();

      if (failure === "read failure") {
        await expect(execution).rejects.toBe(readError);
      } else {
        await expect(execution).rejects.toMatchObject({
          reason: "snapshot-changed",
          segmentId: failure,
        });
      }
      await expect(execution.settled).resolves.toMatchObject({
        outcome: "unchanged",
        reason: "not-started",
      });
      expect(submitToQueue).not.toHaveBeenCalled();
      expect(releases).toEqual([11, 21, 31]);
      expect(state.releaseFullSegmentNodeFetchOwner).toHaveBeenCalledTimes(2);
    },
  );

  it("rejects a loading-policy change before reading another input", async () => {
    const { cachedSegments, layer, releases, state } = makeHarness([11]);
    const read = createDeferred<void>();
    state.getFullSegmentNodes.mockImplementation(async (_layer, segmentId) => {
      await read.promise;
      cachedSegments.add(segmentId);
      return [];
    });
    let required = [{ segmentId: 11 }];
    let loadable = [{ segmentId: 21 }, { segmentId: 31 }];
    const command = {
      ...makeCommand([]),
      getQueueInputRequirements: () => ({ required, loadable }),
    };
    const submitToQueue = vi.fn(() => optimisticExecution(Promise.resolve()));
    const execution = prepareAndSubmitSpatialSkeletonEdit(
      layer,
      command,
      submitToQueue,
    );
    required = [{ segmentId: 11 }, { segmentId: 31 }];
    loadable = [{ segmentId: 21 }];
    read.resolve();

    await expect(execution).rejects.toMatchObject({
      reason: "requirements-changed",
      segmentId: 31,
    });
    expect(submitToQueue).not.toHaveBeenCalled();
    expect(state.getFullSegmentNodes).toHaveBeenCalledTimes(1);
    expect(releases).toEqual([11]);
  });

  it("does not admit an action when loading target queue input fails", async () => {
    const { layer, state } = makeHarness([11]);
    const readError = new Error("target unavailable");
    state.getFullSegmentNodes.mockRejectedValueOnce(readError);
    const submitToQueue = vi.fn(() => optimisticExecution(Promise.resolve()));
    const execution = prepareAndSubmitSpatialSkeletonEdit(
      layer,
      makeCommand(
        [{ segmentId: 11, nodeId: 12 }],
        [
          {
            segmentId: 21,
            nodeId: 22,
          },
        ],
      ),
      submitToQueue,
      SpatialSkeletonActions.mergeSkeletons,
    );

    await expect(execution.acceptedByQueue).rejects.toBe(readError);
    await expect(execution).rejects.toBe(readError);
    expect(submitToQueue).not.toHaveBeenCalled();
  });
});
