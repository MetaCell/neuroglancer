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

import type {
  SpatialSkeletonAction,
  SpatialSkeletonCommandContext,
  SpatialSkeletonEditCommand,
  SpatialSkeletonQueueInput,
  SpatialSkeletonQueueInputRequirement,
  SpatialSkeletonQueueInputRequirements,
} from "#src/skeleton/command_protocol.js";
import { SpatialSkeletonInspectionRequiredError } from "#src/skeleton/edit_errors.js";
import {
  type SpatialSkeletonOptimisticEditSettlement,
  unchangedSpatialSkeletonOptimisticEditSettlement,
} from "#src/skeleton/optimistic_edit/lifecycle.js";
import type {
  SpatialSkeletonInputReference,
  SpatialSkeletonLayerContext,
  SpatialSkeletonOptimisticEditExecution,
} from "#src/skeleton/spatial_skeleton_manager.js";
import { createDeferred, withPromiseProperties } from "#src/util/promise.js";

/** A queue input requirement paired with a reference to its complete snapshot. */
interface QueueInputBinding {
  /**
   * What the edit needs: e.g. { segmentId: 23, nodeId: 5 } requires a complete
   * snapshot of skeleton 23 containing node 5.
   */
  readonly requirement: SpatialSkeletonQueueInputRequirement;
  /**
   * The acquired reference: e.g. skeleton 23's snapshot at cache revision 10.
   * reference.snapshot.handle.getNode(5) reads node 5 from that version.
   * reference.release() removes this preparation step's cache registration.
   */
  readonly reference: SpatialSkeletonInputReference;
}

/** Release this preparation step's references to complete cached snapshots. */
function releaseInputReferences(bindings: readonly QueueInputBinding[]) {
  for (const { reference } of bindings) reference.release();
}

function captureQueueInputRequirements(
  command: SpatialSkeletonEditCommand,
  context: SpatialSkeletonCommandContext,
) {
  const { required, loadable = [] } =
    command.getQueueInputRequirements(context);
  return Object.freeze({
    required: Object.freeze(
      required.map((requirement) => Object.freeze({ ...requirement })),
    ),
    loadable: Object.freeze(
      loadable.map((requirement) => Object.freeze({ ...requirement })),
    ),
  });
}

/** Bind required inputs to acquired cached references before starting any reads. */
function prepareRequiredInputBindings(
  layer: SpatialSkeletonLayerContext,
  requirements: readonly SpatialSkeletonQueueInputRequirement[],
  action?: SpatialSkeletonAction,
): QueueInputBinding[] {
  const bindings: QueueInputBinding[] = [];
  try {
    for (const requirement of requirements) {
      bindings.push({
        requirement,
        reference: layer.spatialSkeletonState.acquireInputReference(
          requirement,
          action,
        ),
      });
    }
    return bindings;
  } catch (error) {
    releaseInputReferences(bindings);
    throw error;
  }
}

async function loadQueueInputRequirement(
  layer: SpatialSkeletonLayerContext,
  requirement: SpatialSkeletonQueueInputRequirement,
) {
  const skeletonLayer = layer.getSpatiallyIndexedSkeletonLayer();
  if (skeletonLayer === undefined) {
    throw new Error(
      "No active spatial skeleton layer is available to load queue inputs.",
    );
  }
  const requestOwner = {};
  try {
    await layer.spatialSkeletonState.getFullSegmentNodes(
      skeletonLayer,
      requirement.segmentId,
      {
        // The edit still needs this read if the pointer leaves the skeleton.
        // Its owner joins any existing fetch without taking sole ownership.
        retainWhileInactive: true,
        requestOwner,
      },
    );
  } finally {
    layer.spatialSkeletonState.releaseFullSegmentNodeFetchOwner(requestOwner);
  }
}

function assertInputReferencesCurrent(
  bindings: readonly QueueInputBinding[],
  action?: SpatialSkeletonAction,
) {
  for (const { reference, requirement } of bindings) {
    if (reference.isCurrent()) continue;
    throw new SpatialSkeletonInspectionRequiredError(
      requirement,
      action,
      "snapshot-changed",
    );
  }
}

/** Require unchanged endpoints and loading policies; order alone may differ. */
function assertQueueInputRequirementsUnchanged(
  original: SpatialSkeletonQueueInputRequirements,
  current: SpatialSkeletonQueueInputRequirements,
  action?: SpatialSkeletonAction,
): void {
  for (const group of ["required", "loadable"] as const) {
    const unmatched = [...(current[group] ?? [])];
    for (const requirement of original[group] ?? []) {
      const index = unmatched.findIndex(
        (candidate) =>
          candidate.segmentId === requirement.segmentId &&
          candidate.nodeId === requirement.nodeId,
      );
      if (index === -1) {
        throw new SpatialSkeletonInspectionRequiredError(
          requirement,
          action,
          "requirements-changed",
        );
      }
      unmatched.splice(index, 1);
    }
    if (unmatched.length !== 0) {
      throw new SpatialSkeletonInspectionRequiredError(
        unmatched[0],
        action,
        "requirements-changed",
      );
    }
  }
}

/** Capture requirements and bind each to a reference, loading missing inputs in order. */
function prepareQueueInputBindings(
  layer: SpatialSkeletonLayerContext,
  command: SpatialSkeletonEditCommand,
  context: SpatialSkeletonCommandContext,
  action?: SpatialSkeletonAction,
): readonly QueueInputBinding[] | Promise<readonly QueueInputBinding[]> {
  const requirements = captureQueueInputRequirements(command, context);
  const requiredBindings = prepareRequiredInputBindings(
    layer,
    requirements.required,
    action,
  );
  const state = layer.spatialSkeletonState;
  const loadableBindings: (QueueInputBinding | undefined)[] = [];
  const getBindings = () => [
    ...requiredBindings,
    ...loadableBindings.filter((binding) => binding !== undefined),
  ];
  try {
    // Keep every cached input available before any read can yield.
    for (const requirement of requirements.loadable) {
      const reference = state.tryAcquireInputReference(requirement);
      loadableBindings.push(
        reference === undefined ? undefined : { requirement, reference },
      );
    }
    assertInputReferencesCurrent(getBindings(), action);
    if (loadableBindings.every((binding) => binding !== undefined))
      return getBindings();
  } catch (error) {
    releaseInputReferences(getBindings());
    throw error;
  }

  return (async () => {
    try {
      for (const [index, requirement] of requirements.loadable.entries()) {
        if (loadableBindings[index] !== undefined) continue;
        state.assertOptimisticEditingAllowed();
        assertInputReferencesCurrent(getBindings(), action);
        // An earlier endpoint may have loaded this same complete skeleton.
        let reference = state.tryAcquireInputReference(requirement);
        if (reference === undefined) {
          await loadQueueInputRequirement(layer, requirement);
          state.assertOptimisticEditingAllowed();
          assertInputReferencesCurrent(getBindings(), action);
          assertQueueInputRequirementsUnchanged(
            requirements,
            captureQueueInputRequirements(command, context),
            action,
          );
          reference = state.acquireInputReference(requirement, action);
        }
        loadableBindings[index] = { requirement, reference };
      }
      const bindings = getBindings();
      assertInputReferencesCurrent(bindings, action);
      return bindings;
    } catch (error) {
      releaseInputReferences(getBindings());
      throw error;
    }
  })();
}

function createQueueInput(
  bindings: readonly QueueInputBinding[],
): SpatialSkeletonQueueInput {
  const segments = new Map<
    number,
    SpatialSkeletonQueueInput["segments"][number]
  >();
  for (const { reference } of bindings) {
    const candidate = Object.freeze({
      segmentId: reference.segmentId,
      snapshot: reference.snapshot.handle,
      cacheRevision: reference.snapshot.cacheRevision,
    });
    const existing = segments.get(candidate.segmentId);
    if (existing === undefined) {
      segments.set(candidate.segmentId, candidate);
      continue;
    }
    if (
      existing.snapshot !== candidate.snapshot ||
      existing.cacheRevision !== candidate.cacheRevision
    ) {
      throw new Error(
        `Queue input requirements for segment ${candidate.segmentId} resolved to different complete snapshots.`,
      );
    }
  }
  return Object.freeze({ segments: Object.freeze([...segments.values()]) });
}

function submitEdit<T>(
  layer: SpatialSkeletonLayerContext,
  bindings: readonly QueueInputBinding[],
  submitToQueue: (
    queueInput: SpatialSkeletonQueueInput,
  ) => SpatialSkeletonOptimisticEditExecution<T>,
  action?: SpatialSkeletonAction,
) {
  try {
    // A fatal state can be latched while a queue input read is in flight.
    // Recheck at the last synchronous boundary before queue admission.
    layer.spatialSkeletonState.assertOptimisticEditingAllowed();
    assertInputReferencesCurrent(bindings, action);
    return submitToQueue(createQueueInput(bindings));
  } catch (error) {
    releaseInputReferences(bindings);
    throw error;
  }
}

/**
 * Prepares the complete snapshots declared by a new edit, then submits it to
 * the queue while preserving acceptance, exact-preview, and settlement promises.
 */
export function prepareAndSubmitSpatialSkeletonEdit<T>(
  layer: SpatialSkeletonLayerContext,
  command: SpatialSkeletonEditCommand,
  submitToQueue: (
    queueInput: SpatialSkeletonQueueInput,
  ) => SpatialSkeletonOptimisticEditExecution<T>,
  action?: SpatialSkeletonAction,
): SpatialSkeletonOptimisticEditExecution<T> {
  const context = {
    identities:
      layer.spatialSkeletonState.getOptimisticEditingIdentityService(),
  };
  // Expose the promise now so callers can wait for acceptance. It stays pending
  // while inputs load, then follows the queue's acceptance or rejection.
  const acceptedByQueue = createDeferred<void>();
  const settled = createDeferred<SpatialSkeletonOptimisticEditSettlement>();
  // Keep the independent admission rejection observed for callers that only
  // await the exact-preview/execution promise.
  void acceptedByQueue.promise.catch(() => undefined);
  let inputBindings:
    | readonly QueueInputBinding[]
    | Promise<readonly QueueInputBinding[]>;
  try {
    // Do not start even bounded admission hydration after the layer has
    // entered Reload required.
    layer.spatialSkeletonState.assertOptimisticEditingAllowed();
    inputBindings = prepareQueueInputBindings(layer, command, context, action);
  } catch (error) {
    acceptedByQueue.reject(error);
    settled.resolve(
      unchangedSpatialSkeletonOptimisticEditSettlement("not-started", error),
    );
    return withPromiseProperties(Promise.reject(error), {
      acceptedByQueue: acceptedByQueue.promise,
      settled: settled.promise,
    });
  }

  if (!(inputBindings instanceof Promise)) {
    let inner: SpatialSkeletonOptimisticEditExecution<T>;
    try {
      inner = submitEdit(layer, inputBindings, submitToQueue, action);
    } catch (error) {
      acceptedByQueue.reject(error);
      settled.resolve(
        unchangedSpatialSkeletonOptimisticEditSettlement("not-started", error),
      );
      return withPromiseProperties(Promise.reject(error), {
        acceptedByQueue: acceptedByQueue.promise,
        settled: settled.promise,
      });
    }
    inner.acceptedByQueue.then(acceptedByQueue.resolve, acceptedByQueue.reject);
    return withPromiseProperties(
      inner.finally(() => releaseInputReferences(inputBindings)),
      { acceptedByQueue: acceptedByQueue.promise, settled: inner.settled },
    );
  }

  const execution = (async () => {
    let bindings: readonly QueueInputBinding[];
    let inner: SpatialSkeletonOptimisticEditExecution<T>;
    try {
      bindings = await inputBindings;
      inner = submitEdit(layer, bindings, submitToQueue, action);
    } catch (error) {
      // No execution was returned, so this wrapper reports the failure.
      acceptedByQueue.reject(error);
      settled.resolve(
        unchangedSpatialSkeletonOptimisticEditSettlement("not-started", error),
      );
      throw error;
    }

    // An execution now exists. Forward its outcome independently of whether
    // the preview succeeds, fails, or finishes before settlement.
    inner.acceptedByQueue.then(acceptedByQueue.resolve, acceptedByQueue.reject);
    inner.settled.then(settled.resolve, settled.reject);
    try {
      return await inner;
    } finally {
      releaseInputReferences(bindings);
    }
  })();
  // A caller may observe only acceptance while the asynchronous read fails.
  void execution.catch(() => undefined);
  return withPromiseProperties(execution, {
    acceptedByQueue: acceptedByQueue.promise,
    settled: settled.promise,
  });
}
