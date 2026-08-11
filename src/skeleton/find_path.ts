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

import { RefCounted } from "#src/util/disposable.js";
import { verifyObject } from "#src/util/json.js";
import { NullarySignal } from "#src/util/signal.js";
import type { Trackable } from "#src/util/trackable.js";

/**
 * Representation-neutral endpoint identity for skeleton path finding.
 *
 * Spatial skeletons use their node and segment IDs. A regular skeleton can
 * use its object ID as `segmentId` and a stable vertex index as `nodeId`.
 */
export interface SkeletonFindPathEndpoint {
  readonly nodeId: bigint;
  readonly segmentId: bigint;
  readonly position: Float32Array;
}

export interface SkeletonFindPathResultNode {
  readonly nodeId: bigint;
  readonly position: Float32Array;
}

export interface SkeletonFindPathEndpointJson {
  nodeId: string;
  segmentId: string;
  position: number[];
}

export interface SkeletonFindPathResultNodeJson {
  nodeId: string;
  position: number[];
}

export interface SkeletonFindPathStateJson {
  source?: SkeletonFindPathEndpointJson;
  target?: SkeletonFindPathEndpointJson;
  result?: SkeletonFindPathResultNodeJson[];
}

function parsePositiveUint64(value: unknown, description: string): bigint {
  let parsed: bigint;
  try {
    if (typeof value === "bigint") {
      parsed = value;
    } else if (
      typeof value === "number" &&
      Number.isSafeInteger(value) &&
      value > 0
    ) {
      parsed = BigInt(value);
    } else if (typeof value === "string" && /^[1-9][0-9]*$/.test(value)) {
      parsed = BigInt(value);
    } else {
      throw new Error();
    }
  } catch {
    throw new Error(`Expected ${description} to be a positive uint64 value.`);
  }
  if (parsed <= 0n || parsed > 0xffff_ffff_ffff_ffffn) {
    throw new Error(`Expected ${description} to be a positive uint64 value.`);
  }
  return parsed;
}

function clonePosition(value: unknown, description: string): Float32Array {
  if (
    (!Array.isArray(value) && !(value instanceof Float32Array)) ||
    value.length !== 3
  ) {
    throw new Error(`Expected ${description} to be a finite 3D position.`);
  }
  const position = new Float32Array(3);
  for (let i = 0; i < 3; ++i) {
    const component = value[i];
    if (typeof component !== "number" || !Number.isFinite(component)) {
      throw new Error(`Expected ${description} to be a finite 3D position.`);
    }
    position[i] = component;
    if (!Number.isFinite(position[i])) {
      throw new Error(`Expected ${description} to be a finite 3D position.`);
    }
  }
  return position;
}

function parseEndpoint(value: unknown): SkeletonFindPathEndpoint {
  const obj = verifyObject(value);
  return {
    nodeId: parsePositiveUint64(obj.nodeId, "nodeId"),
    segmentId: parsePositiveUint64(obj.segmentId, "segmentId"),
    position: clonePosition(obj.position, "endpoint position"),
  };
}

function parseResultNode(value: unknown): SkeletonFindPathResultNode {
  const obj = verifyObject(value);
  return {
    nodeId: parsePositiveUint64(obj.nodeId, "nodeId"),
    position: clonePosition(obj.position, "result node position"),
  };
}

function parseResult(value: unknown): readonly SkeletonFindPathResultNode[] {
  if (!Array.isArray(value)) {
    throw new Error("Expected find-path result to be an array.");
  }
  return value.map(parseResultNode);
}

function positionsEqual(a: Float32Array, b: Float32Array) {
  return a[0] === b[0] && a[1] === b[1] && a[2] === b[2];
}

function endpointsEqual(
  a: SkeletonFindPathEndpoint | undefined,
  b: SkeletonFindPathEndpoint | undefined,
) {
  return (
    a === b ||
    (a !== undefined &&
      b !== undefined &&
      a.nodeId === b.nodeId &&
      a.segmentId === b.segmentId &&
      positionsEqual(a.position, b.position))
  );
}

function resultNodesEqual(
  a: SkeletonFindPathResultNode,
  b: SkeletonFindPathResultNode,
) {
  return a.nodeId === b.nodeId && positionsEqual(a.position, b.position);
}

function resultsEqual(
  a: readonly SkeletonFindPathResultNode[] | undefined,
  b: readonly SkeletonFindPathResultNode[] | undefined,
) {
  if (a === b) return true;
  if (a === undefined || b === undefined || a.length !== b.length) {
    return false;
  }
  return a.every((node, index) => resultNodesEqual(node, b[index]));
}

function validateStateRelationships(
  source: SkeletonFindPathEndpoint | undefined,
  target: SkeletonFindPathEndpoint | undefined,
  result: readonly SkeletonFindPathResultNode[] | undefined,
) {
  if (source !== undefined && target !== undefined) {
    if (source.segmentId !== target.segmentId) {
      throw new Error("Find-path endpoints must belong to the same segment.");
    }
    if (source.nodeId === target.nodeId) {
      throw new Error("Find-path endpoints must be distinct nodes.");
    }
  }
  if (result === undefined) return;
  if (source === undefined || target === undefined) {
    throw new Error("Both find-path endpoints are required for a result.");
  }
  if (result.length === 0) {
    throw new Error("A find-path result must contain at least one node.");
  }
  if (
    result[0].nodeId !== source.nodeId ||
    result[result.length - 1].nodeId !== target.nodeId
  ) {
    throw new Error(
      "A find-path result must start at the source and end at the target.",
    );
  }
}

function endpointToJson(
  endpoint: SkeletonFindPathEndpoint,
): SkeletonFindPathEndpointJson {
  const validated = parseEndpoint(endpoint);
  return {
    nodeId: validated.nodeId.toString(),
    segmentId: validated.segmentId.toString(),
    position: Array.from(validated.position),
  };
}

function resultNodeToJson(
  node: SkeletonFindPathResultNode,
): SkeletonFindPathResultNodeJson {
  const validated = parseResultNode(node);
  return {
    nodeId: validated.nodeId.toString(),
    position: Array.from(validated.position),
  };
}

/**
 * Serializable state shared by skeleton find-path implementations.
 *
 * Request tokens are runtime-only. A completion must present the token returned
 * by `beginRequest`; endpoint, result, or request mutations make older tokens
 * stale. Source ownership belongs to the skeleton state containing this object,
 * matching Graphene's datasource-owned state model.
 */
export class SkeletonFindPathState extends RefCounted implements Trackable {
  readonly changed = new NullarySignal();

  private source_: SkeletonFindPathEndpoint | undefined;
  private target_: SkeletonFindPathEndpoint | undefined;
  private result_: readonly SkeletonFindPathResultNode[] | undefined;
  private requestGeneration_ = 0;
  private pendingRequestGeneration_: number | undefined;

  get source() {
    return this.source_;
  }

  get target() {
    return this.target_;
  }

  get result() {
    return this.result_;
  }

  get requestGeneration() {
    return this.requestGeneration_;
  }

  get pendingRequestGeneration() {
    return this.pendingRequestGeneration_;
  }

  get requestPending() {
    return this.pendingRequestGeneration_ !== undefined;
  }

  private advanceRequestGeneration() {
    this.requestGeneration_ =
      this.requestGeneration_ === Number.MAX_SAFE_INTEGER
        ? 1
        : this.requestGeneration_ + 1;
    this.pendingRequestGeneration_ = undefined;
  }

  setSource(value: SkeletonFindPathEndpoint | undefined): boolean {
    return this.setEndpoints(value, this.target_);
  }

  setTarget(value: SkeletonFindPathEndpoint | undefined): boolean {
    return this.setEndpoints(this.source_, value);
  }

  /** Atomically replaces either or both endpoints. */
  setEndpoints(
    source: SkeletonFindPathEndpoint | undefined,
    target: SkeletonFindPathEndpoint | undefined,
  ): boolean {
    const parsedSource =
      source === undefined ? undefined : parseEndpoint(source);
    const parsedTarget =
      target === undefined ? undefined : parseEndpoint(target);
    validateStateRelationships(parsedSource, parsedTarget, undefined);
    if (
      endpointsEqual(this.source_, parsedSource) &&
      endpointsEqual(this.target_, parsedTarget)
    ) {
      return false;
    }
    this.source_ = parsedSource;
    this.target_ = parsedTarget;
    this.result_ = undefined;
    this.advanceRequestGeneration();
    this.changed.dispatch();
    return true;
  }

  /**
   * Starts a latest-request-wins operation and returns its opaque generation.
   * Starting a request clears any previously resolved result.
   */
  beginRequest(): number {
    this.advanceRequestGeneration();
    this.pendingRequestGeneration_ = this.requestGeneration_;
    this.result_ = undefined;
    this.changed.dispatch();
    return this.requestGeneration_;
  }

  isRequestCurrent(generation: number) {
    return (
      this.wasDisposed !== true && this.pendingRequestGeneration_ === generation
    );
  }

  /** Resolves a request if it is still current. Stale completions are ignored. */
  completeRequest(
    generation: number,
    result: readonly SkeletonFindPathResultNode[],
  ): boolean {
    if (!this.isRequestCurrent(generation)) return false;
    const parsedResult = parseResult(result);
    validateStateRelationships(this.source_, this.target_, parsedResult);
    const source = this.source_!;
    const target = this.target_!;
    this.source_ = {
      ...source,
      position: new Float32Array(parsedResult[0].position),
    };
    this.target_ = {
      ...target,
      position: new Float32Array(
        parsedResult[parsedResult.length - 1].position,
      ),
    };
    this.pendingRequestGeneration_ = undefined;
    this.result_ = parsedResult;
    this.changed.dispatch();
    return true;
  }

  /** Marks a request as no longer pending if it is still current. */
  failRequest(generation: number): boolean {
    if (!this.isRequestCurrent(generation)) return false;
    this.pendingRequestGeneration_ = undefined;
    this.changed.dispatch();
    return true;
  }

  /** Invalidates pending work without changing any persisted find-path state. */
  invalidatePendingRequest(): boolean {
    if (this.pendingRequestGeneration_ === undefined) return false;
    this.advanceRequestGeneration();
    this.changed.dispatch();
    return true;
  }

  /** Clears only the resolved result and invalidates any pending request. */
  invalidateResult(): boolean {
    if (
      this.result_ === undefined &&
      this.pendingRequestGeneration_ === undefined
    ) {
      return false;
    }
    this.result_ = undefined;
    this.advanceRequestGeneration();
    this.changed.dispatch();
    return true;
  }

  /** Clears the endpoints and resolved result. */
  clear(): boolean {
    if (
      this.source_ === undefined &&
      this.target_ === undefined &&
      this.result_ === undefined &&
      this.pendingRequestGeneration_ === undefined
    ) {
      return false;
    }
    this.source_ = undefined;
    this.target_ = undefined;
    this.result_ = undefined;
    this.advanceRequestGeneration();
    this.changed.dispatch();
    return true;
  }

  /** Restores the default empty state. */
  reset(): void {
    if (
      this.source_ === undefined &&
      this.target_ === undefined &&
      this.result_ === undefined &&
      this.pendingRequestGeneration_ === undefined
    ) {
      return;
    }
    this.source_ = undefined;
    this.target_ = undefined;
    this.result_ = undefined;
    this.advanceRequestGeneration();
    this.changed.dispatch();
  }

  toJSON(): SkeletonFindPathStateJson | undefined {
    const { source_, target_, result_ } = this;
    if (
      source_ === undefined &&
      target_ === undefined &&
      result_ === undefined
    ) {
      return undefined;
    }
    validateStateRelationships(source_, target_, result_);
    return {
      source: source_ === undefined ? undefined : endpointToJson(source_),
      target: target_ === undefined ? undefined : endpointToJson(target_),
      result: result_ === undefined ? undefined : result_.map(resultNodeToJson),
    };
  }

  /** Validates the complete input before replacing any current state. */
  restoreState(value: unknown): void {
    if (value === undefined) {
      this.reset();
      return;
    }
    const obj = verifyObject(value);
    const source =
      obj.source === undefined ? undefined : parseEndpoint(obj.source);
    const target =
      obj.target === undefined ? undefined : parseEndpoint(obj.target);
    const result =
      obj.result === undefined ? undefined : parseResult(obj.result);
    validateStateRelationships(source, target, result);

    const visibleStateChanged =
      !endpointsEqual(this.source_, source) ||
      !endpointsEqual(this.target_, target) ||
      !resultsEqual(this.result_, result);
    const requestWasPending = this.pendingRequestGeneration_ !== undefined;
    if (!visibleStateChanged && !requestWasPending) return;

    this.source_ = source;
    this.target_ = target;
    this.result_ = result;
    this.advanceRequestGeneration();
    this.changed.dispatch();
  }

  disposed() {
    this.advanceRequestGeneration();
    super.disposed();
  }
}
