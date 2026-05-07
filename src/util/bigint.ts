/**
 * @license
 * Copyright 2025 Google Inc.
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

export function bigintCompare(a: bigint, b: bigint) {
  return a < b ? -1 : a > b ? 1 : 0;
}

export function uint64FromLowHigh(low: number, high: number) {
  return BigInt(low) | (BigInt(high) << 32n);
}

export function randomUint64(): bigint {
  const low = (Math.random() * 0x100000000) >>> 0;
  const high = (Math.random() * 0x100000000) >>> 0;
  return uint64FromLowHigh(low, high);
}

export function wrapSigned32BitIntegerToUint64(value: number): bigint {
  return uint64FromLowHigh(value >>> 0, value < 0 ? 0xffffffff : 0);
}

export function bigintMin(a: bigint, b: bigint): bigint {
  return a < b ? a : b;
}

export function bigintMax(a: bigint, b: bigint): bigint {
  return a > b ? a : b;
}

export const UINT64_MAX = 0xffffffffffffffffn;

function describeIntegerValue(value: unknown) {
  return typeof value === "bigint" ? value.toString() : JSON.stringify(value);
}

export function clampToUint64(x: bigint): bigint {
  if (x < 0n) return 0n;
  if (x > UINT64_MAX) return UINT64_MAX;
  return x;
}

export function parsePositiveUint64Id(value: unknown, label = "id"): bigint {
  let id: bigint;
  switch (typeof value) {
    case "bigint":
      id = value;
      break;
    case "number":
      if (!Number.isSafeInteger(value)) {
        throw new Error(
          `Expected ${label} to be a safe integer number, but received: ${describeIntegerValue(value)}.`,
        );
      }
      id = BigInt(value);
      break;
    case "string":
      if (!/^(?:0|[1-9][0-9]*)$/.test(value)) {
        throw new Error(
          `Expected ${label} to be a base-10 uint64 string, but received: ${JSON.stringify(value)}.`,
        );
      }
      id = BigInt(value);
      break;
    default:
      throw new Error(
        `Expected ${label} to be a uint64 id, but received: ${describeIntegerValue(value)}.`,
      );
  }
  if (id <= 0n || id > UINT64_MAX) {
    throw new Error(
      `Expected ${label} to be in range [1, ${UINT64_MAX}], but received: ${id}.`,
    );
  }
  return id;
}

export function compareUint64Ids(a: bigint, b: bigint) {
  return a < b ? -1 : a > b ? 1 : 0;
}

export function stringifySpatialSkeletonId(id: bigint) {
  return id.toString();
}

export function bigintAbs(x: bigint): bigint {
  return x < 0n ? -x : x;
}

export function roundToUint64(x: number | bigint): bigint {
  if (typeof x === "number") {
    if (x === Number.POSITIVE_INFINITY) return UINT64_MAX;
    if (!Number.isFinite(x)) return 0n;
    x = BigInt(Math.round(x));
  }
  return clampToUint64(x);
}
