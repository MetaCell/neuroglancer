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

// Sorted by segment ID.
export interface SwcSegments {
  readonly ids: BigUint64Array<ArrayBuffer>;
  readonly fileNames: readonly string[];
}

export const swcExtensionPattern = /\.swc$/i;

const fnvOffsetBasisHigh = 0xcbf29ce4;
const fnvOffsetBasisLow = 0x84222325;
const fnvPrimeHigh = 0x100;
const fnvPrimeLow = 0x1b3;
const utf8Encoder = new TextEncoder();
let utf8Buffer = new Uint8Array(256);

// Computed in two 32-bit halves, because `bigint` arithmetic for each byte is slow.
export function fnv1a64(text: string): bigint {
  // A UTF-16 code unit encodes to at most 3 UTF-8 bytes.
  if (utf8Buffer.length < 3 * text.length) {
    utf8Buffer = new Uint8Array(3 * text.length);
  }
  const { written } = utf8Encoder.encodeInto(text, utf8Buffer);
  let high = fnvOffsetBasisHigh;
  let low = fnvOffsetBasisLow;
  for (let index = 0; index < written; ++index) {
    low = (low ^ utf8Buffer[index]) >>> 0;
    // Below 2 ** 41, so exact in a double.
    const lowProduct = low * fnvPrimeLow;
    high =
      (Math.imul(high, fnvPrimeLow) +
        Math.imul(low, fnvPrimeHigh) +
        Math.floor(lowProduct / 0x100000000)) >>>
      0;
    low = lowProduct >>> 0;
  }
  return (BigInt(high) << 32n) | BigInt(low);
}

export function getSwcLabel(fileName: string): string {
  return fileName.replace(swcExtensionPattern, "");
}

// Segment IDs are saved in viewer state, so this rule must never change.
export function assignSegmentIds(
  fileNames: readonly string[],
  hash: (key: string) => bigint,
): SwcSegments {
  const labels = fileNames.map(getSwcLabel);
  let segmentIds = BigUint64Array.from(labels, (label) => hash(label));
  if (hasZeroOrRepeatedId(segmentIds)) {
    segmentIds = probeSegmentIds(fileNames, labels, hash);
  }
  const idOrder = getIdOrder(segmentIds);
  return {
    ids: BigUint64Array.from(idOrder, (index) => segmentIds[index]),
    fileNames: Array.from(idOrder, (index) => fileNames[index]),
  };
}

// Without a zero or repeated first hash, probing leaves every label at its first hash.
function hasZeroOrRepeatedId(segmentIds: BigUint64Array): boolean {
  const sortedIds = segmentIds.slice().sort();
  if (sortedIds[0] === 0n) return true;
  for (let index = 1; index < sortedIds.length; ++index) {
    if (sortedIds[index] === sortedIds[index - 1]) return true;
  }
  return false;
}

// Labels are probed in UTF-16 code unit order, not locale order.
function probeSegmentIds(
  fileNames: readonly string[],
  labels: readonly string[],
  hash: (key: string) => bigint,
): BigUint64Array<ArrayBuffer> {
  const labelOrder = Array.from(labels.keys()).sort((a, b) =>
    labels[a] < labels[b] ? -1 : labels[a] > labels[b] ? 1 : 0,
  );
  for (let position = 1; position < labelOrder.length; ++position) {
    const earlier = labelOrder[position - 1];
    const later = labelOrder[position];
    if (labels[earlier] === labels[later]) {
      throw new Error(
        `SWC files ${JSON.stringify(fileNames[earlier])} and ${JSON.stringify(fileNames[later])} have the same name`,
      );
    }
  }
  const takenIds = new Set<bigint>();
  const segmentIds = new BigUint64Array(labels.length);
  for (const index of labelOrder) {
    const label = labels[index];
    let segmentId = hash(label);
    for (
      let attempt = 1;
      segmentId === 0n || takenIds.has(segmentId);
      ++attempt
    ) {
      segmentId = hash(`${label}\0${attempt}`);
    }
    takenIds.add(segmentId);
    segmentIds[index] = segmentId;
  }
  return segmentIds;
}

// Compares 32-bit halves, because a `bigint` comparator is slow.
function getIdOrder(segmentIds: BigUint64Array): Uint32Array {
  const high = Uint32Array.from(segmentIds, (segmentId) =>
    Number(segmentId >> 32n),
  );
  const low = Uint32Array.from(segmentIds, (segmentId) =>
    Number(segmentId & 0xffffffffn),
  );
  return Uint32Array.from(segmentIds, (_, index) => index).sort(
    (a, b) => high[a] - high[b] || low[a] - low[b],
  );
}
