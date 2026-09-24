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
// Labels sort by UTF-16 code unit, not by locale.
export function assignSegmentIds(
  fileNames: readonly string[],
  hash: (key: string) => bigint,
): SwcSegments {
  const fileNameByLabel = new Map<string, string>();
  for (const fileName of fileNames) {
    const label = getSwcLabel(fileName);
    const otherFileName = fileNameByLabel.get(label);
    if (otherFileName !== undefined) {
      throw new Error(
        `SWC files ${JSON.stringify(otherFileName)} and ${JSON.stringify(fileName)} have the same name`,
      );
    }
    fileNameByLabel.set(label, fileName);
  }
  const fileNameBySegmentId = new Map<bigint, string>();
  for (const label of Array.from(fileNameByLabel.keys()).sort()) {
    let segmentId = hash(label);
    for (
      let attempt = 1;
      segmentId === 0n || fileNameBySegmentId.has(segmentId);
      ++attempt
    ) {
      segmentId = hash(`${label}\0${attempt}`);
    }
    fileNameBySegmentId.set(segmentId, fileNameByLabel.get(label)!);
  }
  const ids = BigUint64Array.from(fileNameBySegmentId.keys()).sort();
  return {
    ids,
    fileNames: Array.from(
      ids,
      (segmentId) => fileNameBySegmentId.get(segmentId)!,
    ),
  };
}
