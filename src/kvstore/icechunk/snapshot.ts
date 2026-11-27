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

import { decodeFlatbuffersSnapshot } from "#src/kvstore/icechunk/decode_flatbuffers.js";
import { decodeMsgpackSnapshot } from "#src/kvstore/icechunk/decode_msgpack.js";
import type { ManifestId, NodeId } from "#src/kvstore/icechunk/decode_utils.js";
import {
  decodeIcechunkFile,
  PayloadEncoding,
} from "#src/kvstore/icechunk/decode_utils.js";
import { pipelineUrlJoin } from "#src/kvstore/url.js";
import { binarySearch } from "#src/util/array.js";
import { defaultStringCompare } from "#src/util/string.js";

export type SnapshotId = string;

export type ManifestExtents = readonly [
  lower: readonly number[],
  upper: readonly number[],
];

export interface ManifestRef {
  readonly id: ManifestId;
  readonly extents: ManifestExtents;
}

export interface ArrayNode {
  readonly numChunks: readonly number[];
  readonly manifests: readonly ManifestRef[];
}

export interface NodeSnapshot {
  readonly id: NodeId;
  // Empty for the root, else ends in "/".
  readonly path: string;
  readonly zarrJson: Uint8Array<ArrayBuffer>;
  readonly array: ArrayNode | undefined;
}

export interface Snapshot {
  // Sorted by `path`.
  readonly nodes: readonly NodeSnapshot[];
  readonly estimatedSize: number;
}

const SNAPSHOT_FILE_TYPE = 1;

export async function decodeSnapshot(
  buffer: ArrayBuffer,
  signal: AbortSignal | undefined,
): Promise<Snapshot> {
  const file = await decodeIcechunkFile(buffer, SNAPSHOT_FILE_TYPE, signal);
  return file.encoding === PayloadEncoding.MSGPACK
    ? decodeMsgpackSnapshot(file)
    : decodeFlatbuffersSnapshot(file);
}

export function findNode(snapshot: Snapshot, path: string) {
  const { nodes } = snapshot;
  const index = binarySearch(nodes, path, (a, b) =>
    defaultStringCompare(a, b.path),
  );
  if (index < 0) {
    throw new Error(`Node not found: ${JSON.stringify(path)}`);
  }
  return nodes[index];
}

export function getSnapshotUrl(baseUrl: string, id: SnapshotId): string {
  return pipelineUrlJoin(baseUrl, `snapshots/${id}`);
}
