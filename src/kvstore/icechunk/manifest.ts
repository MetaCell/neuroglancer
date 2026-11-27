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

import { decodeFlatbuffersManifest } from "#src/kvstore/icechunk/decode_flatbuffers.js";
import { decodeMsgpackManifest } from "#src/kvstore/icechunk/decode_msgpack.js";
import type {
  ChunkId,
  ManifestId,
  NodeId,
} from "#src/kvstore/icechunk/decode_utils.js";
import {
  decodeIcechunkFile,
  PayloadEncoding,
} from "#src/kvstore/icechunk/decode_utils.js";
import { pipelineUrlJoin } from "#src/kvstore/url.js";

export type ChunkPayload =
  | { readonly kind: "inline"; readonly data: Uint8Array<ArrayBuffer> }
  | {
      readonly kind: "native";
      readonly chunkId: ChunkId;
      readonly offset: number;
      readonly length: number;
    }
  | {
      readonly kind: "virtual";
      readonly location: string;
      readonly offset: number;
      readonly length: number;
    };

export interface Manifest {
  // Keyed by node, then by the chunk grid position joined with ",".
  readonly chunks: ReadonlyMap<NodeId, ReadonlyMap<string, ChunkPayload>>;
  readonly estimatedSize: number;
}

const MANIFEST_FILE_TYPE = 2;

export async function decodeManifest(
  buffer: ArrayBuffer,
  signal: AbortSignal | undefined,
): Promise<Manifest> {
  const file = await decodeIcechunkFile(buffer, MANIFEST_FILE_TYPE, signal);
  return file.encoding === PayloadEncoding.MSGPACK
    ? decodeMsgpackManifest(file)
    : decodeFlatbuffersManifest(file, signal);
}

export function getManifestUrl(baseUrl: string, id: ManifestId): string {
  return pipelineUrlJoin(baseUrl, `manifests/${id}`);
}
