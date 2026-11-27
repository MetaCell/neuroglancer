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

import type { SharedKvStoreContextCounterpart } from "#src/kvstore/backend.js";
import {
  FileByteRangeHandle,
  handleByteRangeRequestFromUint8Array,
} from "#src/kvstore/byte_range/file_handle.js";
import type { ChunkId } from "#src/kvstore/icechunk/decode_utils.js";
import type { ChunkPayload } from "#src/kvstore/icechunk/manifest.js";
import { getManifest } from "#src/kvstore/icechunk/metadata_cache.js";
import type {
  ArrayNode,
  ManifestExtents,
  NodeSnapshot,
  Snapshot,
} from "#src/kvstore/icechunk/snapshot.js";
import { findNode } from "#src/kvstore/icechunk/snapshot.js";
import type {
  DriverReadOptions,
  ReadResponse,
  StatOptions,
  StatResponse,
} from "#src/kvstore/index.js";
import { pipelineUrlJoin } from "#src/kvstore/url.js";
import type { ProgressOptions } from "#src/util/progress_listener.js";

export interface ResolvedIcechunkPath {
  node: NodeSnapshot;
  chunk?: number[];
}

export function resolveIcechunkPath(
  snapshot: Snapshot,
  path: string,
): ResolvedIcechunkPath | undefined {
  let nodePath: string;
  let chunk: number[] | undefined;
  const zarrJsonMatch = path.match(/(?:^|\/)(zarr\.json)$/);
  if (zarrJsonMatch !== null) {
    nodePath = path.slice(0, -zarrJsonMatch[1].length);
  } else {
    const chunkMatch = path.match(/c(?:[./][0-9]+)*$/);
    if (chunkMatch === null) {
      return undefined;
    }
    nodePath = path.slice(0, -chunkMatch[0].length);
    const parts = chunkMatch[0].split(/[./]/);
    const n = parts.length - 1;
    chunk = new Array<number>(n);
    for (let i = 0; i < n; ++i) {
      chunk[i] = Number(parts[i + 1]);
    }
  }
  const node = findNode(snapshot, nodePath);
  if (chunk === undefined) {
    return { node };
  }
  if (node.array === undefined) {
    // chunk path not valid for groups
    return undefined;
  }
  const { numChunks } = node.array;
  const rank = numChunks.length;
  if (rank !== chunk.length) {
    return undefined;
  }
  for (let i = 0; i < rank; ++i) {
    if (chunk[i] >= numChunks[i]) {
      return undefined;
    }
  }
  return { node, chunk };
}

function manifestExtentsContain(
  [lower, upper]: ManifestExtents,
  chunk: number[],
) {
  for (let i = 0, n = chunk.length; i < n; ++i) {
    const c = chunk[i];
    if (c < lower[i] || c >= upper[i]) return false;
  }
  return true;
}

export async function resolveChunkPayload(
  sharedKvStoreContext: SharedKvStoreContextCounterpart,
  baseUrl: string,
  node: NodeSnapshot,
  chunk: number[],
  options: Partial<ProgressOptions>,
): Promise<ChunkPayload | undefined> {
  const { manifests } = node.array as ArrayNode;
  const chunkKey = chunk.join();
  const nodeId = node.id;
  for (const manifestRef of manifests) {
    if (!manifestExtentsContain(manifestRef.extents, chunk)) continue;
    const manifest = await getManifest(
      sharedKvStoreContext,
      baseUrl,
      manifestRef.id,
      options,
    );
    const chunks = manifest.chunks.get(nodeId);
    if (chunks === undefined) continue;
    const chunkPayload = chunks.get(chunkKey);
    if (chunkPayload !== undefined) return chunkPayload;
  }
  return undefined;
}

export async function stat(
  sharedKvStoreContext: SharedKvStoreContextCounterpart,
  baseUrl: string,
  snapshot: Snapshot,
  path: string,
  options: StatOptions,
): Promise<StatResponse | undefined> {
  const resolvedPath = resolveIcechunkPath(snapshot, path);
  if (resolvedPath === undefined) return undefined;
  const { node, chunk } = resolvedPath;
  if (chunk === undefined) {
    // zarr.json file.
    return { totalSize: undefined };
  }
  const payload = await resolveChunkPayload(
    sharedKvStoreContext,
    baseUrl,
    node,
    chunk,
    options,
  );
  if (payload === undefined) return undefined;
  const totalSize =
    payload.kind === "inline" ? payload.data.length : payload.length;
  return { totalSize };
}

async function readFromChunkPayload(
  sharedKvStoreContext: SharedKvStoreContextCounterpart,
  baseUrl: string,
  payload: ChunkPayload,
  options: DriverReadOptions,
): Promise<ReadResponse> {
  if (payload.kind === "inline") {
    return handleByteRangeRequestFromUint8Array(
      payload.data,
      options.byteRange,
    );
  }
  const { offset, length } = payload;
  if (
    payload.kind === "virtual" &&
    payload.location.startsWith(VIRTUAL_CHUNK_CONTAINER_SCHEME)
  ) {
    throw new Error(
      `Virtual chunk locations relative to a virtual chunk container are not supported: ${payload.location}`,
    );
  }
  const url =
    payload.kind === "virtual"
      ? payload.location
      : getChunkUrl(baseUrl, payload.chunkId);
  return new FileByteRangeHandle(
    sharedKvStoreContext.kvStoreContext.getFileHandle(url),
    { offset, length },
  ).read(options);
}

// The container's URL prefix is in the repository config, which is not read.
const VIRTUAL_CHUNK_CONTAINER_SCHEME = "vcc://";

function getChunkUrl(baseUrl: string, id: ChunkId) {
  return pipelineUrlJoin(baseUrl, `chunks/${id}`);
}

export async function read(
  sharedKvStoreContext: SharedKvStoreContextCounterpart,
  baseUrl: string,
  snapshot: Snapshot,
  path: string,
  options: DriverReadOptions,
): Promise<ReadResponse | undefined> {
  const resolvedPath = resolveIcechunkPath(snapshot, path);
  if (resolvedPath === undefined) return undefined;
  const { node, chunk } = resolvedPath;
  if (chunk === undefined) {
    return handleByteRangeRequestFromUint8Array(
      node.zarrJson,
      options.byteRange,
    );
  }
  const payload = await resolveChunkPayload(
    sharedKvStoreContext,
    baseUrl,
    node,
    chunk,
    options,
  );
  if (payload === undefined) return undefined;
  return readFromChunkPayload(sharedKvStoreContext, baseUrl, payload, options);
}
