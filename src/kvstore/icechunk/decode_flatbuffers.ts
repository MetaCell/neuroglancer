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

// Readers for the Icechunk flatbuffers schemas in
// https://github.com/earth-mover/icechunk/tree/main/icechunk-format/flatbuffers
// Fields are addressed by their index in the schema table.

import { decodeZstdStringsWithDictionary } from "#src/async_computation/decode_zstd_dictionary_request.js";
import { requestAsyncComputation } from "#src/async_computation/request.js";
import { crockfordBase32Encode } from "#src/kvstore/icechunk/crockford_base32.js";
import type {
  IcechunkFile,
  NodeId,
} from "#src/kvstore/icechunk/decode_utils.js";
import {
  decodeIcechunkFile,
  PayloadEncoding,
} from "#src/kvstore/icechunk/decode_utils.js";
import type { ChunkPayload, Manifest } from "#src/kvstore/icechunk/manifest.js";
import type { RepoInfo } from "#src/kvstore/icechunk/repo_info.js";
import type {
  ArrayNode,
  ManifestRef,
  NodeSnapshot,
  Snapshot,
  SnapshotId,
} from "#src/kvstore/icechunk/snapshot.js";
import { defaultStringCompare } from "#src/util/string.js";

const MAX_SAFE_INTEGER_BIGINT = BigInt(Number.MAX_SAFE_INTEGER);

const OBJECT_ID_12_SIZE = 12;
const OBJECT_ID_8_SIZE = 8;
const DIMENSION_SHAPE_SIZE = 16;
const CHUNK_INDEX_RANGE_SIZE = 8;

const NODE_DATA_ARRAY = 1;
const NODE_DATA_GROUP = 2;

const COMPRESSION_ALGORITHM_ZSTD_DICTIONARY = 1;

// Icechunk decodes compressed locations into a buffer of this size.
const MAX_DECOMPRESSED_LOCATION_SIZE = 1024;

const REPO_INFO_FILE_TYPE = 6;

// Reads past the end of the content throw, as `DataView` checks every access.
class FlatbuffersReader {
  private readonly view: DataView;
  private readonly textDecoder = new TextDecoder("utf-8", { fatal: true });

  constructor(private readonly content: Uint8Array<ArrayBuffer>) {
    this.view = new DataView(
      content.buffer,
      content.byteOffset,
      content.byteLength,
    );
  }

  root(): number {
    return this.view.getUint32(0, true);
  }

  private fieldOffset(table: number, field: number): number {
    const vtable = table - this.view.getInt32(table, true);
    const vtableSize = this.view.getUint16(vtable, true);
    const slot = 4 + 2 * field;
    return slot < vtableSize ? this.view.getUint16(vtable + slot, true) : 0;
  }

  private indirect(position: number): number {
    return position + this.view.getUint32(position, true);
  }

  private checkRange(start: number, length: number) {
    if (start < 0 || start + length > this.content.length) {
      throw new RangeError(
        `Range [${start}, ${start + length}) is outside the ${this.content.length}-byte content`,
      );
    }
  }

  has(table: number, field: number): boolean {
    return this.fieldOffset(table, field) !== 0;
  }

  table(table: number, field: number): number | undefined {
    const offset = this.fieldOffset(table, field);
    return offset === 0 ? undefined : this.indirect(table + offset);
  }

  struct(table: number, field: number): number | undefined {
    const offset = this.fieldOffset(table, field);
    return offset === 0 ? undefined : table + offset;
  }

  uint8(table: number, field: number, defaultValue: number): number {
    const offset = this.fieldOffset(table, field);
    return offset === 0 ? defaultValue : this.view.getUint8(table + offset);
  }

  uint32(table: number, field: number): number {
    const offset = this.fieldOffset(table, field);
    return offset === 0 ? 0 : this.uint32At(table + offset);
  }

  uint64(table: number, field: number): number {
    const offset = this.fieldOffset(table, field);
    return offset === 0 ? 0 : this.uint64At(table + offset);
  }

  uint64At(position: number): number {
    const value = this.view.getBigUint64(position, true);
    if (value > MAX_SAFE_INTEGER_BIGINT) {
      throw new Error(`Number outside supported range: ${value}`);
    }
    return Number(value);
  }

  uint32At(position: number): number {
    return this.view.getUint32(position, true);
  }

  vector(
    table: number,
    field: number,
    elementSize: number,
  ): { start: number; length: number } | undefined {
    const offset = this.fieldOffset(table, field);
    if (offset === 0) return undefined;
    const lengthPosition = this.indirect(table + offset);
    const length = this.view.getUint32(lengthPosition, true);
    const start = lengthPosition + 4;
    this.checkRange(start, length * elementSize);
    return { start, length };
  }

  tables(table: number, field: number): number[] {
    const vector = this.vector(table, field, 4);
    if (vector === undefined) return [];
    const result = new Array<number>(vector.length);
    for (let i = 0; i < vector.length; ++i) {
      result[i] = this.indirect(vector.start + 4 * i);
    }
    return result;
  }

  bytes(table: number, field: number): Uint8Array<ArrayBuffer> | undefined {
    const vector = this.vector(table, field, 1);
    if (vector === undefined) return undefined;
    return this.content.subarray(vector.start, vector.start + vector.length);
  }

  strings(table: number, field: number): string[] {
    return this.tables(table, field).map((position) => {
      const length = this.view.getUint32(position, true);
      this.checkRange(position + 4, length);
      return this.textDecoder.decode(
        this.content.subarray(position + 4, position + 4 + length),
      );
    });
  }

  string(table: number, field: number): string | undefined {
    const bytes = this.bytes(table, field);
    return bytes === undefined ? undefined : this.textDecoder.decode(bytes);
  }

  objectId(position: number, size: number): string {
    this.checkRange(position, size);
    return crockfordBase32Encode(
      this.content.subarray(position, position + size),
    );
  }
}

function required<T>(value: T | undefined, name: string): T {
  if (value === undefined) {
    throw new Error(`Missing required field: ${name}`);
  }
  return value;
}

const SNAPSHOT_NODES = 2;
// NodeSnapshot fields.
const NODE_ID = 0;
const NODE_PATH = 1;
const NODE_USER_DATA = 2;
// A union takes two fields: its type, then its value.
const NODE_DATA_TYPE = 3;
const NODE_DATA = 4;
// ArrayNodeData fields.
const ARRAY_SHAPE = 0;
const ARRAY_MANIFESTS = 2;
const ARRAY_SHAPE_V2 = 3;
// DimensionShapeV2 fields.
const DIMENSION_NUM_CHUNKS = 1;
const MANIFEST_REF_OBJECT_ID = 0;
const MANIFEST_REF_EXTENTS = 1;

export function decodeFlatbuffersSnapshot(file: IcechunkFile): Snapshot {
  const reader = new FlatbuffersReader(file.content);
  const root = reader.root();
  const nodes = reader
    .tables(root, SNAPSHOT_NODES)
    .map((node) => decodeNode(reader, node))
    .sort((a, b) => defaultStringCompare(a.path, b.path));
  return { nodes, estimatedSize: file.estimatedSize };
}

function decodeNode(reader: FlatbuffersReader, node: number): NodeSnapshot {
  const path = required(reader.string(node, NODE_PATH), "path");
  const userData = required(reader.bytes(node, NODE_USER_DATA), "user_data");
  const nodeData = required(reader.table(node, NODE_DATA), "node_data");
  const nodeDataType = reader.uint8(node, NODE_DATA_TYPE, 0);
  if (nodeDataType !== NODE_DATA_ARRAY && nodeDataType !== NODE_DATA_GROUP) {
    throw new Error(`Unknown node data type: ${nodeDataType}`);
  }
  return {
    id: reader.objectId(
      required(reader.struct(node, NODE_ID), "id"),
      OBJECT_ID_8_SIZE,
    ),
    path: path === "/" ? "" : path.slice(1) + "/",
    zarrJson: userData.slice(),
    array:
      nodeDataType === NODE_DATA_ARRAY
        ? decodeArrayNode(reader, nodeData)
        : undefined,
  };
}

function decodeArrayNode(reader: FlatbuffersReader, array: number): ArrayNode {
  let numChunks: number[];
  if (reader.has(array, ARRAY_SHAPE_V2)) {
    numChunks = reader
      .tables(array, ARRAY_SHAPE_V2)
      .map((dimension) => reader.uint32(dimension, DIMENSION_NUM_CHUNKS));
  } else {
    const shape = required(
      reader.vector(array, ARRAY_SHAPE, DIMENSION_SHAPE_SIZE),
      "shape",
    );
    numChunks = new Array<number>(shape.length);
    for (let i = 0; i < shape.length; ++i) {
      const position = shape.start + i * DIMENSION_SHAPE_SIZE;
      const arrayLength = reader.uint64At(position);
      const chunkLength = reader.uint64At(position + 8);
      numChunks[i] =
        chunkLength === 0 ? 0 : Math.ceil(arrayLength / chunkLength);
    }
  }
  const manifests = reader
    .tables(array, ARRAY_MANIFESTS)
    .map((manifestRef): ManifestRef => {
      const extents = required(
        reader.vector(
          manifestRef,
          MANIFEST_REF_EXTENTS,
          CHUNK_INDEX_RANGE_SIZE,
        ),
        "extents",
      );
      const lower = new Array<number>(extents.length);
      const upper = new Array<number>(extents.length);
      for (let i = 0; i < extents.length; ++i) {
        const position = extents.start + i * CHUNK_INDEX_RANGE_SIZE;
        lower[i] = reader.uint32At(position);
        upper[i] = reader.uint32At(position + 4);
      }
      return {
        id: reader.objectId(
          required(
            reader.struct(manifestRef, MANIFEST_REF_OBJECT_ID),
            "object_id",
          ),
          OBJECT_ID_12_SIZE,
        ),
        extents: [lower, upper],
      };
    });
  return { numChunks, manifests };
}

const MANIFEST_ARRAYS = 1;
const MANIFEST_LOCATION_DICTIONARY = 2;
const MANIFEST_COMPRESSION_ALGORITHM = 3;
const ARRAY_MANIFEST_NODE_ID = 0;
const ARRAY_MANIFEST_REFS = 1;
const CHUNK_REF_INDEX = 0;
const CHUNK_REF_INLINE = 1;
const CHUNK_REF_OFFSET = 2;
const CHUNK_REF_LENGTH = 3;
const CHUNK_REF_CHUNK_ID = 4;
const CHUNK_REF_LOCATION = 5;
const CHUNK_REF_COMPRESSED_LOCATION = 8;

export async function decodeFlatbuffersManifest(
  file: IcechunkFile,
  signal: AbortSignal | undefined,
): Promise<Manifest> {
  const reader = new FlatbuffersReader(file.content);
  const root = reader.root();
  const chunks = new Map<NodeId, Map<string, ChunkPayload>>();
  const compressedLocations: {
    payloads: Map<string, ChunkPayload>;
    chunkKey: string;
    frame: Uint8Array;
    offset: number;
    length: number;
  }[] = [];
  for (const arrayManifest of reader.tables(root, MANIFEST_ARRAYS)) {
    const nodeId = reader.objectId(
      required(reader.struct(arrayManifest, ARRAY_MANIFEST_NODE_ID), "node_id"),
      OBJECT_ID_8_SIZE,
    );
    const payloads = new Map<string, ChunkPayload>();
    chunks.set(nodeId, payloads);
    for (const chunkRef of reader.tables(arrayManifest, ARRAY_MANIFEST_REFS)) {
      const index = required(
        reader.vector(chunkRef, CHUNK_REF_INDEX, 4),
        "index",
      );
      const position = new Array<number>(index.length);
      for (let i = 0; i < index.length; ++i) {
        position[i] = reader.uint32At(index.start + 4 * i);
      }
      const chunkKey = position.join();
      const offset = reader.uint64(chunkRef, CHUNK_REF_OFFSET);
      const length = reader.uint64(chunkRef, CHUNK_REF_LENGTH);
      const chunkId = reader.struct(chunkRef, CHUNK_REF_CHUNK_ID);
      if (chunkId !== undefined) {
        payloads.set(chunkKey, {
          kind: "native",
          chunkId: reader.objectId(chunkId, OBJECT_ID_12_SIZE),
          offset,
          length,
        });
        continue;
      }
      const compressedLocation = reader.bytes(
        chunkRef,
        CHUNK_REF_COMPRESSED_LOCATION,
      );
      if (compressedLocation !== undefined) {
        compressedLocations.push({
          payloads,
          chunkKey,
          frame: compressedLocation,
          offset,
          length,
        });
        continue;
      }
      const location = reader.string(chunkRef, CHUNK_REF_LOCATION);
      if (location !== undefined) {
        payloads.set(chunkKey, { kind: "virtual", location, offset, length });
        continue;
      }
      const inline = reader.bytes(chunkRef, CHUNK_REF_INLINE);
      if (inline !== undefined) {
        payloads.set(chunkKey, { kind: "inline", data: inline.slice() });
        continue;
      }
      throw new Error(
        `Chunk ${chunkKey} has no chunk id, location or inline data`,
      );
    }
  }
  if (compressedLocations.length > 0) {
    const dictionary = reader.bytes(root, MANIFEST_LOCATION_DICTIONARY);
    if (
      dictionary === undefined ||
      reader.uint8(
        root,
        MANIFEST_COMPRESSION_ALGORITHM,
        COMPRESSION_ALGORITHM_ZSTD_DICTIONARY,
      ) !== COMPRESSION_ALGORITHM_ZSTD_DICTIONARY
    ) {
      throw new Error(
        "Manifest has compressed locations but no zstd location dictionary",
      );
    }
    const frameEnds = new Uint32Array(compressedLocations.length);
    let totalLength = 0;
    compressedLocations.forEach(({ frame }, i) => {
      totalLength += frame.length;
      frameEnds[i] = totalLength;
    });
    const frames = new Uint8Array(totalLength);
    compressedLocations.forEach(({ frame }, i) => {
      frames.set(frame, frameEnds[i] - frame.length);
    });
    const dictionaryCopy = dictionary.slice();
    const locations = await requestAsyncComputation(
      decodeZstdStringsWithDictionary,
      signal,
      [dictionaryCopy.buffer, frames.buffer, frameEnds.buffer],
      dictionaryCopy,
      frames,
      frameEnds,
      MAX_DECOMPRESSED_LOCATION_SIZE,
    );
    compressedLocations.forEach(({ payloads, chunkKey, offset, length }, i) => {
      payloads.set(chunkKey, {
        kind: "virtual",
        location: locations[i],
        offset,
        length,
      });
    });
  }
  return { chunks, estimatedSize: file.estimatedSize };
}

const REPO_TAGS = 1;
const REPO_BRANCHES = 2;
const REPO_DELETED_TAGS = 3;
const REPO_SNAPSHOTS = 4;
const REF_NAME = 0;
const REF_SNAPSHOT_INDEX = 1;
const SNAPSHOT_INFO_ID = 0;

export async function decodeRepoInfo(
  buffer: ArrayBuffer,
  signal: AbortSignal | undefined,
): Promise<RepoInfo> {
  const file = await decodeIcechunkFile(buffer, REPO_INFO_FILE_TYPE, signal);
  if (file.encoding !== PayloadEncoding.FLATBUFFERS) {
    throw new Error("Expected a flatbuffers repo info file");
  }
  const reader = new FlatbuffersReader(file.content);
  const root = reader.root();
  const snapshotInfos = reader.tables(root, REPO_SNAPSHOTS);
  const decodeRefs = (field: number) => {
    const refs = new Map<string, SnapshotId>();
    for (const ref of reader.tables(root, field)) {
      const name = required(reader.string(ref, REF_NAME), "name");
      const snapshotIndex = reader.uint32(ref, REF_SNAPSHOT_INDEX);
      const snapshotInfo = required(
        snapshotInfos[snapshotIndex],
        `snapshot ${snapshotIndex}`,
      );
      refs.set(
        name,
        reader.objectId(
          required(reader.struct(snapshotInfo, SNAPSHOT_INFO_ID), "id"),
          OBJECT_ID_12_SIZE,
        ),
      );
    }
    return refs;
  };
  return {
    branches: decodeRefs(REPO_BRANCHES),
    tags: decodeRefs(REPO_TAGS),
    deletedTags: new Set(reader.strings(root, REPO_DELETED_TAGS)),
    estimatedSize: file.estimatedSize,
  };
}
