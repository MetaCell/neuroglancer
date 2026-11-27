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

import * as v from "valibot";
import type { IcechunkFile } from "#src/kvstore/icechunk/decode_utils.js";
import {
  DataId12,
  unpackMsgpack,
  Integer,
  ManifestId,
  NodeId,
  ChunkId,
  parseDecodedMsgpack,
  tupleToObject,
} from "#src/kvstore/icechunk/decode_utils.js";
import type { ChunkPayload, Manifest } from "#src/kvstore/icechunk/manifest.js";
import type { ArrayNode, Snapshot } from "#src/kvstore/icechunk/snapshot.js";
import { defaultStringCompare } from "#src/util/string.js";

// const ATTRIBUTES_FILE_TYPE = 3;

const MsgpackSnapshotId = DataId12;
const AttributesId = DataId12;

const ManifestFileInfo = tupleToObject({
  id: ManifestId,
  sizeBytes: Integer,
  numRows: Integer,
});

const AttributeFileInfo = tupleToObject({
  id: AttributesId,
});

// const UserAttributesRef = tupleToObject({
//   objectId: AttributesId,
//   location: Integer,
// });

const UserAttributesSnapshot = v.pipe(
  v.map(v.string(), v.any()),
  v.transform<Map<string, any>, Record<string, any>>(Object.fromEntries),
  v.union([
    v.strictObject({
      Inline: v.any(),
    }),
  ]),
  // v.map(v.picklist(["Ref"]), UserAttributesRef),
);

const ChunkKeyEncoding = v.picklist(["Slash", "Dot"]);

const Configuration = v.map(v.string(), v.any());

const Codec = tupleToObject({
  name: v.string(),
  configuration: Configuration,
});

const FillValue = v.pipe(
  v.map(v.string(), v.any()),
  v.transform((obj) => {
    const values = Array.from(obj.values());
    if (values.length !== 1) {
      throw new Error(
        `Expected a single key, but received: ${JSON.stringify(Array.from(obj.keys()))}`,
      );
    }
    return values[0];
  }),
);

const StorageTransformer = tupleToObject({
  name: v.string(),
  configuration: Configuration,
});

const DimensionNames = v.array(v.nullable(v.string()));

const MsgpackZarrArrayMetadata = tupleToObject({
  shape: v.array(Integer),
  dataType: v.string(),
  chunkShape: v.array(Integer),
  chunkKeyEncoding: ChunkKeyEncoding,
  fillValue: FillValue,
  codecs: v.array(Codec),
  storageTransformers: v.array(StorageTransformer),
  dimensionNames: v.nullable(DimensionNames),
});

type MsgpackZarrArrayMetadata = v.InferOutput<typeof MsgpackZarrArrayMetadata>;

const ChunkIndices = v.array(Integer);

const ManifestExtents = v.strictTuple([ChunkIndices, ChunkIndices]);
type ManifestExtents = v.InferOutput<typeof ManifestExtents>;

const MsgpackManifestRef = tupleToObject({
  objectId: ManifestId,
  extents: ManifestExtents,
});
type MsgpackManifestRef = v.InferOutput<typeof MsgpackManifestRef>;

const NodeDataGroup = v.picklist(["Group"]);
const MsgpackNodeDataArray = v.strictObject({
  Array: tupleToObject({
    metadata: MsgpackZarrArrayMetadata,
    manifests: v.array(MsgpackManifestRef),
  }),
});
type MsgpackNodeDataArray = v.InferOutput<typeof MsgpackNodeDataArray>;

const NodeData = v.union([
  NodeDataGroup,
  v.pipe(
    v.map(v.string(), v.any()),
    v.transform<Map<string, any>, Record<string, any>>(Object.fromEntries),
    MsgpackNodeDataArray,
  ),
]);

const MsgpackNodeSnapshot = tupleToObject({
  id: NodeId,
  path: v.pipe(
    v.string(),
    v.transform((s) => (s === "/" ? "" : s.slice(1) + "/")),
  ),
  userAttributes: UserAttributesSnapshot,
  nodeData: NodeData,
});

type MsgpackNodeSnapshot = v.InferOutput<typeof MsgpackNodeSnapshot>;

const Nodes = v.pipe(
  v.map(v.string(), MsgpackNodeSnapshot),
  v.transform((obj) =>
    Array.from(obj.values()).sort((a, b) =>
      defaultStringCompare(a.path, b.path),
    ),
  ),
);

const MsgpackSnapshot = tupleToObject({
  id: MsgpackSnapshotId,
  parentId: v.nullable(MsgpackSnapshotId),
  flushedAt: v.string(),
  message: v.string(),
  metadata: v.record(v.string(), v.any()),
  manifestFiles: v.pipe(
    v.array(ManifestFileInfo),
    v.transform((obj) => {
      const map = new Map();
      for (const entry of obj) {
        map.set(entry.id, entry);
      }
      return map;
    }),
  ),
  attributeFiles: v.array(AttributeFileInfo),
  nodes: Nodes,
});

type MsgpackSnapshot = v.InferOutput<typeof MsgpackSnapshot>;

export function decodeMsgpackSnapshot(file: IcechunkFile): Snapshot {
  const snapshot = parseDecodedMsgpack(
    MsgpackSnapshot,
    "snapshot",
    unpackMsgpack(file),
  );
  const textEncoder = new TextEncoder();
  return {
    nodes: snapshot.nodes.map((node) => ({
      id: node.id,
      path: node.path,
      zarrJson: textEncoder.encode(encodeZarrJson(node)),
      array: node.nodeData === "Group" ? undefined : toArrayNode(node.nodeData),
    })),
    estimatedSize: snapshot.estimatedSize,
  };
}

function toArrayNode({
  Array: { metadata, manifests },
}: MsgpackNodeDataArray): ArrayNode {
  const { shape, chunkShape } = metadata;
  return {
    numChunks: shape.map((size, i) =>
      chunkShape[i] === 0 ? 0 : Math.ceil(size / chunkShape[i]),
    ),
    manifests: manifests.map(({ objectId, extents }) => ({
      id: objectId,
      extents,
    })),
  };
}

function encodeZarrJson(node: MsgpackNodeSnapshot) {
  const { userAttributes, nodeData } = node;
  let attributes: Map<string, any>;
  if (userAttributes === null) {
    attributes = new Map();
  } else {
    attributes = userAttributes.Inline;
  }

  const obj =
    nodeData !== "Group"
      ? encodeArrayZarrJson(nodeData.Array.metadata, attributes)
      : { zarr_format: 3, node_type: "group", attributes };

  return JSON.stringify(obj, (_key, value) => {
    if (value instanceof Map) {
      return Object.fromEntries(value);
    }
    return value;
  });
}

function encodeArrayZarrJson(
  metadata: MsgpackZarrArrayMetadata,
  attributes: Record<string, any>,
) {
  const {
    shape,
    chunkShape,
    chunkKeyEncoding,
    dataType,
    fillValue,
    codecs,
    storageTransformers,
    dimensionNames,
  } = metadata;
  return {
    zarr_format: 3,
    node_type: "array",
    shape,
    data_type: dataType,
    chunk_grid: { name: "regular", configuration: { chunk_shape: chunkShape } },
    chunk_key_encoding: {
      name: "default",
      configuration: { separator: chunkKeyEncoding === "Dot" ? "." : "/" },
    },
    fill_value: fillValue,
    codecs,
    storage_transformers: storageTransformers,
    dimension_names: dimensionNames ?? undefined,
    attributes,
  };
}

const InlineChunkPayload = v.strictObject({
  Inline: v.instance(Uint8Array),
});

const Chunksum = v.any();

const VirtualChunkLocation = v.string();

const VirtualChunkRef = tupleToObject({
  location: VirtualChunkLocation,
  offset: Integer,
  length: Integer,
  chunksum: Chunksum,
});

const VirtualChunkRefPayload = v.strictObject({
  Virtual: VirtualChunkRef,
});

const ChunkRef = tupleToObject({
  id: ChunkId,
  offset: Integer,
  length: Integer,
});

const ChunkRefPayload = v.strictObject({
  Ref: ChunkRef,
});

const MsgpackChunkPayload = v.pipe(
  v.map(v.string(), v.any()),
  v.transform<Map<string, any>, Record<string, any>>(Object.fromEntries),
  v.union([InlineChunkPayload, VirtualChunkRefPayload, ChunkRefPayload]),
);

type MsgpackChunkPayload = v.InferOutput<typeof MsgpackChunkPayload>;

const MsgpackManifest = tupleToObject({
  id: ManifestId,
  chunks: v.map(
    NodeId,
    v.map(
      v.pipe(
        v.array(Integer),
        v.transform((chunk) => chunk.join()),
      ),
      MsgpackChunkPayload,
    ),
  ),
});

type MsgpackManifest = v.InferOutput<typeof MsgpackManifest>;

export function decodeMsgpackManifest(file: IcechunkFile): Manifest {
  const manifest = parseDecodedMsgpack(
    MsgpackManifest,
    "chunk manifest",
    unpackMsgpack(file),
  );
  const chunks = new Map<NodeId, Map<string, ChunkPayload>>();
  for (const [nodeId, nodeChunks] of manifest.chunks) {
    const payloads = new Map<string, ChunkPayload>();
    for (const [chunkKey, payload] of nodeChunks) {
      payloads.set(chunkKey, toChunkPayload(payload));
    }
    chunks.set(nodeId, payloads);
  }
  return { chunks, estimatedSize: manifest.estimatedSize };
}

function toChunkPayload(payload: MsgpackChunkPayload): ChunkPayload {
  if ("Inline" in payload) {
    return {
      kind: "inline",
      data: payload.Inline as Uint8Array<ArrayBuffer>,
    };
  }
  if ("Virtual" in payload) {
    const { location, offset, length } = payload.Virtual;
    return { kind: "virtual", location, offset, length };
  }
  const { id, offset, length } = payload.Ref;
  return { kind: "native", chunkId: id, offset, length };
}
