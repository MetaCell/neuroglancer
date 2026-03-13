import { describe, expect, it } from "vitest";

import {
  appendNodeToSpatialChunk,
  rebuildSpatialChunkConnections,
  rebuildTargetSpatialChunkConnections,
  removeNodeFromSpatialChunk,
  updateNodePositionInSpatialChunk,
  type SpatiallyIndexedEditableChunkData,
  type SpatiallyIndexedEditableChunkEntry,
} from "#src/skeleton/spatial_edit.js";

function makeChunkEntry(options: {
  chunkId: string;
  chunkKey: string;
  sourceId: string;
  nodes: Array<{
    nodeId: number;
    position: [number, number, number];
    segmentId: number;
  }>;
}): SpatiallyIndexedEditableChunkEntry {
  const nodeMap = new Map<number, number>();
  const positions = new Float32Array(options.nodes.length * 3);
  const segmentIds = new Uint32Array(options.nodes.length);
  options.nodes.forEach((node, vertexIndex) => {
    nodeMap.set(node.nodeId, vertexIndex);
    positions.set(node.position, vertexIndex * 3);
    segmentIds[vertexIndex] = node.segmentId;
  });
  return {
    chunkId: options.chunkId,
    chunkKey: options.chunkKey,
    sourceId: options.sourceId,
    data: {
      nodeMap,
      positions,
      segmentIds,
      indices: new Uint32Array(),
      missingConnections: [],
    },
  };
}

describe("skeleton/spatial_edit", () => {
  it("adds a node to a loaded chunk and links it locally when the parent is resident", () => {
    const chunk = makeChunkEntry({
      chunkId: "s:0,0,0:0",
      chunkKey: "0,0,0:0",
      sourceId: "s",
      nodes: [
        {
          nodeId: 1,
          position: [1, 2, 3],
          segmentId: 10,
        },
      ],
    });

    chunk.data = appendNodeToSpatialChunk(chunk.data, {
      nodeId: 2,
      segmentId: 10,
      position: [4, 5, 6],
    }).data;

    const rebuilt = rebuildSpatialChunkConnections([chunk], new Map([[2, 1]]));

    expect(Array.from(chunk.data.nodeMap.entries())).toEqual([
      [1, 0],
      [2, 1],
    ]);
    expect(chunk.data.positions).toEqual(Float32Array.of(1, 2, 3, 4, 5, 6));
    expect(rebuilt).toEqual([
      {
        chunkId: "s:0,0,0:0",
        indices: Uint32Array.of(1, 0),
        missingConnections: [],
      },
    ]);
  });

  it("updates a resident node position in-place for same-chunk moves", () => {
    const chunk = makeChunkEntry({
      chunkId: "s:0,0,0:0",
      chunkKey: "0,0,0:0",
      sourceId: "s",
      nodes: [
        {
          nodeId: 11,
          position: [1, 2, 3],
          segmentId: 10,
        },
      ],
    });

    const updated = updateNodePositionInSpatialChunk(chunk.data, 11, [7, 8, 9]);

    expect(updated?.data.positions).toEqual(Float32Array.of(7, 8, 9));
    expect(updated?.vertexIndex).toBe(0);
  });

  it("preserves child edges by converting them to missing connections on cross-chunk moves", () => {
    const oldChunk = makeChunkEntry({
      chunkId: "s:0,0,0:0",
      chunkKey: "0,0,0:0",
      sourceId: "s",
      nodes: [
        {
          nodeId: 1,
          position: [0, 0, 0],
          segmentId: 10,
        },
        {
          nodeId: 2,
          position: [1, 1, 1],
          segmentId: 10,
        },
        {
          nodeId: 3,
          position: [2, 2, 2],
          segmentId: 10,
        },
      ],
    });
    const newChunk = makeChunkEntry({
      chunkId: "s:1,0,0:0",
      chunkKey: "1,0,0:0",
      sourceId: "s",
      nodes: [],
    });

    const parentByNodeId = new Map<number, number | undefined>([
      [2, 1],
      [3, 2],
    ]);

    oldChunk.data = rebuildChunk(oldChunk, parentByNodeId);
    const removed = removeNodeFromSpatialChunk(oldChunk.data, 2);
    expect(removed).toBeDefined();
    oldChunk.data = removed!.data;
    newChunk.data = appendNodeToSpatialChunk(newChunk.data, {
      nodeId: 2,
      segmentId: 10,
      position: [10, 10, 10],
    }).data;

    const rebuilt = rebuildSpatialChunkConnections(
      [oldChunk, newChunk],
      parentByNodeId,
    );

    expect(rebuilt).toEqual([
      {
        chunkId: "s:0,0,0:0",
        indices: Uint32Array.of(),
        missingConnections: [
          {
            nodeId: 3,
            parentId: 2,
            vertexIndex: 1,
            skeletonId: 10,
          },
        ],
      },
      {
        chunkId: "s:1,0,0:0",
        indices: Uint32Array.of(),
        missingConnections: [
          {
            nodeId: 2,
            parentId: 1,
            vertexIndex: 0,
            skeletonId: 10,
          },
        ],
      },
    ]);
  });

  it("rebuilds only the target chunks when locators are resolved externally", () => {
    const targetChunk = makeChunkEntry({
      chunkId: "s:1,0,0:0",
      chunkKey: "1,0,0:0",
      sourceId: "s",
      nodes: [
        {
          nodeId: 2,
          position: [10, 10, 10],
          segmentId: 10,
        },
        {
          nodeId: 3,
          position: [11, 11, 11],
          segmentId: 10,
        },
      ],
    });

    const rebuilt = rebuildTargetSpatialChunkConnections(
      [targetChunk],
      new Map<number, number | undefined>([
        [2, 1],
        [3, 2],
      ]),
      (sourceId, nodeId) => {
        if (sourceId !== "s") return undefined;
        if (nodeId === 1) {
          return {
            chunkKey: "0,0,0:0",
            vertexIndex: 0,
          };
        }
        if (nodeId === 2) {
          return {
            chunkKey: "1,0,0:0",
            vertexIndex: 0,
          };
        }
        return undefined;
      },
    );

    expect(rebuilt).toEqual([
      {
        chunkId: "s:1,0,0:0",
        indices: Uint32Array.of(1, 0),
        missingConnections: [
          {
            nodeId: 2,
            parentId: 1,
            vertexIndex: 0,
            skeletonId: 10,
          },
        ],
      },
    ]);
  });

  it("returns undefined when a direct same-chunk move cannot find the resident node", () => {
    const chunk = makeChunkEntry({
      chunkId: "s:0,0,0:0",
      chunkKey: "0,0,0:0",
      sourceId: "s",
      nodes: [],
    });

    expect(updateNodePositionInSpatialChunk(chunk.data, 99, [1, 2, 3])).toBe(
      undefined,
    );
  });
});

function rebuildChunk(
  chunk: SpatiallyIndexedEditableChunkEntry,
  parentByNodeId: ReadonlyMap<number, number | undefined>,
): SpatiallyIndexedEditableChunkData {
  const [rebuilt] = rebuildSpatialChunkConnections([chunk], parentByNodeId);
  return {
    ...chunk.data,
    indices: rebuilt.indices,
    missingConnections: rebuilt.missingConnections,
  };
}
