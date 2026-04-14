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

import { describe, expect, it, vi } from "vitest";

import { CatmaidClient } from "#src/datasource/catmaid/api.js";

describe("CatmaidClient skeleton editing methods", () => {
  it("does not cache transient metadata discovery failures as null", async () => {
    const client = new CatmaidClient("https://example.invalid", 1);
    const warnSpy = vi.spyOn(console, "warn").mockImplementation(() => {});
    (client as any).listStacks = vi
      .fn()
      .mockRejectedValueOnce(new Error("temporary stack lookup failure"))
      .mockResolvedValueOnce([{ id: 7, title: "stack" }]);
    (client as any).getStackInfo = vi.fn().mockResolvedValue({
      dimension: { x: 10, y: 20, z: 30 },
      resolution: { x: 2, y: 3, z: 4 },
      translation: { x: 5, y: 6, z: 7 },
    });

    await expect(client.getDimensions()).resolves.toBeNull();
    await expect(client.getDimensions()).resolves.toEqual({
      min: { x: 5, y: 6, z: 7 },
      max: { x: 25, y: 66, z: 127 },
    });

    expect((client as any).listStacks).toHaveBeenCalledTimes(2);
    expect((client as any).getStackInfo).toHaveBeenCalledTimes(1);
    warnSpy.mockRestore();
  });

  it("parses live compact-detail history rows and label maps", async () => {
    const client = new CatmaidClient("https://example.invalid", 1);
    const fetchMock = vi.fn().mockResolvedValue([
      [
        [
          22107946,
          null,
          2,
          23697030.0,
          15055839.0,
          16651262.0,
          2000.0,
          5,
          "2026-03-29T10:15:00Z",
          "2026-03-29T10:15:00Z",
        ],
        [
          22107946,
          null,
          2,
          23697030.0,
          15055839.0,
          16651262.0,
          2000.0,
          5,
          "2026-03-28T08:00:00Z",
          "2026-03-29T10:15:00Z",
        ],
        [
          22107955,
          22107954,
          2,
          23705874.0,
          15093672.0,
          16682375.0,
          2000.0,
          5,
          "2026-03-29T10:16:00Z",
          "2026-03-29T10:15:00Z",
        ],
        [
          22107959,
          22107958,
          2,
          23704520.0,
          15085237.0,
          16708998.0,
          2000.0,
          5,
          "2026-03-29T10:17:00Z",
          "2026-03-29T10:16:00Z",
        ],
      ],
      [],
      {
        "afonso reviewed it": [22107946],
        "test 123 4": [22107955],
        ends: [22107959],
      },
      [],
      [],
    ]);
    (client as any).fetch = fetchMock;

    await expect(client.getSkeleton(2)).resolves.toEqual([
      {
        id: 22107946,
        parent_id: null,
        x: 23697030,
        y: 15055839,
        z: 16651262,
        skeleton_id: 2,
        radius: 2000,
        confidence: 100,
        labels: ["afonso reviewed it"],
        revisionToken: "2026-03-29T10:15:00Z",
      },
      {
        id: 22107955,
        parent_id: 22107954,
        x: 23705874,
        y: 15093672,
        z: 16682375,
        skeleton_id: 2,
        radius: 2000,
        confidence: 100,
        labels: ["test 123 4"],
        revisionToken: "2026-03-29T10:16:00Z",
      },
      {
        id: 22107959,
        parent_id: 22107958,
        x: 23704520,
        y: 15085237,
        z: 16708998,
        skeleton_id: 2,
        radius: 2000,
        confidence: 100,
        labels: ["ends"],
        revisionToken: "2026-03-29T10:17:00Z",
      },
    ]);
  });

  it("ignores zero-width history rows when compact-detail includes ordering", async () => {
    const client = new CatmaidClient("https://example.invalid", 1);
    const fetchMock = vi.fn().mockResolvedValue([
      [
        [
          11422971,
          11422970,
          2,
          24313028.0,
          14983333.0,
          6761820.5,
          2000.0,
          5,
          "2026-04-14 08:56:49.985049+00:00",
          "2026-04-14 08:56:49.985049+00:00",
          2,
        ],
        [
          11422972,
          11422971,
          2,
          24318870.0,
          14984255.0,
          6765134.0,
          2000.0,
          5,
          "2026-04-14 08:56:49.985049+00:00",
          "2026-04-14 08:56:49.985049+00:00",
          2,
        ],
      ],
      [],
      {},
      [],
      [],
    ]);
    (client as any).fetch = fetchMock;

    await expect(client.getSkeleton(1140285)).resolves.toEqual([]);
  });

  it("merges skeletons using from/to treenode ids", async () => {
    const client = new CatmaidClient("https://example.invalid", 1);
    const fetchMock = vi.fn().mockResolvedValue({
      result_skeleton_id: 17,
      deleted_skeleton_id: 21,
      stable_annotation_swap: false,
    });
    (client as any).fetch = fetchMock;

    await expect(
      client.mergeSkeletons(101, 202, {
        nodes: [
          { nodeId: 101, revisionToken: "2026-03-29T11:50:00Z" },
          { nodeId: 202, revisionToken: "2026-03-29T11:51:00Z" },
        ],
      }),
    ).resolves.toEqual({
      resultSkeletonId: 17,
      deletedSkeletonId: 21,
      stableAnnotationSwap: false,
    });

    expect(fetchMock).toHaveBeenCalledTimes(1);
    expect(fetchMock.mock.calls[0]?.[0]).toBe("skeleton/join");
    expect(
      (fetchMock.mock.calls[0]?.[1] as { body: URLSearchParams }).body.get(
        "from_id",
      ),
    ).toBe("101");
    expect(
      (fetchMock.mock.calls[0]?.[1] as { body: URLSearchParams }).body.get(
        "to_id",
      ),
    ).toBe("202");
    expect(
      (fetchMock.mock.calls[0]?.[1] as { body: URLSearchParams }).body.get(
        "state",
      ),
    ).toBe(
      JSON.stringify([
        [101, "2026-03-29T11:50:00Z"],
        [202, "2026-03-29T11:51:00Z"],
      ]),
    );
  });

  it("rejects merge state when the provided node ids do not match the request", async () => {
    const client = new CatmaidClient("https://example.invalid", 1);
    const fetchMock = vi.fn();
    (client as any).fetch = fetchMock;

    await expect(
      client.mergeSkeletons(101, 202, {
        nodes: [{ nodeId: 101, revisionToken: "2026-03-29T11:50:00Z" }],
      }),
    ).rejects.toThrow(
      "CATMAID merge-skeleton node state does not match the requested node ids.",
    );

    expect(fetchMock).not.toHaveBeenCalled();
  });

  it("returns ids and revisions from addNode and sends CATMAID parent state", async () => {
    const client = new CatmaidClient("https://example.invalid", 1);
    const fetchMock = vi
      .fn()
      .mockResolvedValue({
        treenode_id: 88,
        skeleton_id: 13,
        edition_time: "2026-03-29T12:00:00Z",
        parent_edition_time: "2026-03-29T12:00:01Z",
      });
    (client as any).fetch = fetchMock;

    await expect(
      client.addNode(13, 1, 2, 3, 7, {
        node: {
          nodeId: 7,
          revisionToken: "2026-03-29T11:59:00Z",
        },
      }),
    ).resolves.toEqual({
      treenodeId: 88,
      skeletonId: 13,
      revisionToken: "2026-03-29T12:00:00Z",
      parentRevisionToken: "2026-03-29T12:00:01Z",
    });

    expect(
      (fetchMock.mock.calls[0]?.[1] as { body: URLSearchParams }).body.get(
        "state",
      ),
    ).toBe(JSON.stringify({ parent: [7, "2026-03-29T11:59:00Z"] }));
  });

  it("sends CATMAID root parent state when creating a root node", async () => {
    const client = new CatmaidClient("https://example.invalid", 1);
    const fetchMock = vi
      .fn()
      .mockResolvedValue({
        treenode_id: 88,
        skeleton_id: 13,
        edition_time: "2026-03-29T12:00:00Z",
      });
    (client as any).fetch = fetchMock;

    await expect(client.addNode(13, 1, 2, 3)).resolves.toEqual({
      treenodeId: 88,
      skeletonId: 13,
      revisionToken: "2026-03-29T12:00:00Z",
      parentRevisionToken: undefined,
    });

    expect(
      (fetchMock.mock.calls[0]?.[1] as { body: URLSearchParams }).body.get(
        "state",
      ),
    ).toBe(JSON.stringify({ parent: [-1, ""] }));
  });

  it("inserts nodes using CATMAID local parent-and-child state", async () => {
    const client = new CatmaidClient("https://example.invalid", 1);
    const fetchMock = vi.fn().mockResolvedValue({
      treenode_id: 89,
      skeleton_id: 13,
      edition_time: "2026-03-29T12:01:00Z",
      parent_edition_time: "2026-03-29T12:01:01Z",
      child_edition_times: [
        [11, "2026-03-29T12:01:02Z"],
        [12, "2026-03-29T12:01:03Z"],
      ],
    });
    (client as any).fetch = fetchMock;

    await expect(
      client.insertNode(13, 1, 2, 3, 7, [11, 12], {
        node: {
          nodeId: 7,
          revisionToken: "2026-03-29T12:00:30Z",
        },
        children: [
          { nodeId: 11, revisionToken: "2026-03-29T12:00:31Z" },
          { nodeId: 12, revisionToken: "2026-03-29T12:00:32Z" },
        ],
      }),
    ).resolves.toEqual({
      treenodeId: 89,
      skeletonId: 13,
      revisionToken: "2026-03-29T12:01:00Z",
      parentRevisionToken: "2026-03-29T12:01:01Z",
      childRevisionUpdates: [
        { nodeId: 11, revisionToken: "2026-03-29T12:01:02Z" },
        { nodeId: 12, revisionToken: "2026-03-29T12:01:03Z" },
      ],
    });

    expect(fetchMock).toHaveBeenCalledTimes(1);
    expect(fetchMock.mock.calls[0]?.[0]).toBe("treenode/insert");
    expect(
      (fetchMock.mock.calls[0]?.[1] as { body: URLSearchParams }).body.get(
        "parent_id",
      ),
    ).toBe("7");
    expect(
      (fetchMock.mock.calls[0]?.[1] as { body: URLSearchParams }).body.get(
        "child_id",
      ),
    ).toBe("11");
    expect(
      (fetchMock.mock.calls[0]?.[1] as { body: URLSearchParams }).body.get(
        "takeover_child_ids[0]",
      ),
    ).toBe("12");
    expect(
      (fetchMock.mock.calls[0]?.[1] as { body: URLSearchParams }).body.get(
        "state",
      ),
    ).toBe(
      JSON.stringify({
        edition_time: "2026-03-29T12:00:30Z",
        children: [
          [11, "2026-03-29T12:00:31Z"],
          [12, "2026-03-29T12:00:32Z"],
        ],
        links: [],
      }),
    );
  });

  it("reroots skeletons using treenode ids", async () => {
    const client = new CatmaidClient("https://example.invalid", 1);
    const fetchMock = vi
      .fn()
      .mockResolvedValue({ newroot: 202, skeleton_id: 17 });
    (client as any).fetch = fetchMock;

    await expect(
      client.rerootSkeleton(202, {
        node: {
          nodeId: 202,
          parentNodeId: 201,
          revisionToken: "2026-03-29T12:05:00Z",
        },
        parent: {
          nodeId: 201,
          revisionToken: "2026-03-29T12:04:00Z",
        },
        children: [
          { nodeId: 203, revisionToken: "2026-03-29T12:06:00Z" },
          { nodeId: 204, revisionToken: "2026-03-29T12:07:00Z" },
        ],
      }),
    ).resolves.toBeUndefined();

    expect(fetchMock).toHaveBeenCalledTimes(1);
    expect(fetchMock.mock.calls[0]?.[0]).toBe("skeleton/reroot");
    expect(
      (fetchMock.mock.calls[0]?.[1] as { body: URLSearchParams }).body.get(
        "treenode_id",
      ),
    ).toBe("202");
    expect(
      (fetchMock.mock.calls[0]?.[1] as { body: URLSearchParams }).body.get(
        "state",
      ),
    ).toBe(
      JSON.stringify({
        edition_time: "2026-03-29T12:05:00Z",
        parent: [201, "2026-03-29T12:04:00Z"],
        children: [
          [203, "2026-03-29T12:06:00Z"],
          [204, "2026-03-29T12:07:00Z"],
        ],
        links: [],
      }),
    );
  });

  it("rejects reroot state when the parent neighborhood is incomplete", async () => {
    const client = new CatmaidClient("https://example.invalid", 1);
    const fetchMock = vi.fn();
    (client as any).fetch = fetchMock;

    await expect(
      client.rerootSkeleton(202, {
        node: {
          nodeId: 202,
          parentNodeId: 201,
          revisionToken: "2026-03-29T12:05:00Z",
        },
        children: [{ nodeId: 203, revisionToken: "2026-03-29T12:06:00Z" }],
      }),
    ).rejects.toThrow(
      "CATMAID reroot-skeleton parent state does not match the cached skeleton neighborhood.",
    );

    expect(fetchMock).not.toHaveBeenCalled();
  });

  it("splits skeletons using neighborhood state", async () => {
    const client = new CatmaidClient("https://example.invalid", 1);
    const fetchMock = vi.fn().mockResolvedValue({
      existing_skeleton_id: 17,
      new_skeleton_id: 21,
    });
    (client as any).fetch = fetchMock;

    await expect(
      client.splitSkeleton(202, {
        node: {
          nodeId: 202,
          parentNodeId: 201,
          revisionToken: "2026-03-29T12:05:00Z",
        },
        parent: {
          nodeId: 201,
          revisionToken: "2026-03-29T12:04:00Z",
        },
        children: [
          { nodeId: 203, revisionToken: "2026-03-29T12:06:00Z" },
        ],
      }),
    ).resolves.toEqual({
      existingSkeletonId: 17,
      newSkeletonId: 21,
    });

    expect(fetchMock).toHaveBeenCalledTimes(1);
    expect(fetchMock.mock.calls[0]?.[0]).toBe("skeleton/split");
    expect(
      (fetchMock.mock.calls[0]?.[1] as { body: URLSearchParams }).body.get(
        "treenode_id",
      ),
    ).toBe("202");
    expect(
      (fetchMock.mock.calls[0]?.[1] as { body: URLSearchParams }).body.get(
        "state",
      ),
    ).toBe(
      JSON.stringify({
        edition_time: "2026-03-29T12:05:00Z",
        parent: [201, "2026-03-29T12:04:00Z"],
        children: [[203, "2026-03-29T12:06:00Z"]],
        links: [],
      }),
    );
  });

  it("fetches node revision updates from treenode compact-detail", async () => {
    const client = new CatmaidClient("https://example.invalid", 1);
    const fetchMock = vi.fn().mockResolvedValue([
      [11, 7, 1, 2, 3, 5, 2000, 13, 1711711711.25, 9],
      [12, 11, 4, 5, 6, 5, 2000, 13, 1711711712.5, 9],
    ]);
    (client as any).fetch = fetchMock;

    await expect(client.getNodeRevisionUpdates([12, 11, 12])).resolves.toEqual([
      { nodeId: 11, revisionToken: "2024-03-29T11:28:31.250Z" },
      { nodeId: 12, revisionToken: "2024-03-29T11:28:32.500Z" },
    ]);

    expect(fetchMock).toHaveBeenCalledTimes(1);
    expect(fetchMock.mock.calls[0]?.[0]).toBe("treenodes/compact-detail");
    expect(
      (fetchMock.mock.calls[0]?.[1] as { body: URLSearchParams }).body.toString(),
    ).toBe("treenode_ids%5B0%5D=11&treenode_ids%5B1%5D=12");
  });

  it("rejects partial treenode compact-detail revision responses", async () => {
    const client = new CatmaidClient("https://example.invalid", 1);
    const fetchMock = vi.fn().mockResolvedValue([
      [11, 7, 1, 2, 3, 5, 2000, 13, 1711711711.25, 9],
    ]);
    (client as any).fetch = fetchMock;

    await expect(client.getNodeRevisionUpdates([11, 12])).rejects.toThrow(
      "CATMAID treenodes/compact-detail did not return revision metadata for node(s) 12.",
    );
  });

  it("moves nodes using node revision state and returns the updated revision", async () => {
    const client = new CatmaidClient("https://example.invalid", 1);
    const fetchMock = vi.fn().mockResolvedValue({
      updated: 1,
      old_treenodes: [[42, "2026-03-29T12:10:00Z", 1, 2, 3]],
      old_connectors: [],
    });
    (client as any).fetch = fetchMock;

    await expect(
      client.moveNode(42, 10, 11, 12, {
        node: {
          nodeId: 42,
          revisionToken: "2026-03-29T12:00:00Z",
        },
      }),
    ).resolves.toEqual({
      revisionToken: "2026-03-29T12:10:00Z",
    });

    expect(
      (fetchMock.mock.calls[0]?.[1] as { body: URLSearchParams }).body.get(
        "state",
      ),
    ).toBe(JSON.stringify([[42, "2026-03-29T12:00:00Z"]]));
  });

  it("deletes nodes using neighborhood state and returns child revisions", async () => {
    const client = new CatmaidClient("https://example.invalid", 1);
    const fetchMock = vi.fn().mockResolvedValue({
      success: "Removed treenode successfully.",
      children: [
        [12, "2026-03-29T12:20:00Z"],
        [13, "2026-03-29T12:20:01Z"],
      ],
    });
    (client as any).fetch = fetchMock;

    await expect(
      client.deleteNode(11, {
        childNodeIds: [12, 13],
        editContext: {
          node: {
            nodeId: 11,
            parentNodeId: 7,
            revisionToken: "2026-03-29T12:15:00Z",
          },
          parent: {
            nodeId: 7,
            revisionToken: "2026-03-29T12:14:00Z",
          },
          children: [
            { nodeId: 12, revisionToken: "2026-03-29T12:13:00Z" },
            { nodeId: 13, revisionToken: "2026-03-29T12:13:01Z" },
          ],
        },
      }),
    ).resolves.toEqual({
      childRevisionUpdates: [
        { nodeId: 12, revisionToken: "2026-03-29T12:20:00Z" },
        { nodeId: 13, revisionToken: "2026-03-29T12:20:01Z" },
      ],
    });

    expect(
      (fetchMock.mock.calls[0]?.[1] as { body: URLSearchParams }).body.get(
        "state",
      ),
    ).toBe(
      JSON.stringify({
        edition_time: "2026-03-29T12:15:00Z",
        parent: [7, "2026-03-29T12:14:00Z"],
        children: [
          [12, "2026-03-29T12:13:00Z"],
          [13, "2026-03-29T12:13:01Z"],
        ],
        links: [],
      }),
    );
  });

  it("updates descriptions without CATMAID node state", async () => {
    const client = new CatmaidClient("https://example.invalid", 1);
    const fetchMock = vi
      .fn()
      .mockResolvedValue({ edition_time: "2026-03-29T13:00:00Z" });
    (client as any).fetch = fetchMock;

    await expect(
      client.updateDescription(11, "updated description", { trueEnd: false }),
    ).resolves.toEqual({
      revisionToken: "2026-03-29T13:00:00Z",
    });

    expect(
      (fetchMock.mock.calls[0]?.[1] as { body: URLSearchParams }).body.get(
        "state",
      ),
    ).toBeNull();
    expect(
      (fetchMock.mock.calls[0]?.[1] as { body: URLSearchParams }).body.get(
        "tags",
      ),
    ).toBe("updated description");
    expect(
      (fetchMock.mock.calls[0]?.[1] as { body: URLSearchParams }).body.get(
        "delete_existing",
      ),
    ).toBe("true");
  });

  it("toggles true-end labels without CATMAID node state", async () => {
    const client = new CatmaidClient("https://example.invalid", 1);
    const fetchMock = vi
      .fn()
      .mockResolvedValueOnce({ edition_time: "2026-03-29T13:10:00Z" })
      .mockResolvedValueOnce({ edition_time: "2026-03-29T13:11:00Z" });
    (client as any).fetch = fetchMock;

    await expect(client.setTrueEnd(11)).resolves.toEqual({
      revisionToken: "2026-03-29T13:10:00Z",
    });
    await expect(client.removeTrueEnd(11)).resolves.toEqual({
      revisionToken: "2026-03-29T13:11:00Z",
    });

    expect(
      (fetchMock.mock.calls[0]?.[1] as { body: URLSearchParams }).body.get(
        "state",
      ),
    ).toBeNull();
    expect(
      (fetchMock.mock.calls[1]?.[1] as { body: URLSearchParams }).body.get(
        "state",
      ),
    ).toBeNull();
    expect(
      (fetchMock.mock.calls[0]?.[1] as { body: URLSearchParams }).body.get(
        "tags",
      ),
    ).toBe("ends");
    expect(
      (fetchMock.mock.calls[0]?.[1] as { body: URLSearchParams }).body.get(
        "delete_existing",
      ),
    ).toBe("false");
    expect(
      (fetchMock.mock.calls[1]?.[1] as { body: URLSearchParams }).body.get(
        "tag",
      ),
    ).toBe("ends");
  });

  it("maps CATMAID state validation failures to a refresh-specific error", async () => {
    const client = new CatmaidClient("https://example.invalid", 1);
    const fetchMock = vi
      .spyOn(globalThis, "fetch")
      .mockResolvedValue(
        new Response(
          JSON.stringify({
            type: "StateMatchingError",
            error:
              "The provided state differs from the database state: {'edition_time': '2026-03-29T13:12:00Z'}",
          }),
          {
            status: 400,
            headers: {
              "Content-Type": "application/json",
            },
          },
        ),
      );

    await expect(
      client.moveNode(11, 1, 2, 3, {
        node: {
          nodeId: 11,
          revisionToken: "2026-03-29T13:11:00Z",
        },
      }),
    ).rejects.toThrow(
      "CATMAID rejected the edit because the inspected skeleton is out of date. Refresh the skeleton and try again.",
    );

    fetchMock.mockRestore();
  });

  it("preserves generic CATMAID 400 value errors", async () => {
    const client = new CatmaidClient("https://example.invalid", 1);
    const fetchMock = vi
      .spyOn(globalThis, "fetch")
      .mockResolvedValue(
        new Response(
          JSON.stringify({
            type: "ValueError",
            error: "No valid state provided, missing edition time",
          }),
          {
            status: 400,
            headers: {
              "Content-Type": "application/json",
            },
          },
        ),
      );

    await expect(
      client.moveNode(11, 1, 2, 3, {
        node: {
          nodeId: 11,
          revisionToken: "2026-03-29T13:11:00Z",
        },
      }),
    ).rejects.toMatchObject({
      name: "HttpError",
      status: 400,
    });

    fetchMock.mockRestore();
  });
});
