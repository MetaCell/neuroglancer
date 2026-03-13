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
  it("merges skeletons using from/to treenode ids", async () => {
    const client = new CatmaidClient("https://example.invalid", 1);
    const fetchMock = vi
      .fn()
      .mockResolvedValue({
        result_skeleton_id: 17,
        deleted_skeleton_id: 21,
        stable_annotation_swap: false,
      });
    (client as any).fetch = fetchMock;

    await expect(client.mergeSkeletons(101, 202)).resolves.toEqual({
      resultSkeletonId: 17,
      deletedSkeletonId: 21,
      stableAnnotationSwap: false,
    });

    expect(fetchMock).toHaveBeenCalledTimes(1);
    expect(fetchMock.mock.calls[0]?.[0]).toBe("skeleton/join");
    expect((fetchMock.mock.calls[0]?.[1] as { body: URLSearchParams }).body.get("from_id")).toBe(
      "101",
    );
    expect((fetchMock.mock.calls[0]?.[1] as { body: URLSearchParams }).body.get("to_id")).toBe(
      "202",
    );
  });

  it("returns treenode and skeleton ids from addNode", async () => {
    const client = new CatmaidClient("https://example.invalid", 1);
    const fetchMock = vi
      .fn()
      .mockResolvedValue({ treenode_id: 88, skeleton_id: 13 });
    (client as any).fetch = fetchMock;

    await expect(client.addNode(13, 1, 2, 3, 7)).resolves.toEqual({
      treenodeId: 88,
      skeletonId: 13,
    });
  });
});
