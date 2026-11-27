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

import { bypass, http, HttpResponse, passthrough } from "msw";
import { beforeEach, describe, expect, test } from "vitest";
import { getKvStoreCompletions } from "#src/datasource/kvstore_completions.js";
import { formatRefSpec, parseRefSpec } from "#src/kvstore/icechunk/url.js";
import { listKvStoreRecursively, readKvStore } from "#src/kvstore/index.js";
import {
  createBucket,
  fakeS3ServerFixture,
  writeObject,
} from "#tests/fixtures/fake_s3_server.js";
import { mswFixture } from "#tests/fixtures/msw";
import { sharedKvStoreContextFixture } from "#tests/fixtures/shared_kvstore_context.js";
import {
  sharedKvStoreContext,
  testAutoDetect,
} from "#tests/kvstore/test_util.js";

declare const TEST_DATA_SERVER: string;

const BASE_URL = `${TEST_DATA_SERVER}kvstore/icechunk/single_array.icechunk/`;

describe("completion", () => {
  test("empty prefix", async () => {
    const url = `${BASE_URL}|icechunk:`;
    const completions = await getKvStoreCompletions(
      await sharedKvStoreContext(),
      {
        url,
      },
    );
    expect(completions).toMatchInlineSnapshot(`
      {
        "completions": [
          {
            "description": "Ref specifier",
            "value": "@",
          },
          {
            "value": "zarr.json|",
          },
        ],
        "defaultCompletion": undefined,
        "offset": 72,
      }
    `);
  });

  test("@ prefix", async () => {
    const url = `${BASE_URL}|icechunk:@`;
    const completions = await getKvStoreCompletions(
      await sharedKvStoreContext(),
      {
        url,
      },
    );
    expect(completions).toMatchInlineSnapshot(`
      {
        "completions": [
          {
            "description": "Branch",
            "value": "branch.main/",
          },
          {
            "description": "Branch",
            "value": "branch.other_branch/",
          },
          {
            "description": "Tag",
            "value": "tag.tag1/",
          },
          {
            "description": "Tag",
            "value": "tag.tag2/",
          },
          {
            "description": "Tag",
            "value": "tag.tag3/",
          },
          {
            "description": "Snapshot",
            "value": "FWWFQGAW742XMX0F5MF0/",
          },
          {
            "description": "Snapshot",
            "value": "K85ER2WWGSGYGXW0GJC0/",
          },
        ],
        "offset": 73,
      }
    `);
  });

  test("@tag.t prefix", async () => {
    const url = `${BASE_URL}|icechunk:@tag.t`;
    const completions = await getKvStoreCompletions(
      await sharedKvStoreContext(),
      {
        url,
      },
    );
    expect(completions).toMatchInlineSnapshot(`
      {
        "completions": [
          {
            "description": "Tag",
            "value": "tag.tag1/",
          },
          {
            "description": "Tag",
            "value": "tag.tag2/",
          },
          {
            "description": "Tag",
            "value": "tag.tag3/",
          },
        ],
        "offset": 73,
      }
    `);
  });

  test("@branch.m prefix", async () => {
    const url = `${BASE_URL}|icechunk:@branch.m`;
    const completions = await getKvStoreCompletions(
      await sharedKvStoreContext(),
      {
        url,
      },
    );
    expect(completions).toMatchInlineSnapshot(`
      {
        "completions": [
          {
            "description": "Branch",
            "value": "branch.main/",
          },
        ],
        "offset": 73,
      }
    `);
  });
});

function listErrorMessages(error: unknown): string[] {
  const messages: string[] = [];
  for (let cause = error; cause instanceof Error; cause = cause.cause) {
    messages.push(cause.message);
  }
  return messages;
}

const msw = mswFixture();
const fakeS3Server = fakeS3ServerFixture({ msw });

beforeEach(async () => {
  (await msw()).use(http.all(`${TEST_DATA_SERVER}*`, () => passthrough()));
});

describe.for([
  {
    format: "0.1",
    directory: "",
    mainSnapshot: "K85ER2WWGSGYGXW0GJC0",
  },
  {
    format: "1",
    directory: "v1/",
    mainSnapshot: "AT8B56EZR2JFSFQ10M50",
  },
  {
    format: "2",
    directory: "v2/",
    mainSnapshot: "P56Z16N5AQF38YZ733Y0",
  },
])("Icechunk $format repository", ({ directory, mainSnapshot }) => {
  // In Node tests the main thread and the chunk worker share one global
  // scope, so their RPC ids collide once one context has made many calls.
  // A second context only loads its worker after the module-level context
  // has loaded its own, so these tests time out when filtered alone with -t.
  const formatKvStoreContext = sharedKvStoreContextFixture();
  const testDataUrl = `${TEST_DATA_SERVER}kvstore/icechunk/${directory}`;
  const singleArrayUrl = `${testDataUrl}single_array.icechunk/|icechunk:`;

  test.for(["single_array", "hierarchy"])(
    "reads every zarr.json and chunk of %s",
    async (name) => {
      const context = await formatKvStoreContext();
      const fileKvStore = context.kvStoreContext.getKvStore(
        `${testDataUrl}${name}/`,
      );
      const icechunkKvStore = context.kvStoreContext.getKvStore(
        `${testDataUrl}${name}.icechunk/|icechunk:`,
      );
      const fileKeys = await listKvStoreRecursively(
        fileKvStore.store,
        fileKvStore.path,
        { responseKeys: "suffix" },
      );
      const icechunkKeys = await listKvStoreRecursively(
        icechunkKvStore.store,
        icechunkKvStore.path,
        { responseKeys: "suffix" },
      );

      // The icechunk driver does not list chunk keys.
      expect(icechunkKeys.map(({ key }) => key)).toEqual(
        fileKeys
          .filter(({ key }) => key.endsWith("zarr.json"))
          .map(({ key }) => key),
      );

      for (const { key } of fileKeys) {
        const expectedResponse = (
          await readKvStore(fileKvStore.store, fileKvStore.path + key, {
            throwIfMissing: true,
          })
        ).response;
        const actualResponse = (
          await readKvStore(icechunkKvStore.store, icechunkKvStore.path + key, {
            throwIfMissing: true,
          })
        ).response;
        if (key.endsWith("zarr.json")) {
          expect
            .soft(await actualResponse.json(), key)
            .toEqual(await expectedResponse.json());
        } else {
          expect
            .soft(await actualResponse.arrayBuffer(), key)
            .toEqual(await expectedResponse.arrayBuffer());
        }
      }
    },
  );

  test.for([
    "@branch.main/",
    "@branch.other_branch/",
    "@tag.tag1/",
    "@tag.tag2/",
    `@${mainSnapshot}/`,
  ])("reads %s as the same data as no ref", async (ref) => {
    const context = await formatKvStoreContext();
    for (const key of ["zarr.json", "c/0/0"]) {
      const { response: expectedResponse } = await context.kvStoreContext.read(
        `${singleArrayUrl}${key}`,
        {
          throwIfMissing: true,
        },
      );
      const { response } = await context.kvStoreContext.read(
        `${singleArrayUrl}${ref}${key}`,
        { throwIfMissing: true },
      );
      expect
        .soft(await response.arrayBuffer(), key)
        .toEqual(await expectedResponse.arrayBuffer());
    }
  });

  test("rejects a deleted tag", async () => {
    const context = await formatKvStoreContext();
    const error = await context.kvStoreContext
      .read(`${singleArrayUrl}@tag.tag3/zarr.json`, { throwIfMissing: true })
      .then(
        () => undefined,
        (reason) => reason,
      );
    expect(listErrorMessages(error)).toContain("Tag is marked as deleted");
  });

  test("completes the names of branches and tags", async () => {
    const { completions } = (await getKvStoreCompletions(
      await formatKvStoreContext(),
      { url: `${singleArrayUrl}@` },
    ))!;
    expect(completions).toEqual(
      expect.arrayContaining([
        { value: "branch.main/", description: "Branch" },
        { value: "branch.other_branch/", description: "Branch" },
        { value: "tag.tag1/", description: "Tag" },
        { value: "tag.tag2/", description: "Tag" },
      ]),
    );
  });

  test("is detected as an Icechunk repository", async () => {
    expect(
      await testAutoDetect(`${testDataUrl}single_array.icechunk/`),
    ).toEqual([{ suffix: "icechunk:", description: "Icechunk repository" }]);
  });

  test("reads a virtual chunk", async () => {
    const content = Uint8Array.from({ length: 25 }, (_, i) => i);
    await createBucket(fakeS3Server, "mybucket");
    await writeObject(fakeS3Server, "mybucket", "myobject", content);
    const context = await formatKvStoreContext();
    const { response } = await context.kvStoreContext.read(
      `${testDataUrl}virtual_ref.icechunk/|icechunk:c/0/0`,
      { throwIfMissing: true },
    );
    expect(new Uint8Array(await response.arrayBuffer())).toEqual(
      content.slice(5, 25),
    );
  });
});

test("reads a virtual chunk of an Icechunk 2 repository whose location is dictionary-compressed", async () => {
  const content = Uint8Array.from({ length: 25 }, (_, i) => i);
  await createBucket(fakeS3Server, "mybucket");
  await writeObject(fakeS3Server, "mybucket", "object_2_3", content);
  const context = await sharedKvStoreContext();
  // Chunk (2, 3) refers to byte 2 * 5 + 3 of object_2_3.
  const { response } = await context.kvStoreContext.read(
    `${TEST_DATA_SERVER}kvstore/icechunk/v2/compressed_virtual_ref.icechunk/|icechunk:c/2/3`,
    { throwIfMissing: true },
  );
  expect(new Uint8Array(await response.arrayBuffer())).toEqual(
    Uint8Array.of(13),
  );
});

describe("bucket that denies reading missing objects", () => {
  const isolatedKvStoreContext = sharedKvStoreContextFixture();

  test("opens an Icechunk 1 repository", async () => {
    const repositoryUrl = `${TEST_DATA_SERVER}kvstore/icechunk/v1/single_array.icechunk/`;
    (await msw()).use(
      http.get(
        `${repositoryUrl}repo`,
        () => new HttpResponse(null, { status: 403 }),
      ),
    );
    const context = await isolatedKvStoreContext();
    const { response } = await context.kvStoreContext.read(
      `${repositoryUrl}|icechunk:@branch.main/zarr.json`,
      { throwIfMissing: true },
    );
    expect(await response.json()).toMatchObject({ node_type: "array" });
  });
});

describe("Icechunk 2 repository that changes while open", () => {
  const isolatedKvStoreContext = sharedKvStoreContextFixture();

  test("opens a branch created after the repository was first read", async () => {
    const testDataUrl = `${TEST_DATA_SERVER}kvstore/icechunk/v2/`;
    const repositoryUrl = `${testDataUrl}single_array.icechunk/`;
    let branchCreated = false;
    (await msw()).use(
      http.get(`${repositoryUrl}repo`, () =>
        branchCreated
          ? passthrough()
          : fetch(bypass(`${testDataUrl}single_array_repo_before_refs`)),
      ),
    );
    const context = await isolatedKvStoreContext();
    await context.kvStoreContext.read(
      `${repositoryUrl}|icechunk:@branch.main/zarr.json`,
      { throwIfMissing: true },
    );

    branchCreated = true;
    const { response } = await context.kvStoreContext.read(
      `${repositoryUrl}|icechunk:@branch.other_branch/zarr.json`,
      { throwIfMissing: true },
    );
    expect(await response.json()).toMatchObject({ node_type: "array" });
  });
});

describe("migrated Icechunk 2 repository with a damaged repo file", () => {
  const isolatedKvStoreContext = sharedKvStoreContextFixture();

  test("reports the error instead of reading the commit from before the migration", async () => {
    const repositoryUrl = `${TEST_DATA_SERVER}kvstore/icechunk/v2/migrated_single_array.icechunk/`;
    (await msw()).use(
      http.get(
        `${repositoryUrl}repo`,
        () => new HttpResponse(new Uint8Array(100), { status: 200 }),
      ),
    );
    const context = await isolatedKvStoreContext();
    const error = await context.kvStoreContext
      .read(`${repositoryUrl}|icechunk:zarr.json`, { throwIfMissing: true })
      .then(
        () => undefined,
        (reason) => reason,
      );
    expect(listErrorMessages(error)).toContainEqual(
      `Error reading icechunk repo info from ${repositoryUrl}repo`,
    );
  });
});

test("reads the newest commit of a repository migrated to Icechunk 2 that still has refs/", async () => {
  const context = await sharedKvStoreContext();
  // Only the repo info names the commit made after the migration.
  const { response } = await context.kvStoreContext.read(
    `${TEST_DATA_SERVER}kvstore/icechunk/v2/migrated_single_array.icechunk/|icechunk:zarr.json`,
    { throwIfMissing: true },
  );
  expect(await response.json()).toMatchObject({
    attributes: { after_migration: true },
  });
});

describe("ref spec parsing", () => {
  describe("valid round trip", () => {
    test.for([
      "ZZZZZZZZZZZZZZZZZZZZ",
      "branch.main",
      "branch.other",
      "tag.foo",
    ])("%s", (version) => {
      const parsed = parseRefSpec(version)!;
      expect(formatRefSpec(parsed!)).toEqual(version);
    });
  });

  describe("invalid", () => {
    test.for(["branch.", "b", "ZZZZZZZZZZZ"])("%s", (version) => {
      expect(() => parseRefSpec(version)).toThrow();
    });
  });
});
