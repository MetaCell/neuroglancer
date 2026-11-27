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

import { SimpleAsyncCache } from "#src/chunk_manager/generic_file_source.js";
import type { SharedKvStoreContextCounterpart } from "#src/kvstore/backend.js";
import { decodeRepoInfo } from "#src/kvstore/icechunk/decode_flatbuffers.js";
import type { ManifestId } from "#src/kvstore/icechunk/decode_utils.js";
import {
  decodeManifest,
  getManifestUrl,
} from "#src/kvstore/icechunk/manifest.js";
import { decodeRef, isBranchRef } from "#src/kvstore/icechunk/ref.js";
import type { RepoInfo } from "#src/kvstore/icechunk/repo_info.js";
import type { SnapshotId } from "#src/kvstore/icechunk/snapshot.js";
import {
  decodeSnapshot,
  getSnapshotUrl,
} from "#src/kvstore/icechunk/snapshot.js";
import type { RefSpec } from "#src/kvstore/icechunk/url.js";
import { formatRefSpec } from "#src/kvstore/icechunk/url.js";
import { pipelineUrlJoin } from "#src/kvstore/url.js";
import {
  ProgressSpan,
  type ProgressOptions,
} from "#src/util/progress_listener.js";

type DecodeMetadata<T> = (
  response: Response,
  signal: AbortSignal | undefined,
) => Promise<{ data: T; size: number }>;

async function readMetadata<T>(
  sharedKvStoreContext: SharedKvStoreContextCounterpart,
  description: string,
  decode: DecodeMetadata<T>,
  url: string,
  options: Partial<ProgressOptions>,
): Promise<{ data: T; size: number } | undefined> {
  const readResponse = await sharedKvStoreContext.kvStoreContext.read(url, {
    ...options,
    throwIfMissing: false,
  });
  if (readResponse === undefined) return undefined;
  try {
    return await decode(readResponse.response, options.signal);
  } catch (e) {
    throw new Error(`Error reading icechunk ${description} from ${url}`, {
      cause: e,
    });
  }
}

function makeCache<T>(
  sharedKvStoreContext: SharedKvStoreContextCounterpart,
  description: string,
  decode: DecodeMetadata<T>,
  handleMissing: (url: string) => { data: T; size: number },
) {
  const cache = new SimpleAsyncCache<string, T>(
    sharedKvStoreContext.chunkManager.addRef(),
    {
      get: async (url: string, progressOptions: ProgressOptions) =>
        (await readMetadata(
          sharedKvStoreContext,
          description,
          decode,
          url,
          progressOptions,
        )) ?? handleMissing(url),
    },
  );
  cache.registerDisposer(sharedKvStoreContext.addRef());
  return cache;
}

function makeMetadataCache<T>(
  sharedKvStoreContext: SharedKvStoreContextCounterpart,
  description: string,
  decode: DecodeMetadata<T>,
) {
  return makeCache(sharedKvStoreContext, description, decode, (url) => {
    throw new Error(`Missing icechunk ${description} at ${url}`);
  });
}

function makeOptionalMetadataCache<T>(
  sharedKvStoreContext: SharedKvStoreContextCounterpart,
  description: string,
  decode: DecodeMetadata<T>,
) {
  return makeCache<T | undefined>(
    sharedKvStoreContext,
    description,
    decode,
    () => ({ data: undefined, size: 0 }),
  );
}

const decodeRepoInfoResponse: DecodeMetadata<RepoInfo> = async (
  response,
  signal,
) => {
  const repoInfo = await decodeRepoInfo(await response.arrayBuffer(), signal);
  return { data: repoInfo, size: repoInfo.estimatedSize };
};

function getRepoInfoUrl(baseUrl: string) {
  return pipelineUrlJoin(baseUrl, "repo");
}

// Undefined when the repository predates Icechunk 2.
export function getRepoInfo(
  sharedKvStoreContext: SharedKvStoreContextCounterpart,
  baseUrl: string,
  options: Partial<ProgressOptions>,
): Promise<RepoInfo | undefined> {
  const cache = sharedKvStoreContext.chunkManager.memoize.get(
    "icechunk:repo",
    () =>
      makeOptionalMetadataCache(
        sharedKvStoreContext,
        "repo info",
        decodeRepoInfoResponse,
      ),
  );
  return cache.get(getRepoInfoUrl(baseUrl), options);
}

// Bypasses the cache, to find refs created after `getRepoInfo` cached its copy.
async function readLatestRepoInfo(
  sharedKvStoreContext: SharedKvStoreContextCounterpart,
  baseUrl: string,
  options: Partial<ProgressOptions>,
): Promise<RepoInfo | undefined> {
  return (
    await readMetadata(
      sharedKvStoreContext,
      "repo info",
      decodeRepoInfoResponse,
      getRepoInfoUrl(baseUrl),
      options,
    )
  )?.data;
}

export function getSnapshot(
  sharedKvStoreContext: SharedKvStoreContextCounterpart,
  baseUrl: string,
  id: SnapshotId,
  options: Partial<ProgressOptions>,
) {
  const cache = sharedKvStoreContext.chunkManager.memoize.get(
    "icechunk:snapshot",
    () =>
      makeMetadataCache(
        sharedKvStoreContext,
        "snapshot",
        async (response, signal) => {
          const value = await decodeSnapshot(
            await response.arrayBuffer(),
            signal,
          );
          return { data: value, size: value.estimatedSize };
        },
      ),
  );
  return cache.get(getSnapshotUrl(baseUrl, id), options);
}

export function getManifest(
  sharedKvStoreContext: SharedKvStoreContextCounterpart,
  baseUrl: string,
  id: ManifestId,
  options: Partial<ProgressOptions>,
) {
  const cache = sharedKvStoreContext.chunkManager.memoize.get(
    "icechunk:manifest",
    () =>
      makeMetadataCache(
        sharedKvStoreContext,
        "manifest",
        async (response, signal) => {
          const value = await decodeManifest(
            await response.arrayBuffer(),
            signal,
          );
          return { data: value, size: value.estimatedSize };
        },
      ),
  );
  return cache.get(getManifestUrl(baseUrl, id), options);
}

export function getRef(
  sharedKvStoreContext: SharedKvStoreContextCounterpart,
  url: string,
  options: Partial<ProgressOptions>,
) {
  const cache = sharedKvStoreContext.chunkManager.memoize.get(
    "icechunk:ref",
    () =>
      makeOptionalMetadataCache(
        sharedKvStoreContext,
        "ref",
        async (response) => ({
          data: decodeRef(await response.json()),
          size: 0,
        }),
      ),
  );
  return cache.get(url, options);
}

// Icechunk 0.1 writes one ref file per branch version, and the newest has
// the lowest name.
export function getBranch(
  sharedKvStoreContext: SharedKvStoreContextCounterpart,
  url: string,
  options: Partial<ProgressOptions>,
) {
  const cache = sharedKvStoreContext.chunkManager.memoize.get(
    "icechunk:branch",
    () => {
      const cache = new SimpleAsyncCache<string, string>(
        sharedKvStoreContext.chunkManager.addRef(),
        {
          get: async (url: string, progressOptions: ProgressOptions) => {
            using _span = new ProgressSpan(progressOptions.progressListener, {
              message: `Resolving icechunk branch at ${url}`,
            });
            try {
              const listResponse =
                await sharedKvStoreContext.kvStoreContext.list(url, {
                  ...progressOptions,
                  responseKeys: "suffix",
                });
              const headKey = listResponse.entries.find((entry) =>
                isBranchRef(entry.key),
              );
              if (headKey === undefined) {
                throw new Error(`Failed to find any refs`);
              }
              const snapshotId = await getRef(
                sharedKvStoreContext,
                pipelineUrlJoin(url, headKey.key),
                progressOptions,
              );
              if (snapshotId === undefined) {
                throw new Error(`Ref ${headKey.key} is missing`);
              }
              return { data: snapshotId, size: 0 };
            } catch (e) {
              throw new Error(`Error resolving icechunk branch at ${url}`, {
                cause: e,
              });
            }
          },
        },
      );
      cache.registerDisposer(sharedKvStoreContext.addRef());
      return cache;
    },
  );
  return cache.get(url, options);
}

export function getTag(
  sharedKvStoreContext: SharedKvStoreContextCounterpart,
  url: string,
  options: Partial<ProgressOptions>,
) {
  const cache = sharedKvStoreContext.chunkManager.memoize.get(
    "icechunk:tag",
    () => {
      const cache = new SimpleAsyncCache<string, string>(
        sharedKvStoreContext.chunkManager.addRef(),
        {
          get: async (url: string, progressOptions: ProgressOptions) => {
            using _span = new ProgressSpan(progressOptions.progressListener, {
              message: `Resolving icechunk tag at ${url}`,
            });
            try {
              const [tagResponse, deletedResponse] = await Promise.all([
                getRef(
                  sharedKvStoreContext,
                  pipelineUrlJoin(url, "ref.json"),
                  progressOptions,
                ),
                sharedKvStoreContext.kvStoreContext.stat(
                  pipelineUrlJoin(url, "ref.json.deleted"),
                  progressOptions,
                ),
              ]);
              if (deletedResponse !== undefined) {
                throw new Error(`Tag is marked as deleted`);
              }
              if (tagResponse === undefined) {
                throw new Error(`Tag not found`);
              }
              return { data: tagResponse, size: 0 };
            } catch (e) {
              throw new Error(`Error resolving icechunk tag at ${url}`, {
                cause: e,
              });
            }
          },
        },
      );
      cache.registerDisposer(sharedKvStoreContext.addRef());
      return cache;
    },
  );
  return cache.get(url, options);
}

// The repo info wins over ref files that a migration left behind.
export async function resolveRefSpec(
  sharedKvStoreContext: SharedKvStoreContextCounterpart,
  url: string,
  refSpec: RefSpec,
  options: Partial<ProgressOptions>,
): Promise<SnapshotId> {
  if ("snapshot" in refSpec) {
    return refSpec.snapshot;
  }
  const branchUrl =
    "branch" in refSpec
      ? pipelineUrlJoin(url, `refs/branch.${refSpec.branch}/`)
      : undefined;
  const [repoInfo, refFile] = await Promise.allSettled([
    getRepoInfo(sharedKvStoreContext, url, options),
    "branch" in refSpec
      ? getRef(
          sharedKvStoreContext,
          pipelineUrlJoin(branchUrl!, "ref.json"),
          options,
        )
      : getTag(
          sharedKvStoreContext,
          pipelineUrlJoin(url, `refs/tag.${refSpec.tag}/`),
          options,
        ),
  ]);
  if (repoInfo.status === "rejected") {
    throw repoInfo.reason;
  }
  if (repoInfo.value !== undefined) {
    return resolveRefInRepoInfo(
      sharedKvStoreContext,
      url,
      refSpec,
      repoInfo.value,
      options,
    );
  }
  if (refFile.status === "rejected") {
    throw refFile.reason;
  }
  return refFile.value ?? getBranch(sharedKvStoreContext, branchUrl!, options);
}

type NamedRefSpec = Exclude<RefSpec, { snapshot: string }>;

function findRef(repoInfo: RepoInfo, refSpec: NamedRefSpec) {
  return "branch" in refSpec
    ? repoInfo.branches.get(refSpec.branch)
    : repoInfo.tags.get(refSpec.tag);
}

async function resolveRefInRepoInfo(
  sharedKvStoreContext: SharedKvStoreContextCounterpart,
  url: string,
  refSpec: NamedRefSpec,
  cachedRepoInfo: RepoInfo,
  options: Partial<ProgressOptions>,
): Promise<SnapshotId> {
  const snapshotId = findRef(cachedRepoInfo, refSpec);
  if (snapshotId !== undefined) return snapshotId;
  const repoInfo =
    (await readLatestRepoInfo(sharedKvStoreContext, url, options)) ??
    cachedRepoInfo;
  const latestSnapshotId = findRef(repoInfo, refSpec);
  if (latestSnapshotId !== undefined) return latestSnapshotId;
  const kind = "branch" in refSpec ? "branch" : "tag";
  const reason =
    "tag" in refSpec && repoInfo.deletedTags.has(refSpec.tag)
      ? "Tag is marked as deleted"
      : "Not found";
  throw new Error(
    `Error resolving icechunk ${kind} ${JSON.stringify(formatRefSpec(refSpec))} at ${getRepoInfoUrl(url)}`,
    { cause: new Error(reason) },
  );
}
