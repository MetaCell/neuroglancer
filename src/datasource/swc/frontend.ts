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

import type { ChunkManager } from "#src/chunk_manager/frontend.js";
import { WithParameters } from "#src/chunk_manager/frontend.js";
import {
  makeCoordinateSpace,
  makeIdentityTransform,
} from "#src/coordinate_transform.js";
import type {
  DataSource,
  DataSourceLookupResult,
  GetKvStoreBasedDataSourceOptions,
  KvStoreBasedDataSourceProvider,
} from "#src/datasource/index.js";
import { SwcSkeletonSourceParameters } from "#src/datasource/swc/base.js";
import type { SwcSegments } from "#src/datasource/swc/segment_ids.js";
import {
  assignSegmentIds,
  fnv1a64,
  getSwcLabel,
  swcExtensionPattern,
} from "#src/datasource/swc/segment_ids.js";
import { WithSharedKvStoreContext } from "#src/kvstore/chunk_source_frontend.js";
import type { SharedKvStoreContext } from "#src/kvstore/frontend.js";
import {
  ensureEmptyUrlSuffix,
  kvstoreEnsureDirectoryPipelineUrl,
} from "#src/kvstore/url.js";
import { SegmentPropertyMap } from "#src/segmentation_display_state/property_map.js";
import { SkeletonSource } from "#src/skeleton/frontend.js";
import { swcVertexAttributes } from "#src/skeleton/swc_base.js";
import type { Borrowed } from "#src/util/disposable.js";
import type { ProgressOptions } from "#src/util/progress_listener.js";
import type { RPC } from "#src/worker_rpc.js";

// `segments` is sent to the worker but kept out of the chunk source key, which must stay small.
class SwcSegmentSkeletonSource extends SkeletonSource {
  declare OPTIONS: { segments: SwcSegments };
  private readonly segments: SwcSegments;

  constructor(
    chunkManager: Borrowed<ChunkManager>,
    options: { segments: SwcSegments },
  ) {
    super(chunkManager, options);
    this.segments = options.segments;
  }

  initializeCounterpart(rpc: RPC, options: any) {
    options.segments = this.segments;
    super.initializeCounterpart(rpc, options);
  }

  get vertexAttributes() {
    return swcVertexAttributes;
  }
}

class SwcSkeletonSource extends WithParameters(
  WithSharedKvStoreContext(SwcSegmentSkeletonSource),
  SwcSkeletonSourceParameters,
) {}

async function listSwcFileNames(
  sharedKvStoreContext: SharedKvStoreContext,
  url: string,
  options: Partial<ProgressOptions>,
): Promise<string[]> {
  const { entries, directories } =
    await sharedKvStoreContext.kvStoreContext.list(url, {
      responseKeys: "suffix",
      ...options,
    });
  if (directories.length > 0) {
    throw new Error(
      `SWC folder ${url} contains subfolders, which are not supported: ${directories.join(", ")}`,
    );
  }
  const fileNames = entries
    .map((entry) => entry.key)
    .filter((key) => swcExtensionPattern.test(key));
  if (fileNames.length === 0) {
    throw new Error(`SWC folder ${url} contains no .swc files`);
  }
  return fileNames;
}

function getLabelSegmentPropertyMap(segments: SwcSegments): SegmentPropertyMap {
  return new SegmentPropertyMap({
    inlineProperties: {
      ids: segments.ids,
      properties: [
        {
          id: "label",
          type: "label",
          values: segments.fileNames.map(getSwcLabel),
        },
      ],
    },
  });
}

async function getSwcDataSource(
  sharedKvStoreContext: SharedKvStoreContext,
  url: string,
  options: Partial<ProgressOptions>,
): Promise<DataSource> {
  const fileNames = await listSwcFileNames(sharedKvStoreContext, url, options);
  const segments = assignSegmentIds(fileNames, fnv1a64);
  const skeletons = sharedKvStoreContext.chunkManager.getChunkSource(
    SwcSkeletonSource,
    {
      sharedKvStoreContext,
      segments,
      parameters: { url },
    },
  );
  return {
    canonicalUrl: `${url}|swc:`,
    modelTransform: makeIdentityTransform(
      makeCoordinateSpace({
        names: ["x", "y", "z"],
        units: ["m", "m", "m"],
        scales: Float64Array.of(1e-6, 1e-6, 1e-6),
      }),
    ),
    subsources: [
      { id: "default", default: true, subsource: { mesh: skeletons } },
      {
        id: "properties",
        default: true,
        subsource: {
          segmentPropertyMap: getLabelSegmentPropertyMap(segments),
        },
      },
    ],
  };
}

export class SwcDataSource implements KvStoreBasedDataSourceProvider {
  get scheme() {
    return "swc";
  }
  get expectsDirectory() {
    return true;
  }
  get description() {
    return "Directory of SWC skeleton files";
  }

  get(
    options: GetKvStoreBasedDataSourceOptions,
  ): Promise<DataSourceLookupResult> {
    ensureEmptyUrlSuffix(options.url);
    const url = kvstoreEnsureDirectoryPipelineUrl(options.kvStoreUrl);
    return options.registry.chunkManager.memoize.getAsync(
      { type: "swc:get", url },
      options,
      (progressOptions) =>
        getSwcDataSource(
          options.registry.sharedKvStoreContext,
          url,
          progressOptions,
        ),
    );
  }
}
