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

import { WithParameters } from "#src/chunk_manager/backend.js";
import { SwcSkeletonSourceParameters } from "#src/datasource/swc/base.js";
import type { SwcSegments } from "#src/datasource/swc/segment_ids.js";
import { WithSharedKvStoreContextCounterpart } from "#src/kvstore/backend.js";
import { readKvStore } from "#src/kvstore/index.js";
import type { SkeletonChunk } from "#src/skeleton/backend.js";
import { SkeletonSource } from "#src/skeleton/backend.js";
import {
  decodeSwcSkeletonChunk,
  SwcValidation,
} from "#src/skeleton/decode_swc_skeleton.js";
import type { RPC } from "#src/worker_rpc.js";
import { registerSharedObject } from "#src/worker_rpc.js";

@registerSharedObject()
export class SwcSkeletonSource extends WithParameters(
  WithSharedKvStoreContextCounterpart(SkeletonSource),
  SwcSkeletonSourceParameters,
) {
  private readonly kvStore =
    this.sharedKvStoreContext.kvStoreContext.getKvStore(this.parameters.url);
  private readonly fileNameBySegmentId: Map<bigint, string>;

  constructor(rpc: RPC, options: any) {
    super(rpc, options);
    const segments: SwcSegments = options.segments;
    this.fileNameBySegmentId = new Map(
      Array.from(segments.ids, (segmentId, index) => [
        segmentId,
        segments.fileNames[index],
      ]),
    );
  }

  async download(chunk: SkeletonChunk, signal: AbortSignal): Promise<void> {
    const fileName = this.fileNameBySegmentId.get(chunk.objectId);
    if (fileName === undefined) {
      throw new Error(`No SWC file has segment ID ${chunk.objectId}`);
    }
    const { kvStore } = this;
    const response = await readKvStore(
      kvStore.store,
      `${kvStore.path}${fileName}`,
      { signal, throwIfMissing: true },
    );
    decodeSwcSkeletonChunk(chunk, await response.response.text(), {
      validation: SwcValidation.STRICT,
    });
  }
}
