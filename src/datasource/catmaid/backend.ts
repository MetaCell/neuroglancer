/**
 * @license
 * Copyright 2024 Google Inc.
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
import { SkeletonSource, SkeletonChunk } from "#src/skeleton/backend.js";
import { CatmaidClient } from "#src/datasource/catmaid/api.js";
import { CatmaidSkeletonSourceParameters } from "#src/datasource/catmaid/base.js";
import { registerSharedObject } from "#src/worker_rpc.js";

@registerSharedObject()
export class CatmaidSkeletonSource extends WithParameters(
    SkeletonSource,
    CatmaidSkeletonSourceParameters
) {
    get client(): CatmaidClient {
        const { catmaidParameters } = this.parameters;
        return new CatmaidClient(
            catmaidParameters.url,
            catmaidParameters.projectId,
            catmaidParameters.token
        );
    }

    async download(chunk: SkeletonChunk, _signal: AbortSignal) {
        const { objectId } = chunk;
        // objectId is bigint, but CATMAID uses number.
        // We assume objectId fits in number.
        const skeletonId = Number(objectId);

        const nodes = await this.client.getSkeleton(skeletonId);

        // Convert nodes to vertexPositions and indices
        const numVertices = nodes.length;
        const vertexPositions = new Float32Array(numVertices * 3);
        const indices: number[] = [];
        const nodeMap = new Map<number, number>(); // id -> index

        for (let i = 0; i < numVertices; ++i) {
            const node = nodes[i];
            nodeMap.set(node.id, i);
            vertexPositions[i * 3] = node.x;
            vertexPositions[i * 3 + 1] = node.y;
            vertexPositions[i * 3 + 2] = node.z;
        }

        for (let i = 0; i < numVertices; ++i) {
            const node = nodes[i];
            if (node.parent_id !== null) {
                const parentIndex = nodeMap.get(node.parent_id);
                if (parentIndex !== undefined) {
                    indices.push(i, parentIndex);
                }
            }
        }

        chunk.vertexPositions = vertexPositions;
        chunk.indices = new Uint32Array(indices);
    }
}
