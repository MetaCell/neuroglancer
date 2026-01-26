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
import { SpatiallyIndexedSkeletonSourceBackend, SpatiallyIndexedSkeletonChunk, SkeletonSource, SkeletonChunk } from "#src/skeleton/backend.js";
import { CatmaidClient } from "#src/datasource/catmaid/api.js";
import { CatmaidSkeletonSourceParameters, CatmaidCompleteSkeletonSourceParameters } from "#src/datasource/catmaid/base.js";
import { registerSharedObject } from "#src/worker_rpc.js";
import { vec3 } from "#src/util/geom.js";
import { WithSharedCredentialsProviderCounterpart } from "#src/credentials_provider/shared_counterpart.js";
import type { CatmaidToken } from "#src/datasource/catmaid/api.js";


@registerSharedObject()
export class CatmaidSpatiallyIndexedSkeletonSourceBackend extends WithParameters(
    WithSharedCredentialsProviderCounterpart<CatmaidToken>()(SpatiallyIndexedSkeletonSourceBackend),
    CatmaidSkeletonSourceParameters
) {
    get client(): CatmaidClient {
        const { catmaidParameters } = this.parameters;
        return new CatmaidClient(
            catmaidParameters.url,
            catmaidParameters.projectId,
            catmaidParameters.token,
            this.credentialsProvider
        );
    }

    constructor(...args: any[]) {
        super(args[0], args[1]);
    }

    async download(chunk: SpatiallyIndexedSkeletonChunk, _signal: AbortSignal) {
        const { chunkGridPosition } = chunk;
        const { chunkDataSize } = this.spec;

        const localMin = vec3.multiply(vec3.create(), chunkGridPosition as unknown as vec3, chunkDataSize as unknown as vec3);
        const localMax = vec3.add(vec3.create(), localMin, chunkDataSize as unknown as vec3);

        const bbox = {
            min: { x: localMin[0], y: localMin[1], z: localMin[2] },
            max: { x: localMax[0], y: localMax[1], z: localMax[2] },
        };
        
        // Use currentLod from the source backend
        const lodValue = this.currentLod;
        // Get cache provider from parameters (passed from frontend)
        const cacheProvider = this.parameters.catmaidParameters.cacheProvider;
        const nodes = await this.client.fetchNodes(bbox, lodValue, cacheProvider);

        const numVertices = nodes.length;
        const vertexPositions = new Float32Array(numVertices * 3);
        const vertexAttributes = new Float32Array(numVertices);
        const indices: number[] = [];
        const nodeMap = new Map<number, number>();


        for (let i = 0; i < numVertices; ++i) {
            const node = nodes[i];
            nodeMap.set(node.id, i);
            vertexPositions[i * 3] = node.x;
            vertexPositions[i * 3 + 1] = node.y;
            vertexPositions[i * 3 + 2] = node.z;
            vertexAttributes[i] = node.skeleton_id;
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

        // Pack only segment IDs into vertexAttributes (positions are in vertexPositions)
        chunk.vertexAttributes = [vertexAttributes];
    }
}

@registerSharedObject()
export class CatmaidSkeletonSourceBackend extends WithParameters(
    WithSharedCredentialsProviderCounterpart<CatmaidToken>()(SkeletonSource),
    CatmaidCompleteSkeletonSourceParameters
) {
    get client(): CatmaidClient {
        const { catmaidParameters } = this.parameters;
        return new CatmaidClient(
            catmaidParameters.url,
            catmaidParameters.projectId,
            catmaidParameters.token,
            this.credentialsProvider
        );
    }

    constructor(...args: any[]) {
        super(args[0], args[1]);
    }

    async download(chunk: SkeletonChunk, _signal: AbortSignal) {
        const skeletonId = Number(chunk.objectId);
        const nodes = await this.client.getSkeleton(skeletonId);

        const numVertices = nodes.length;
        const vertexPositions = new Float32Array(numVertices * 3);
        const vertexAttributes = new Float32Array(numVertices);
        const indices: number[] = [];
        const nodeMap = new Map<number, number>();

        // Build vertex positions and create node ID to vertex index mapping
        for (let i = 0; i < numVertices; ++i) {
            const node = nodes[i];
            nodeMap.set(node.id, i);
            vertexPositions[i * 3] = node.x;
            vertexPositions[i * 3 + 1] = node.y;
            vertexPositions[i * 3 + 2] = node.z;
            vertexAttributes[i] = node.skeleton_id;
        }

        // Build edge indices from parent-child relationships
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
        chunk.vertexAttributes = [vertexAttributes];
    }
}
