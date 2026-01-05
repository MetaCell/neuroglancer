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
import { SpatiallyIndexedSkeletonSourceBackend, SpatiallyIndexedSkeletonChunk } from "#src/skeleton/backend.js";
import { CatmaidClient } from "#src/datasource/catmaid/api.js";
import { CatmaidSkeletonSourceParameters } from "#src/datasource/catmaid/base.js";
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
        console.log(`[CATMAID-BACKEND] Fetching nodes for bounds:`, bbox);

        const nodes = await this.client.fetchNodes(bbox);

        console.log('[CATMAID-BACKEND] Downloaded nodes:', nodes);

        const numVertices = nodes.length;
        const vertexPositions = new Float32Array(numVertices * 3);
        const vertexAttributes = new Float32Array(numVertices);
        const indices: number[] = [];
        const nodeMap = new Map<number, number>();

        // Track unique skeleton IDs for debugging
        const uniqueSkeletonIds = new Set<number>();

        for (let i = 0; i < numVertices; ++i) {
            const node = nodes[i];
            nodeMap.set(node.id, i);
            vertexPositions[i * 3] = node.x;
            vertexPositions[i * 3 + 1] = node.y;
            vertexPositions[i * 3 + 2] = node.z;
            vertexAttributes[i] = node.skeleton_id;
            uniqueSkeletonIds.add(node.skeleton_id);
        }

        console.log('[CATMAID-BACKEND] Unique skeleton IDs in chunk:', Array.from(uniqueSkeletonIds));
        console.log('[CATMAID-BACKEND] Sample segment IDs:', Array.from(vertexAttributes.slice(0, Math.min(10, numVertices))));

        for (let i = 0; i < numVertices; ++i) {
            const node = nodes[i];
            if (node.parent_id !== null) {
                const parentIndex = nodeMap.get(node.parent_id);
                if (parentIndex !== undefined) {
                    indices.push(i, parentIndex);
                }
            }
        }

        console.log(`[CATMAID-BACKEND] Created ${indices.length / 2} edges from parent-child relationships`);

        chunk.vertexPositions = vertexPositions;
        chunk.indices = new Uint32Array(indices);

        // Pack only segment IDs into vertexAttributes (positions are in vertexPositions)
        // This will be serialized as a separate attribute
        chunk.vertexAttributes = [vertexAttributes];
    }
}
