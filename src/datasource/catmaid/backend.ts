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
import { ChunkLayout } from "#src/sliceview/chunk_layout.js";



@registerSharedObject()
export class CatmaidSpatiallyIndexedSkeletonSourceBackend extends WithParameters(
    SpatiallyIndexedSkeletonSourceBackend,
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

    async download(chunk: SpatiallyIndexedSkeletonChunk, _signal: AbortSignal) {
        const { chunkGridPosition } = chunk;
        const { chunkDataSize } = this.spec;
        const chunkLayout = ChunkLayout.fromObject(this.spec.chunkLayout);
        const { transform } = chunkLayout;

        const localMin = vec3.multiply(vec3.create(), chunkGridPosition as unknown as vec3, chunkDataSize as unknown as vec3);
        const localMax = vec3.add(vec3.create(), localMin, chunkDataSize as unknown as vec3);

        const globalMin = vec3.transformMat4(vec3.create(), localMin, transform);
        const globalMax = vec3.transformMat4(vec3.create(), localMax, transform);

        const min = vec3.min(vec3.create(), globalMin, globalMax);
        const max = vec3.max(vec3.create(), globalMin, globalMax);

        const nodes = await this.client.fetchNodes({
            min: { x: min[0], y: min[1], z: min[2] },
            max: { x: max[0], y: max[1], z: max[2] },
        });

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
        
        const positionsBytes = new Uint8Array(vertexPositions.buffer);
        const segmentsBytes = new Uint8Array(vertexAttributes.buffer);

        const totalBytes = positionsBytes.byteLength + segmentsBytes.byteLength;
        const packedData = new Uint8Array(totalBytes);
        packedData.set(positionsBytes, 0);
        packedData.set(segmentsBytes, positionsBytes.byteLength);

        chunk.vertexAttributes = [packedData];

        const positionsByteLength = numVertices * 3 * 4;
        const segmentsByteLength = numVertices * 1 * 4;
        (chunk as any).vertexAttributeOffsets = new Uint32Array([
            0,
            positionsByteLength,
            positionsByteLength + segmentsByteLength
        ]);
    }
}
