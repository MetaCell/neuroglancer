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

import {
    AnnotationGeometryChunk,
    AnnotationGeometryChunkSourceBackend,
    AnnotationGeometryData,
    AnnotationSource,
} from "#src/annotation/backend.js";
import {
    Annotation,
    AnnotationId,
    AnnotationType,
    makeAnnotationPropertySerializers,
    annotationTypeHandlers,
} from "#src/annotation/index.js";
import { WithParameters } from "#src/chunk_manager/backend.js";
import { CatmaidClient, CatmaidNode } from "#src/datasource/catmaid/api.js";
import { 
    CatmaidAnnotationSourceParameters,
    CatmaidAnnotationGeometryChunkSourceParameters 
} from "#src/datasource/catmaid/base.js";
import { registerSharedObject } from "#src/worker_rpc.js";
import { vec3 } from "#src/util/geom.js";

function serializeAnnotations(
    annotations: Annotation[]
): AnnotationGeometryData {
    const rank = 3;
    const properties: any[] = [];
    const propertySerializers = makeAnnotationPropertySerializers(
        rank,
        properties
    );

    const typeToAnnotations = new Map<AnnotationType, Annotation[]>();
    for (const ann of annotations) {
        if (!typeToAnnotations.has(ann.type)) {
            typeToAnnotations.set(ann.type, []);
        }
        typeToAnnotations.get(ann.type)!.push(ann);
    }

    const geometryData = new AnnotationGeometryData();
    geometryData.typeToIds = [];
    geometryData.typeToOffset = [];
    geometryData.typeToIdMaps = [];
    geometryData.typeToInstanceCounts = [];
    geometryData.typeToSize = [];

    let totalBytes = 0;
    for (const [type, anns] of typeToAnnotations) {
        const serializer = propertySerializers[type];
        const handler = annotationTypeHandlers[type];
        const geometryBytes = handler.serializedBytes(rank);
        const stride = geometryBytes + serializer.serializedBytes;
        totalBytes += anns.length * stride;
    }

    geometryData.data = new Uint8Array(totalBytes);
    const dv = new DataView(geometryData.data.buffer);
    let currentOffset = 0;

    for (const [type, anns] of typeToAnnotations) {
        geometryData.typeToOffset[type] = currentOffset;
        geometryData.typeToIds[type] = anns.map((a) => a.id);
        geometryData.typeToIdMaps[type] = new Map(anns.map((a, i) => [a.id, i]));
        geometryData.typeToSize[type] = anns.length;
        geometryData.typeToInstanceCounts[type] = anns.map((_, i) => i);

        const serializer = propertySerializers[type];
        const handler = annotationTypeHandlers[type];
        const geometryBytes = handler.serializedBytes(rank);
        const stride = geometryBytes + serializer.serializedBytes;

        for (let i = 0; i < anns.length; ++i) {
            const ann = anns[i];
            const offset = currentOffset + i * stride;

            handler.serialize(dv, offset, true, rank, ann);

            serializer.serialize(
                dv,
                offset + geometryBytes,
                i,
                anns.length,
                true,
                ann.properties
            );
        }
        currentOffset += anns.length * stride;
    }

    return geometryData;
}

@registerSharedObject()
export class CatmaidAnnotationGeometryChunkSourceBackend extends WithParameters(
    AnnotationGeometryChunkSourceBackend,
    CatmaidAnnotationGeometryChunkSourceParameters
) {
    get client(): CatmaidClient {
        const { catmaidParameters } = this.parameters;
        return new CatmaidClient(
            catmaidParameters.url,
            catmaidParameters.projectId,
            catmaidParameters.token
        );
    }

    async download(chunk: AnnotationGeometryChunk, _signal: AbortSignal) {
        console.log('CATMAID Backend: download called for chunk:', chunk.chunkGridPosition);
        const { chunkGridPosition } = chunk;
        const { spec } = this;
        const { chunkDataSize } = spec;

        // Calculate bounding box in voxel coordinates
        const minVoxel = vec3.create();
        const maxVoxel = vec3.create();
        for (let i = 0; i < 3; ++i) {
            minVoxel[i] = chunkGridPosition[i] * chunkDataSize[i];
            maxVoxel[i] = (chunkGridPosition[i] + 1) * chunkDataSize[i];
        }
        
        console.log('CATMAID Backend: Querying box:', minVoxel, 'to', maxVoxel);

        // Since we don't have the full transform chain here easily, and CATMAID usually works in physical coordinates (nm),
        // we assume the chunk grid is aligned with physical space or we need to apply the transform.
        // For this POC, let's assume 1 voxel = 1 nm if no transform is applied, or use the transform if available.
        // spec.chunkToMultiscaleTransform is available.

        // TODO: Apply transform correctly. For now, assuming identity/scaling is handled elsewhere or simple mapping.
        // If chunkToMultiscaleTransform is identity, then voxel coords = physical coords.

        const nodes = await this.client.boxQuery(minVoxel, maxVoxel);
        
        console.log('CATMAID Backend: Retrieved', nodes.length, 'nodes');

        const annotations: Annotation[] = [];
        const nodeMap = new Map<number, CatmaidNode>();
        for (const node of nodes) {
            nodeMap.set(node.id, node);
            annotations.push({
                type: AnnotationType.POINT,
                id: `node_${node.id}`,
                point: new Float32Array([node.x, node.y, node.z]),
                properties: [],
                description: undefined,
                relatedSegments: undefined,
            });
        }

        for (const node of nodes) {
            if (node.parent_id && nodeMap.has(node.parent_id)) {
                const parent = nodeMap.get(node.parent_id)!;
                annotations.push({
                    type: AnnotationType.LINE,
                    id: `edge_${node.id}_${node.parent_id}`,
                    pointA: new Float32Array([node.x, node.y, node.z]),
                    pointB: new Float32Array([parent.x, parent.y, parent.z]),
                    properties: [],
                    description: undefined,
                    relatedSegments: undefined,
                });
            }
        }

        console.log('CATMAID Backend: Created', annotations.length, 'annotations');
        
        chunk.data = serializeAnnotations(annotations);
        
        console.log('CATMAID Backend: Chunk data serialized successfully');
    }
}

@registerSharedObject()
export class CatmaidAnnotationSource extends WithParameters(
    AnnotationSource,
    CatmaidAnnotationSourceParameters
) {
    constructor(rpc: any, options: any) {
        console.log('CATMAID Backend: CatmaidAnnotationSource constructor called with options:', options);
        console.log('CATMAID Backend: options.metadataChunkSource:', options.metadataChunkSource);
        console.log('CATMAID Backend: options.segmentFilteredSource:', options.segmentFilteredSource);
        console.log('CATMAID Backend: options.chunkManager:', options.chunkManager);
        super(rpc, options);
        console.log('CATMAID Backend: CatmaidAnnotationSource constructor completed');
    }

    get client(): CatmaidClient {
        const { catmaidParameters } = this.parameters;
        return new CatmaidClient(
            catmaidParameters.url,
            catmaidParameters.projectId,
            catmaidParameters.token
        );
    }

    async add(annotation: Annotation): Promise<AnnotationId> {
        if (annotation.type === AnnotationType.POINT) {
            const point = (annotation as any).point;
            const id = await this.client.addNode(
                1, // TODO: Handle skeleton ID selection
                point[0],
                point[1],
                point[2]
            );
            return `node_${id}`;
        }
        throw new Error("Only points supported for add in POC");
    }

    async delete(id: AnnotationId): Promise<void> {
        if (id.startsWith("node_")) {
            const nodeId = parseInt(id.substring(5));
            await this.client.deleteNode(nodeId);
        }
    }

    async update(id: AnnotationId, newAnnotation: Annotation): Promise<void> {
        if (newAnnotation.type === AnnotationType.POINT) {
            const nodeId = parseInt(id.substring(5));
            const point = (newAnnotation as any).point;
            await this.client.moveNode(nodeId, point[0], point[1], point[2]);
        }
    }
}
