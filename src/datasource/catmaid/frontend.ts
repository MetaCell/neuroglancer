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
    makeCoordinateSpace,
    makeIdentityTransform,
} from "#src/coordinate_transform.js";
import { makeDataBoundsBoundingBoxAnnotationSet } from "#src/annotation/index.js";
import { SpatiallyIndexedSkeletonSource } from "#src/skeleton/frontend.js";
import { WithParameters } from "#src/chunk_manager/frontend.js";
import {
    DataSource,
    DataSourceProvider,
    GetDataSourceOptions,
} from "#src/datasource/index.js";
import {
    CatmaidSkeletonSourceParameters,
    CatmaidDataSourceParameters,
} from "#src/datasource/catmaid/base.js";
import { mat4, vec3 } from "#src/util/geom.js";
import { CatmaidClient } from "#src/datasource/catmaid/api.js";
import { DataType } from "#src/util/data_type.js";
import { ChunkLayout } from "#src/sliceview/chunk_layout.js";
import { makeSliceViewChunkSpecification } from "#src/sliceview/base.js";
import {
    InlineSegmentPropertyMap,
    SegmentPropertyMap,
    normalizeInlineSegmentPropertyMap,
} from "#src/segmentation_display_state/property_map.js";
import { CatmaidToken, credentialsKey } from "#src/datasource/catmaid/api.js";
import { CredentialsProvider } from "#src/credentials_provider/index.js";
import { WithCredentialsProvider } from "#src/credentials_provider/chunk_source_frontend.js";
import "#src/datasource/catmaid/register_credentials_provider.js";

export class CatmaidSpatiallyIndexedSkeletonSource extends WithParameters(
    WithCredentialsProvider<CatmaidToken>()(SpatiallyIndexedSkeletonSource),
    CatmaidSkeletonSourceParameters
) {
    getChunk(chunkData: any) {
        return super.getChunk(chunkData);
    }
    static encodeOptions(options: any) {
        return super.encodeOptions(options);
    }
}

export class CatmaidDataSourceProvider implements DataSourceProvider {
    get scheme() {
        return "catmaid";
    }

    get description() {
        return "CATMAID";
    }

    async get(options: GetDataSourceOptions): Promise<DataSource> {
        const { providerUrl } = options;

        // Remove scheme if present to handle "catmaid://"
        let cleanUrl = providerUrl;
        if (cleanUrl.startsWith("catmaid://")) {
            cleanUrl = cleanUrl.substring("catmaid://".length);
        }

        const lastSlash = cleanUrl.lastIndexOf('/');
        if (lastSlash === -1) {
            throw new Error("Invalid CATMAID URL. Expected format: catmaid://<base_url>/<project_id>");
        }

        const projectIdStr = cleanUrl.substring(lastSlash + 1);
        const projectId = parseInt(projectIdStr);
        if (isNaN(projectId)) {
            throw new Error(`Invalid project ID: ${projectIdStr}`);
        }

        let baseUrl = cleanUrl.substring(0, lastSlash);
        if (!baseUrl.startsWith("http")) {
            baseUrl = "https://" + baseUrl;
        }

        const credentialsProvider =
            options.registry.credentialsManager.getCredentialsProvider(
                credentialsKey,
                { serverUrl: baseUrl },
            ) as CredentialsProvider<CatmaidToken>;

        const client = new CatmaidClient(baseUrl, projectId, undefined, credentialsProvider);

        const [dimensions, resolution, gridCellSize, skeletonIds] = await Promise.all([
            client.getDimensions(),
            client.getResolution(),
            client.getGridCellSize(),
            client.listSkeletons(),
        ]);

        if (!dimensions || !resolution) {
            throw new Error("Failed to fetch CATMAID stack metadata");
        }

        // Resolution is in nm per voxel, convert to meters
        const scaleFactors = Float64Array.from([
            resolution.x * 1e-9,
            resolution.y * 1e-9,
            resolution.z * 1e-9,
        ]);

        // Dimensions are already in voxel coordinates from the API
        const lowerBounds = Float64Array.from([
            dimensions.min.x,
            dimensions.min.y,
            dimensions.min.z,
        ]);
        const upperBounds = Float64Array.from([
            dimensions.max.x,
            dimensions.max.y,
            dimensions.max.z,
        ]);

        const modelSpace = makeCoordinateSpace({
            names: ["x", "y", "z"],
            units: ["m", "m", "m"],
            scales: scaleFactors,
            boundingBoxes: [
                {
                    box: {
                        lowerBounds,
                        upperBounds,
                    },
                    transform: Float64Array.from([1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0]),
                }
            ]
        });

        const parameters = new CatmaidSkeletonSourceParameters();
        parameters.catmaidParameters = new CatmaidDataSourceParameters();
        parameters.catmaidParameters.url = baseUrl;
        parameters.catmaidParameters.projectId = projectId;
        parameters.url = providerUrl;

        parameters.metadata = {
            transform: mat4.create(),
            vertexAttributes: new Map([
                ["segment", { dataType: DataType.FLOAT32, numComponents: 1 }],
            ]),
            sharding: undefined
        };

        const rank = 3;

        const upperVoxelBound = new Float32Array(rank);
        for (let i = 0; i < rank; ++i) {
            upperVoxelBound[i] = upperBounds[i];
        }

        // Use CATMAID's grid cell size for chunking
        const chunkDataSize = Uint32Array.from([
            gridCellSize.x,
            gridCellSize.y,
            gridCellSize.z,
        ]);

        const chunkLayoutTransform = mat4.create();
        mat4.fromScaling(chunkLayoutTransform, vec3.fromValues(
            scaleFactors[0],
            scaleFactors[1],
            scaleFactors[2]
        ));

        // Create chunk layout
        const chunkLayout = new ChunkLayout(
            vec3.fromValues(chunkDataSize[0], chunkDataSize[1], chunkDataSize[2]),
            chunkLayoutTransform,
            rank
        );

        // Create chunk specification
        const spec = {
            ...makeSliceViewChunkSpecification({
                rank,
                chunkDataSize,
                upperVoxelBound,
            }),
            chunkLayout,
        };

        const source = options.registry.chunkManager.getChunkSource(
            CatmaidSpatiallyIndexedSkeletonSource,
            { parameters, spec, credentialsProvider }
        );

        // Create SegmentPropertyMap
        const ids = new BigUint64Array(skeletonIds.length);
        const labels = new Array<string>(skeletonIds.length);
        for (let i = 0; i < skeletonIds.length; ++i) {
            ids[i] = BigInt(skeletonIds[i]);
            labels[i] = skeletonIds[i].toString();
        }

        const inlineProperties: InlineSegmentPropertyMap = normalizeInlineSegmentPropertyMap({
            ids,
            properties: [{
                id: "label",
                type: "label",
                values: labels,
            }],
        });

        const propertyMap = new SegmentPropertyMap({
            inlineProperties,
        });

        const subsources = [
            {
                id: "skeletons",
                default: true,
                subsource: { mesh: source },
            },
            {
                id: "properties",
                default: true,
                subsource: { segmentPropertyMap: propertyMap },
            },
            {
                id: "bounds",
                default: true,
                subsource: {
                    staticAnnotations: makeDataBoundsBoundingBoxAnnotationSet(
                        modelSpace.bounds
                    ),
                },
            },
        ];

        return {
            modelTransform: makeIdentityTransform(modelSpace),
            subsources,
        };
    }
}
