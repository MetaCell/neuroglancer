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
import { SpatiallyIndexedSkeletonSource, SkeletonSource, MultiscaleSpatiallyIndexedSkeletonSource } from "#src/skeleton/frontend.js";
import { WithParameters } from "#src/chunk_manager/frontend.js";
import {
    DataSource,
    DataSourceProvider,
    GetDataSourceOptions,
} from "#src/datasource/index.js";
import {
    CatmaidSkeletonSourceParameters,
    CatmaidCompleteSkeletonSourceParameters,
    CatmaidDataSourceParameters,
} from "#src/datasource/catmaid/base.js";
import { mat4, vec3 } from "#src/util/geom.js";
import { CatmaidClient, type CatmaidStackInfo } from "#src/datasource/catmaid/api.js";
import { DataType } from "#src/util/data_type.js";
import { ChunkLayout } from "#src/sliceview/chunk_layout.js";
import type { SliceViewSourceOptions } from "#src/sliceview/base.js";
import { makeSliceViewChunkSpecification } from "#src/sliceview/base.js";
import {
    InlineSegmentPropertyMap,
    SegmentPropertyMap,
    normalizeInlineSegmentPropertyMap,
} from "#src/segmentation_display_state/property_map.js";
import { CatmaidToken, credentialsKey } from "#src/datasource/catmaid/api.js";
import { CredentialsProvider } from "#src/credentials_provider/index.js";
import { WithCredentialsProvider } from "#src/credentials_provider/chunk_source_frontend.js";
import { SliceViewSingleResolutionSource } from "#src/sliceview/frontend.js";
import { ChunkManager } from "#src/chunk_manager/frontend.js";
import { Borrowed } from "#src/util/disposable.js";
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

export class CatmaidSkeletonSource extends WithParameters(
    WithCredentialsProvider<CatmaidToken>()(SkeletonSource),
    CatmaidCompleteSkeletonSourceParameters
) {
    get vertexAttributes() {
        return this.parameters.metadata.vertexAttributes;
    }
    
    getChunk(objectId: bigint) {
        return super.getChunk(objectId);
    }
    static encodeOptions(options: any) {
        return super.encodeOptions(options);
    }
}

export class CatmaidMultiscaleSpatiallyIndexedSkeletonSource extends MultiscaleSpatiallyIndexedSkeletonSource {
    get rank(): number {
        return 3;
    }

    private static readonly DEBUG_SCALE_SELECTION = true;
    private sortedGridCellSizes: Array<{ x: number; y: number; z: number }>;

    constructor(
        chunkManager: Borrowed<ChunkManager>,
        private baseUrl: string,
        private projectId: number,
        private credentialsProvider: CredentialsProvider<CatmaidToken>,
        private scaleFactors: Float32Array,
        private upperBounds: Float32Array,
        gridCellSizes: Array<{ x: number; y: number; z: number }>,
        private cacheProvider?: string,
    ) {
        super(chunkManager);
        this.sortedGridCellSizes = [...gridCellSizes].sort(
            (a, b) => Math.min(b.x, b.y, b.z) - Math.min(a.x, a.y, a.z)
        );
    }

    getGridCellSizes(): Array<{ x: number; y: number; z: number }> {
        return this.sortedGridCellSizes;
    }

    getSpatialSkeletonGridSizes(): Array<{ x: number; y: number; z: number }> {
        return this.sortedGridCellSizes;
    }

    getPerspectiveSources(): SliceViewSingleResolutionSource<SpatiallyIndexedSkeletonSource>[] {
        const sources = this.getSources({} as any);
        return sources.length > 0 ? sources[0] : [];
    }

    getSliceViewPanelSources(): SliceViewSingleResolutionSource<SpatiallyIndexedSkeletonSource>[] {
        const sources = this.getSources({} as any);
        return sources.length > 0 ? sources[0] : [];
    }

    getSources(
        _options: SliceViewSourceOptions,
    ): SliceViewSingleResolutionSource<SpatiallyIndexedSkeletonSource>[][] {
        void _options;
        const sources: SliceViewSingleResolutionSource<SpatiallyIndexedSkeletonSource>[] = [];
        
        // Sorted by minimum dimension (Descending: Large/Coarse -> Small/Fine)
        const sortedGridSizes = this.sortedGridCellSizes;

        // Calculate per-dimension min size for relative scaling
        let minX = Number.MAX_VALUE;
        let minY = Number.MAX_VALUE;
        let minZ = Number.MAX_VALUE;
        for (const s of sortedGridSizes) {
            minX = Math.min(minX, s.x);
            minY = Math.min(minY, s.y);
            minZ = Math.min(minZ, s.z);
        }

        if (CatmaidMultiscaleSpatiallyIndexedSkeletonSource.DEBUG_SCALE_SELECTION) {
            console.debug("[CATMAID] Spatially indexed skeleton scales", {
                gridCellSizes: sortedGridSizes,
                minGridCellSize: { x: minX, y: minY, z: minZ },
            });
        }

        for (const [gridIndex, gridCellSize] of sortedGridSizes.entries()) {
            const chunkDataSize = Uint32Array.from([
                gridCellSize.x,
                gridCellSize.y,
                gridCellSize.z,
            ]);

            const chunkLayoutTransform = mat4.create();
            mat4.fromScaling(chunkLayoutTransform, vec3.fromValues(
                this.scaleFactors[0],
                this.scaleFactors[1],
                this.scaleFactors[2]
            ));

            const chunkLayout = new ChunkLayout(
                vec3.fromValues(chunkDataSize[0], chunkDataSize[1], chunkDataSize[2]),
                chunkLayoutTransform,
                3
            );

            const spec = {
                ...makeSliceViewChunkSpecification({
                    rank: 3,
                    chunkDataSize,
                    upperVoxelBound: this.upperBounds,
                }),
                chunkLayout,
            };

            const parameters = new CatmaidSkeletonSourceParameters();
            parameters.catmaidParameters = new CatmaidDataSourceParameters();
            parameters.catmaidParameters.url = this.baseUrl;
            parameters.catmaidParameters.projectId = this.projectId;
            parameters.catmaidParameters.cacheProvider = this.cacheProvider;
            parameters.gridIndex = gridIndex;
            parameters.metadata = {
                transform: mat4.create(),
                vertexAttributes: new Map([
                    ["segment", { dataType: DataType.FLOAT32, numComponents: 1 }],
                ]),
                sharding: undefined
            };

            const chunkSource = this.chunkManager.getChunkSource(
                CatmaidSpatiallyIndexedSkeletonSource,
                { parameters, spec, credentialsProvider: this.credentialsProvider }
            );

            // Calculate relative scale based on grid size vs minimum grid size
            const relativeScaleX = gridCellSize.x / minX;
            const relativeScaleY = gridCellSize.y / minY;
            const relativeScaleZ = gridCellSize.z / minZ;
            
            const chunkToMultiscaleTransform = mat4.create();
            mat4.fromScaling(
                chunkToMultiscaleTransform,
                vec3.fromValues(relativeScaleX, relativeScaleY, relativeScaleZ)
            );
            sources.push({
                chunkSource,
                chunkToMultiscaleTransform,
            });

            if (CatmaidMultiscaleSpatiallyIndexedSkeletonSource.DEBUG_SCALE_SELECTION) {
                console.debug("[CATMAID] Spatially indexed skeleton scale", {
                    gridCellSize,
                    relativeScale: { x: relativeScaleX, y: relativeScaleY, z: relativeScaleZ },
                });
            }
        }

        return [sources];
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

        // Fetch metadata using memoized client methods to avoid duplicate requests
        const [stackInfo, skeletonIds] = await Promise.all([
            options.registry.chunkManager.memoize.getAsync(
                { type: "catmaid:metadata", baseUrl, projectId },
                options,
                () => (client as any).getMetadataInfo() as Promise<CatmaidStackInfo | null>,
            ),
            options.registry.chunkManager.memoize.getAsync(
                { type: "catmaid:skeletons", baseUrl, projectId },
                options,
                () => client.listSkeletons(),
            ),
        ]);

        if (!stackInfo) {
            throw new Error("Failed to fetch CATMAID stack metadata");
        }

        // Extract dimensions from stack info
        const { dimension, translation, resolution } = stackInfo;
        const offX = translation?.x ?? 0;
        const offY = translation?.y ?? 0;
        const offZ = translation?.z ?? 0;

        const dimensions = {
            min: { x: offX, y: offY, z: offZ },
            max: { x: offX + dimension.x, y: offY + dimension.y, z: offZ + dimension.z },
        };

        // Extract grid cell sizes from metadata
        const gridCellSizes: Array<{ x: number; y: number; z: number }> = [];
        if (stackInfo.metadata?.cache_configurations) {
            for (const config of stackInfo.metadata.cache_configurations) {
                if (config.cache_type === "grid") {
                    gridCellSizes.push({
                        x: config.cell_width,
                        y: config.cell_height,
                        z: config.cell_depth,
                    });
                }
            }
        }
        
        // If no grid configs found, use default
        if (gridCellSizes.length === 0) {
            const DEFAULT_CACHE_GRID_CELL_WIDTH = 25000;
            const DEFAULT_CACHE_GRID_CELL_HEIGHT = 25000;
            const DEFAULT_CACHE_GRID_CELL_DEPTH = 40;
            gridCellSizes.push({
                x: DEFAULT_CACHE_GRID_CELL_WIDTH,
                y: DEFAULT_CACHE_GRID_CELL_HEIGHT,
                z: DEFAULT_CACHE_GRID_CELL_DEPTH,
            });
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

        const rank = 3;

        const upperVoxelBound = new Float32Array(rank);
        for (let i = 0; i < rank; ++i) {
            upperVoxelBound[i] = upperBounds[i];
        }

        // Log information about available chunk sizes
        console.info('CATMAID Multiscale Skeleton Source:', {
            chunkSizes: gridCellSizes.map(size => ({ x: size.x, y: size.y, z: size.z })),
            resolution: { x: resolution.x, y: resolution.y, z: resolution.z, unit: 'nm' },
            totalDimension: dimensions,
            totalNumberOfSkeletons: skeletonIds.length
        });

        // Extract cache provider from metadata
        const cacheProvider = stackInfo.metadata?.cache_provider;

        // Create multiscale skeleton source to get individual sources
        const multiscaleSource = new CatmaidMultiscaleSpatiallyIndexedSkeletonSource(
            options.registry.chunkManager,
            baseUrl,
            projectId,
            credentialsProvider,
            new Float32Array(scaleFactors),
            upperVoxelBound,
            gridCellSizes,
            cacheProvider
        );
        // Create complete skeleton source (non-chunked)
        const completeSkeletonParameters = new CatmaidCompleteSkeletonSourceParameters();
        completeSkeletonParameters.catmaidParameters = new CatmaidDataSourceParameters();
        completeSkeletonParameters.catmaidParameters.url = baseUrl;
        completeSkeletonParameters.catmaidParameters.projectId = projectId;
        completeSkeletonParameters.url = providerUrl;
        completeSkeletonParameters.metadata = {
            transform: mat4.create(),
            vertexAttributes: new Map([
                ["segment", { dataType: DataType.FLOAT32, numComponents: 1 }],
            ]),
            sharding: undefined
        };

        const completeSkeletonSource = options.registry.chunkManager.getChunkSource(
            CatmaidSkeletonSource,
            { parameters: completeSkeletonParameters, credentialsProvider }
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
                id: "skeletons-chunked",
                default: true,
                subsource: { mesh: multiscaleSource },
            },
            {
                id: "skeletons",
                default: false,
                subsource: { mesh: completeSkeletonSource },
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
