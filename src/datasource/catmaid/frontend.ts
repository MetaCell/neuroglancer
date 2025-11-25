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
import {
    AnnotationGeometryChunkSource,
    MultiscaleAnnotationSource,
} from "#src/annotation/frontend_source.js";
import { WithParameters } from "#src/chunk_manager/frontend.js";
import {
    DataSource,
    DataSourceProvider,
    GetDataSourceOptions,
} from "#src/datasource/index.js";
import {
    CatmaidAnnotationSourceParameters,
    CatmaidAnnotationGeometryChunkSourceParameters,
    CatmaidDataSourceParameters,
} from "#src/datasource/catmaid/base.js";
import type { SliceViewSingleResolutionSource } from "#src/sliceview/frontend.js";
import { makeSliceViewChunkSpecification } from "#src/sliceview/base.js";
import { mat4 } from "#src/util/geom.js";

export class CatmaidAnnotationGeometryChunkSource extends WithParameters(
    AnnotationGeometryChunkSource,
    CatmaidAnnotationGeometryChunkSourceParameters
) { }

export class CatmaidAnnotationSource extends WithParameters(
    MultiscaleAnnotationSource,
    CatmaidAnnotationSourceParameters
) {
    initializeCounterpart(rpc: any, options: any) {
        console.log('CATMAID Frontend: initializeCounterpart called with options:', options);
        super.initializeCounterpart(rpc, options);
        console.log('CATMAID Frontend: initializeCounterpart completed, options now:', options);
    }

    getSources(): SliceViewSingleResolutionSource<AnnotationGeometryChunkSource>[][] {
        // Create a single resolution level with one chunk source
        const spec = (this.parameters as any).spec;
        if (!spec) {
            throw new Error('CATMAID annotation source: spec is required');
        }
        
        const chunkSourceParameters: CatmaidAnnotationGeometryChunkSourceParameters = {
            catmaidParameters: this.parameters.catmaidParameters,
            spec,
        };
        
        const chunkSource = this.chunkManager.getChunkSource(
            CatmaidAnnotationGeometryChunkSource,
            {
                parent: this,
                parameters: chunkSourceParameters,
                spec,
            },
        );

        // Return a 2D array: [scales][alternatives]
        // We have one scale with one alternative
        return [[
            {
                chunkSource,
                chunkToMultiscaleTransform: spec.chunkToMultiscaleTransform,
            },
        ]];
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
        // Parse URL: catmaid://<server>/<project_id>
        const parts = providerUrl.split("/");
        const projectId = parseInt(parts[parts.length - 1]);
        const baseUrl = "https://" + parts.slice(0, parts.length - 1).join("/");

        const parameters = new CatmaidAnnotationSourceParameters();
        parameters.catmaidParameters = new CatmaidDataSourceParameters();
        parameters.catmaidParameters.url = baseUrl;
        parameters.catmaidParameters.projectId = projectId;
        parameters.rank = 3;
        parameters.properties = []; // Default properties

        // Create a simple chunk specification for spatial indexing
        // Use 1000nm chunks (1 micron) which is reasonable for neuron tracing
        const chunkDataSize = new Uint32Array([1000, 1000, 1000]);
        const upperVoxelBound = new Float32Array([1000000000, 1000000000, 1000000000]); // 1 billion nm = 1 meter cube
        const lowerVoxelBound = new Float32Array([0, 0, 0]);
        
        // Use the helper function to create a proper spec
        const baseSpec = makeSliceViewChunkSpecification({
            rank: 3,
            chunkDataSize,
            upperVoxelBound,
            lowerVoxelBound,
        });
        
        // Add the additional properties needed for AnnotationGeometryChunkSpecification
        const spec = {
            ...baseSpec,
            chunkToMultiscaleTransform: mat4.create(), // Identity transform
            limit: 0, // No limit on annotation density
        };
        
        // Store spec in parameters for getSources to access it
        (parameters as any).spec = spec;

        const source = options.registry.chunkManager.getChunkSource(
            CatmaidAnnotationSource,
            {
                parameters,
                rank: 3,
                relationships: [],
                properties: [],
            } as any
        );

        const modelSpace = makeCoordinateSpace({
            names: ["x", "y", "z"],
            units: ["nm", "nm", "nm"],
            scales: Float64Array.of(1, 1, 1),
        });

        return {
            modelTransform: makeIdentityTransform(modelSpace),
            subsources: [
                {
                    id: "annotations",
                    default: true,
                    subsource: { annotation: source },
                },
            ],
        };
    }
}
