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
import { SkeletonSource } from "#src/skeleton/frontend.js";
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
import { CatmaidClient } from "#src/datasource/catmaid/api.js";
import {
    SegmentPropertyMap,
    InlineSegmentProperty,
} from "#src/segmentation_display_state/property_map.js";
import type { DataSubsourceEntry } from "#src/datasource/index.js";
import { mat4 } from "#src/util/geom.js";

export class CatmaidSkeletonSource extends WithParameters(
    SkeletonSource,
    CatmaidSkeletonSourceParameters
) { }

export class CatmaidDataSourceProvider implements DataSourceProvider {
    get scheme() {
        return "catmaid";
    }

    get description() {
        return "CATMAID";
    }

    async get(options: GetDataSourceOptions): Promise<DataSource> {
        const { providerUrl } = options;

        let baseUrl: string;
        let projectId: number;

        const lastSlash = providerUrl.lastIndexOf('/');
        if (lastSlash === -1) {
            // Maybe just project ID?
            // Assume user knows what they are doing if they provide just a number?
            // But we need a base URL.
            throw new Error("Invalid CATMAID URL. Expected format: <base_url>/<project_id>");
        }

        const projectIdStr = providerUrl.substring(lastSlash + 1);
        projectId = parseInt(projectIdStr);
        if (isNaN(projectId)) {
            throw new Error(`Invalid project ID: ${projectIdStr}`);
        }

        baseUrl = providerUrl.substring(0, lastSlash);

        // If baseUrl doesn't start with http, prepend https://
        if (!baseUrl.startsWith("http")) {
            baseUrl = "https://" + baseUrl;
        }

        const parameters = new CatmaidSkeletonSourceParameters();
        parameters.catmaidParameters = new CatmaidDataSourceParameters();
        parameters.catmaidParameters.url = baseUrl;
        parameters.catmaidParameters.projectId = projectId;
        parameters.url = providerUrl;
        parameters.metadata = {
            transform: mat4.create(),
            vertexAttributes: new Map(),
            sharding: undefined
        };

        const source = options.registry.chunkManager.getChunkSource(
            CatmaidSkeletonSource,
            { parameters }
        );

        const modelSpace = makeCoordinateSpace({
            names: ["x", "y", "z"],
            units: ["nm", "nm", "nm"],
            scales: Float64Array.of(1, 1, 1),
        });

        // Fetch skeleton list to populate segment property map.
        let subsources: DataSubsourceEntry[] = [
            {
                id: "skeletons",
                default: true,
                subsource: { mesh: source },
            },
        ];

        try {
            const client = new CatmaidClient(
                parameters.catmaidParameters.url,
                parameters.catmaidParameters.projectId,
                parameters.catmaidParameters.token,
            );
            const skeletonIds = await client.listSkeletons();
            if (Array.isArray(skeletonIds) && skeletonIds.length > 0) {
                const ids = new BigUint64Array(skeletonIds.length);
                for (let i = 0; i < skeletonIds.length; ++i) {
                    ids[i] = BigInt(skeletonIds[i]);
                }
                const labelProperty: InlineSegmentProperty = {
                    id: "label",
                    type: "label",
                    values: skeletonIds.map((id) => `Skeleton ${id}`),
                };
                const segmentPropertyMap = new SegmentPropertyMap({
                    inlineProperties: { ids, properties: [labelProperty] },
                });
                subsources.push({
                    id: "properties",
                    default: true,
                    subsource: { segmentPropertyMap },
                });
            }
        } catch (e) {
            // Non-fatal: just skip properties if request fails.
            console.warn("Failed to load CATMAID skeleton list", e);
        }

        return {
            modelTransform: makeIdentityTransform(modelSpace),
            subsources,
        };
    }
}
