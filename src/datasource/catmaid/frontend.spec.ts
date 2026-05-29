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

import { describe, expect, it, vi } from "vitest";

import type {
  DataSource,
  GetDataSourceOptions,
} from "#src/datasource/index.js";
import type { SpatiallyIndexedSkeletonMetadata } from "#src/skeleton/api.js";

if (!("WebGL2RenderingContext" in globalThis)) {
  Object.defineProperty(globalThis, "WebGL2RenderingContext", {
    value: new Proxy(class WebGL2RenderingContext {} as any, {
      get(target, property, receiver) {
        if (Reflect.has(target, property)) {
          return Reflect.get(target, property, receiver);
        }
        return 0;
      },
    }),
    configurable: true,
  });
}

const { CatmaidDataSourceProvider } = await import(
  "#src/datasource/catmaid/frontend.js"
);

function makeGetOptions() {
  const getChunkSource = vi.fn();
  const getCredentialsProvider = vi.fn(() => ({}));
  const spatialIndexMetadata = {
    lowerBounds: [0, 0, 0],
    upperBounds: [100, 200, 300],
    readonly: false,
    spatial: [
      {
        chunkSize: [10, 20, 30],
        gridShape: [10, 10, 10],
        limit: 0,
      },
    ],
  } satisfies SpatiallyIndexedSkeletonMetadata;
  const getAsync = vi.fn(
    async (
      key: { type?: string },
      _options: unknown,
      _getter: unknown,
    ): Promise<unknown> => {
      switch (key.type) {
        case "catmaid:spatial-index-metadata":
          return spatialIndexMetadata;
        case "catmaid:cache-provider":
          return "cached_msgpack_grid";
        case "catmaid:skeletons":
          return [1, 2, 3];
        default:
          throw new Error(`Unexpected memoized CATMAID lookup: ${key.type}`);
      }
    },
  );

  return {
    getChunkSource,
    options: {
      providerUrl: "catmaid://catmaid.example/7",
      providerScheme: "catmaid",
      url: "catmaid://catmaid.example/7",
      transform: undefined,
      globalCoordinateSpace: undefined,
      signal: new AbortController().signal,
      registry: {
        credentialsManager: { getCredentialsProvider },
        chunkManager: {
          memoize: { getAsync },
          getChunkSource,
        },
      },
    } as unknown as GetDataSourceOptions,
  };
}

describe("CatmaidDataSourceProvider", () => {
  it("exposes only the spatial skeleton subsource for CATMAID skeletons", async () => {
    const { getChunkSource, options } = makeGetOptions();
    const result = await new CatmaidDataSourceProvider().get(options);
    if ("targetUrl" in result) {
      throw new Error("Expected CATMAID provider to return a data source.");
    }

    const dataSource = result as DataSource;
    expect(dataSource.subsources.map((subsource) => subsource.id)).toEqual([
      "skeletons-chunked",
      "properties",
      "bounds",
    ]);
    expect(getChunkSource).not.toHaveBeenCalled();
  });
});
