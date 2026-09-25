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

import "#src/datasource/swc/register_default.js";
import "#src/kvstore/http/register_frontend.js";

import { describe, expect, it } from "vitest";
import type { DataSource } from "#src/datasource/index.js";
import { dataSourceProviderFixture } from "#tests/fixtures/datasource_provider.js";

declare const TEST_DATA_SERVER: string;

describe("swc datasource", () => {
  const dataSourceProvider = dataSourceProviderFixture();

  async function getDataSource(folderName: string) {
    return (await dataSourceProvider()).get({
      url: `${TEST_DATA_SERVER}datasource/swc/${folderName}/|swc:`,
      globalCoordinateSpace: undefined as any,
      transform: undefined,
    });
  }

  function getSegmentProperties(dataSource: DataSource) {
    return dataSource.subsources.find((entry) => entry.id === "properties")!
      .subsource.segmentPropertyMap!.inlineProperties!;
  }

  function getLabels(dataSource: DataSource) {
    return [...getSegmentProperties(dataSource).properties[0].values].sort();
  }

  it("lists every SWC file in the folder as a segment labelled with its file name without the extension", async () => {
    const dataSource = await getDataSource("labels");
    expect(getSegmentProperties(dataSource).ids.length).toBe(2);
    expect(getLabels(dataSource)).toEqual(["basket_7", "pyramidal_A"]);
  });

  it("finds SWC files whatever the case of the .swc extension", async () => {
    expect(getLabels(await getDataSource("case"))).toEqual(["lower", "upper"]);
  });

  it("ignores files in the folder that are not SWC files", async () => {
    expect(getLabels(await getDataSource("other"))).toEqual(["cell"]);
  });

  it("rejects a folder that contains subfolders", async () => {
    await expect(getDataSource("nested")).rejects.toThrow(
      "contains subfolders",
    );
  });

  it("rejects a folder that contains no SWC files", async () => {
    await expect(getDataSource("empty")).rejects.toThrow(
      "contains no .swc files",
    );
  });
});
