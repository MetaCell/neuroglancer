/**
 * @license
 * Copyright 2025 Google Inc.
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

import type { AutoDetectRegistry } from "#src/kvstore/auto_detect.js";

const FORMAT_0_1_MAIN_REF = "refs/branch.main/ZZZZZZZZ.json";
const FORMAT_1_MAIN_REF = "refs/branch.main/ref.json";
// Icechunk 0.2.17 and later give the first commit this fixed id and never
// delete it.
const INITIAL_SNAPSHOT = "snapshots/1CECHNKREP0F1RSTCMT0";
const FORMAT_2_REPO_INFO = "repo";

export function registerAutoDetect(registry: AutoDetectRegistry) {
  registry.registerDirectoryFormat({
    fileNames: new Set([
      FORMAT_0_1_MAIN_REF,
      FORMAT_1_MAIN_REF,
      INITIAL_SNAPSHOT,
    ]),
    subDirectories: new Set(["refs", "snapshots"]),
    match: async ({ fileNames, subDirectories }) => {
      const isRepository =
        fileNames.has(FORMAT_0_1_MAIN_REF) ||
        fileNames.has(FORMAT_1_MAIN_REF) ||
        fileNames.has(INITIAL_SNAPSHOT) ||
        (subDirectories.has("snapshots") &&
          (subDirectories.has("refs") || fileNames.has(FORMAT_2_REPO_INFO)));
      if (!isRepository) return [];
      return [{ suffix: "icechunk:", description: "Icechunk repository" }];
    },
  });
}
