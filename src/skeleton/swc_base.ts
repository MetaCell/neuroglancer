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

import type { VertexAttributeInfo } from "#src/skeleton/base.js";
import { DataType } from "#src/util/data_type.js";

// https://swc-specification.readthedocs.io/en/latest/swc.html
// Type is float32 because custom type codes above 7 have no upper bound.
export const swcVertexAttributes: Map<string, VertexAttributeInfo> = new Map([
  ["radius", { dataType: DataType.FLOAT32, numComponents: 1 }],
  ["type", { dataType: DataType.FLOAT32, numComponents: 1 }],
]);
