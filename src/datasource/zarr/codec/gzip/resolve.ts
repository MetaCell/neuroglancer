/**
 * @license
 * Copyright 2023 Google Inc.
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

import { CodecKind } from "#src/datasource/zarr/codec/index.js";
import { registerCodec } from "#src/datasource/zarr/codec/resolve.js";
import {
  verifyInt,
  verifyObject,
  verifyObjectProperty,
} from "#src/util/json.js";

export interface Configuration {
  level: number;
}

for (const name of ["gzip", "zlib"]) {
  registerCodec({
    name,
    kind: CodecKind.bytesToBytes,
    resolve(configuration: unknown): { configuration: Configuration } {
      verifyObject(configuration);
      const level = verifyObjectProperty(configuration, "level", verifyInt);
      return { configuration: { level } };
    },
  });
}
