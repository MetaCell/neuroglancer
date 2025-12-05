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
import type { BytesToBytesCodecResolver } from "#src/datasource/zarr/codec/resolve.js";
import { registerCodec } from "#src/datasource/zarr/codec/resolve.js";
import { verifyObject } from "#src/util/json.js";

export type Configuration = object;

const zstdResolver: Omit<BytesToBytesCodecResolver<Configuration>, "name"> = {
  kind: CodecKind.bytesToBytes,
  resolve(configuration: unknown): { configuration: Configuration } {
    verifyObject(configuration);
    return { configuration: {} };
  },
};

// Register "zstd" (v2 and older v3)
registerCodec({
  name: "zstd",
  ...zstdResolver,
});

// Register "zstandard" (v3 0.6+ spec)
registerCodec({
  name: "zstandard",
  ...zstdResolver,
});
