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

// Vitest fixture that creates a temporary directory.

import fs from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { type Fixture, fixture } from "#tests/fixtures/fixture.js";

export function tempDirectoryFixture(prefix: string = ""): Fixture<string> {
  return fixture(async (stack) => {
    const tempDir = await fs.mkdtemp(
      `${os.tmpdir()}${path.sep}neuroglancer-vitest-${prefix}`,
    );
    stack.defer(async () => {
      await fs.rm(tempDir, { recursive: true, force: true });
    });
    return tempDir;
  });
}
