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

import { CatmaidStateValidationError } from "#src/datasource/catmaid/api.js";
import { StatusMessage } from "#src/status.js";

function formatError(error: unknown) {
  return error instanceof Error ? error.message : String(error);
}

export function isSpatialSkeletonOutdatedStateError(error: unknown) {
  return error instanceof CatmaidStateValidationError;
}

export function showSpatialSkeletonActionError(action: string, error: unknown) {
  if (isSpatialSkeletonOutdatedStateError(error)) {
    return StatusMessage.showErrorMessage(
      `Failed to ${action} due to outdated state. Refresh the page to sync.`,
    );
  }
  return StatusMessage.showTemporaryMessage(
    `Failed to ${action}: ${formatError(error)}`,
  );
}
