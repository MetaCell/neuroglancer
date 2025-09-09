/**
 * @license
 * Copyright 2016 Google Inc.
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

import { StatusMessage } from "#src/status.js";
import { registerEventListener } from "#src/util/disposable.js";

function capitalizeFirstLetter(str: string): string {
  return str ? str.charAt(0).toUpperCase() + str.slice(1) : str;
}

/**
 * Sets the given data to the clipboard, and shows a temporary message
 * that `messageName` was copied to the clipboard or `messageName` failed to be
 * copied to the clipboard.
 * @param data The data to set to the clipboard.
 * @param messageName The name of the data to show in the temporary message.
 * If undefined or null explicitly passed, no message will be shown.
 * @param format The format of the data to set to the clipboard.
 * Defaults to `text/plain`.
 * @returns Whether the data was successfully set to the clipboard.
 */
export function setClipboard(
  data: string,
  messageName: string | null | undefined = "value", // Null/undefined means no message
  format = "text/plain",
) {
  let success = false;
  const cleanup = registerEventListener(
    document,
    "copy",
    (event: ClipboardEvent) => {
      const { clipboardData } = event;
      if (clipboardData !== null) {
        clipboardData.setData(format, data);
        success = true;
      }
      event.stopPropagation();
      event.preventDefault();
    },
    true,
  );
  try {
    document.execCommand("copy");
  } finally {
    cleanup();
  }
  if (messageName != null) {
    StatusMessage.showTemporaryMessage(
      success
        ? `${capitalizeFirstLetter(messageName)} copied to clipboard`
        : `Failed to copy ${messageName} to clipboard`,
    );
  }
  return success;
}
