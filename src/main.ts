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
/**
 * @file Main entry point for default neuroglancer viewer.
 */
import {
  dispatchMessage,
  STATE_UPDATE,
} from "#src/services/events/outgoing_events.js";
import { type SessionUpdatePayload } from "#src/services/events/outgoing_events.js";
import { setupDefaultViewer } from "#src/ui/default_viewer_setup.js";
import "#src/util/google_tag_manager.js";
import { encodeFragment } from "#src/ui/url_hash_binding.js";
import "#src/metacell-theme.css";
declare const window: any;

window.neuroglancer = setupDefaultViewer;
setupDefaultViewer();

document.body.classList.add("metacell-theme");

// @metacell

// patch replaceState to trigger an event message with state update
history.replaceState = (() => {
  const originalReplaceState = history.replaceState;
  return (...args: any) => {
    originalReplaceState.apply(history, args);

    const hashURLState = args[2];
    if (hashURLState?.startsWith("#!")) {
      const s = decodeURIComponent(hashURLState.slice(2));
      let state = {};
      try {
        state = JSON.parse(s);
      } catch (error) {
        console.error("error parsing url encoded state:", error);
        return;
      }

      const payload: SessionUpdatePayload = {
        url: encodeFragment(JSON.stringify(state)),
        state: state,
      };

      dispatchMessage(STATE_UPDATE, payload);
    }
  };
})();

// end @metacell
