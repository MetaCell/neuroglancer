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
  LOADING,
} from "#src/services/events/outgoing_events.js";
import {
  type SessionUpdatePayload,
  type LoadingState,
} from "#src/services/events/outgoing_events.js";
import { setupDefaultViewer } from "#src/ui/default_viewer_setup.js";
import "#src/util/google_tag_manager.js";
import "#src/neuroglass-theme.css";

import { getCachedJson } from "#src/util/trackable.js";
import type { Viewer } from "#src/viewer.js";


// @metacell

declare const window: any;
window.neuroglancer = window.setupDefaultViewer = setupDefaultViewer;
declare let viewer: Viewer | undefined;

declare const MANUAL_LOAD: string | undefined;

if (!MANUAL_LOAD) {
  setupDefaultViewer();
  document.body.classList.add("neuroglass-theme");
}

// check viewer is ready and send message
function watchViewerLoadState() {
  const sendEvent = (loaded: boolean) => {
    const state: LoadingState = {
      loaded: loaded,
    };
    dispatchMessage(LOADING, state);
  };

  const hasLoaded = viewer?.isReady() ?? false;
  sendEvent(hasLoaded);

  if (hasLoaded) {
    return;
  }

  const pollInterval = setInterval(() => {
    if (viewer?.isReady()) {
      sendEvent(true);
      clearInterval(pollInterval);
    }
  }, 250);
}

async function dispatchUpdate(hashURLState: string) {

  if (hashURLState?.startsWith("#!")) {
    const hash = hashURLState.slice(2)

    const payload: SessionUpdatePayload = {
      url: hash,
      state: viewer?.state ? getCachedJson(viewer.state).value : null,
    };

    dispatchMessage(STATE_UPDATE, payload);

    watchViewerLoadState();
  }
}
// patch replaceState to trigger an event message with state update
history.replaceState = (() => {
  const originalReplaceState = history.replaceState;
  return (...args: any) => {
    originalReplaceState.apply(history, args);
    const hashURLState = args[2];
    dispatchUpdate(hashURLState);
  };
})();

// const originalUpdateFromUrlHash = UrlHashBinding.prototype.updateFromUrlHash;

// UrlHashBinding.prototype.updateFromUrlHash = function () {
//   if(!window.location.hash) {
//     return;
//   }
//   originalUpdateFromUrlHash.call(this);
// }

// end @metacell
