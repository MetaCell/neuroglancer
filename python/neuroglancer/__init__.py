# @license
# Copyright 2016 Google Inc.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from . import (  # noqa: I001
    segment_colors,
    server,
    skeleton,
)
from .default_credentials_manager import set_boss_token
from .equivalence_map import EquivalenceMap
from .local_volume import LocalVolume
from .screenshot import ScreenshotSaver
from .server import (
    is_server_running,
    set_server_bind_address,
    set_static_content_source,
    set_dev_server_content_source,
    stop,
)
from .url_state import parse_url, to_json_dump, to_url
from .viewer import UnsynchronizedViewer, Viewer
from . import viewer_config_state
from . import viewer_state

__all__ = [
    # Submodules
    "segment_colors",
    "server",
    "skeleton",
    # From default_credentials_manager
    "set_boss_token",
    # From equivalence_map
    "EquivalenceMap",
    # From local_volume
    "LocalVolume",
    # From screenshot
    "ScreenshotSaver",
    # From server
    "is_server_running",
    "set_server_bind_address",
    "set_static_content_source",
    "set_dev_server_content_source",
    "stop",
    # From url_state
    "parse_url",
    "to_json_dump",
    "to_url",
    # From viewer
    "UnsynchronizedViewer",
    "Viewer",
]

# Add exports from viewer_config_state and viewer_state
__all__ += viewer_config_state.__all__
__all__ += viewer_state.__all__


# Make viewer_config_state and viewer_state exported attrs directly accessible
def __getattr__(name):
    if name in viewer_config_state.__all__:
        return getattr(viewer_config_state, name)
    if name in viewer_state.__all__:
        return getattr(viewer_state, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
