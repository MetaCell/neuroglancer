# @license
# Copyright 2023 Google Inc.
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
"""Tests the n5 datasource."""

import pathlib

import neuroglancer
import numpy as np
import pytest


@pytest.mark.parametrize(
    "spec",
    [
        {
            "driver": "neuroglancer_precomputed",
            "scale_metadata": {"encoding": "png", "chunk_size": [8, 9, 1]},
            "dtype": "uint8",
            "schema": {"domain": {"shape": [10, 20, 5, num_channels]}},
        }
        for num_channels in [1, 2, 3, 4]
    ]
    + [
        # Currently TensorStore does not support uint16 with more than one channel.
        {
            "driver": "neuroglancer_precomputed",
            "scale_metadata": {"encoding": "png"},
            "dtype": "uint16",
            "schema": {"domain": {"shape": [10, 20, 5, 1]}},
        }
    ]
    + [
        # Due to a tensorstore bug (https://github.com/google/neuroglancer/issues/677)
        # the block shape must be square.
        {
            "driver": "neuroglancer_precomputed",
            "scale_metadata": {"encoding": "jpeg", "chunk_size": [10, 10, 1]},
            "dtype": "uint8",
            "schema": {"domain": {"shape": [10, 20, 1, num_channels]}},
        }
        for num_channels in [1, 3]
    ],
    ids=str,
)
def test_precomputed(tempdir_server: tuple[pathlib.Path, str], webdriver, spec):
    import tensorstore as ts

    tmp_path, server_url = tempdir_server

    full_spec = {
        "kvstore": {
            "driver": "file",
            "path": str(tmp_path),
        }
    }
    full_spec.update(spec)

    store = ts.open(full_spec, create=True).result()

    a = np.arange(np.prod(store.shape), dtype=store.dtype.numpy_dtype).reshape(
        store.shape
    )

    store[...] = a

    with webdriver.viewer.txn() as s:
        s.layers.append(
            name="a",
            layer=neuroglancer.ImageLayer(source=f"precomputed://{server_url}"),
        )

    if store.shape[-1] == 1:
        # Neuroglancer elides the channel dimension if there is only 1 channel
        store = store[..., 0]

    vol = webdriver.viewer.volume("a").result()
    b = vol.read().result()

    if spec["scale_metadata"]["encoding"] == "jpeg":
        np.testing.assert_allclose(store.read().result(), b, atol=4, rtol=0)
    else:
        np.testing.assert_equal(store.read().result(), b)


def test_reads_every_resolution_that_volume_info_lists(
    tempdir_server: tuple[pathlib.Path, str], webdriver
):
    import tensorstore as ts

    tmp_path, server_url = tempdir_server
    rng = np.random.default_rng(0)
    full = rng.integers(0, 256, size=(41, 30, 20, 1), dtype=np.uint8)
    half = rng.integers(0, 256, size=(20, 15, 10, 1), dtype=np.uint8)
    for resolution, data in [([1, 1, 1], full), ([2, 2, 2], half)]:
        store = ts.open(
            {
                "driver": "neuroglancer_precomputed",
                "kvstore": {"driver": "file", "path": str(tmp_path)},
                "multiscale_metadata": {"data_type": "uint8", "num_channels": 1},
                "scale_metadata": {
                    "resolution": resolution,
                    "encoding": "raw",
                    "chunk_size": [8, 8, 8],
                },
                "schema": {"domain": {"shape": list(data.shape)}},
            },
            create=True,
            open=True,
        ).result()
        store[...] = data

    with webdriver.viewer.txn() as s:
        s.layers.append(
            name="a",
            layer=neuroglancer.ImageLayer(source=f"precomputed://{server_url}"),
        )

    info = webdriver.viewer.volume_info("a").result()
    assert [list(scales) for scales in info.available_scales] == [
        [1e-9, 1e-9, 1e-9],
        [2e-9, 2e-9, 2e-9],
    ]
    for scales, data in zip(info.available_scales, [full, half]):
        dimensions = neuroglancer.CoordinateSpace(
            names=info.dimensions.names,
            units=info.dimensions.units,
            scales=scales,
        )
        volume = webdriver.viewer.volume("a", dimensions=dimensions).result()
        np.testing.assert_equal(volume.read().result(), data[..., 0])


def test_a_volume_with_a_cache_reads_each_chunk_from_the_browser_once(
    tempdir_server: tuple[pathlib.Path, str], webdriver, monkeypatch
):
    import tensorstore as ts

    tmp_path, server_url = tempdir_server
    data = np.random.default_rng(0).integers(0, 256, size=(8, 8, 8, 1), dtype=np.uint8)
    store = ts.open(
        {
            "driver": "neuroglancer_precomputed",
            "kvstore": {"driver": "file", "path": str(tmp_path)},
            "multiscale_metadata": {"data_type": "uint8", "num_channels": 1},
            "scale_metadata": {
                "resolution": [1, 1, 1],
                "encoding": "raw",
                "chunk_size": [8, 8, 8],
            },
            "schema": {"domain": {"shape": [8, 8, 8, 1]}},
        },
        create=True,
    ).result()
    store[...] = data

    with webdriver.viewer.txn() as s:
        s.layers.append(
            name="a",
            layer=neuroglancer.ImageLayer(source=f"precomputed://{server_url}"),
        )

    chunk_requests = []
    request_chunk = webdriver.viewer._volume_chunk

    def count_chunk_request(layer, info, chunk_grid_position):
        chunk_requests.append(list(chunk_grid_position))
        return request_chunk(layer, info, chunk_grid_position)

    monkeypatch.setattr(webdriver.viewer, "_volume_chunk", count_chunk_request)
    volume = webdriver.viewer.volume(
        "a", context=ts.Context({"cache_pool": {"total_bytes_limit": 10_000_000}})
    ).result()
    for z in range(8):
        np.testing.assert_equal(volume[:, :, z].read().result(), data[:, :, z, 0])
    assert chunk_requests == [[0, 0, 0]]
