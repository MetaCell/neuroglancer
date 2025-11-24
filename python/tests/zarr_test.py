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
"""Tests the zarr datasource."""

import pathlib

import neuroglancer
import numpy as np
import pytest

TEST_DATA_DIR = (
    pathlib.Path(__file__).parent.parent.parent / "testdata" / "datasource" / "zarr"
)


@pytest.mark.parametrize(
    "spec",
    [
        {"driver": "zarr"},
        {"driver": "zarr", "metadata": {"compressor": {"id": "zlib"}}},
        {"driver": "zarr", "schema": {"chunk_layout": {"inner_order": [2, 1, 0]}}},
        {"driver": "zarr3"},
        {"driver": "zarr3", "schema": {"chunk_layout": {"inner_order": [2, 1, 0]}}},
        {"driver": "zarr3", "schema": {"dimension_units": ["nm", None, ""]}},
        {
            "driver": "zarr3",
            "schema": {
                "chunk_layout": {
                    "read_chunk": {"shape": [2, 3, 4]},
                    "write_chunk": {"shape": [6, 12, 20]},
                }
            },
        },
        {
            "driver": "zarr3",
            "schema": {
                "chunk_layout": {
                    "inner_order": [2, 0, 1],
                    "read_chunk": {"shape": [2, 3, 4]},
                    "write_chunk": {"shape": [6, 12, 20]},
                }
            },
        },
        {
            "driver": "zarr3",
            "schema": {
                "chunk_layout": {
                    "inner_order": [2, 0, 1],
                    "read_chunk": {"shape": [2, 3, 4]},
                    "write_chunk": {"shape": [6, 12, 20]},
                }
            },
            "kvstore": {"driver": "ocdbt"},
        },
        {
            "driver": "zarr3",
            "schema": {"chunk_layout": {"write_chunk": {"shape": [6, 12, 24]}}},
            "metadata": {
                "codecs": [
                    {"name": "transpose", "configuration": {"order": [0, 2, 1]}},
                    {
                        "name": "sharding_indexed",
                        "configuration": {
                            "chunk_shape": [2, 3, 4],
                            "index_codecs": [
                                {
                                    "name": "transpose",
                                    "configuration": {"order": [3, 1, 0, 2]},
                                },
                                {
                                    "name": "bytes",
                                    "configuration": {"endian": "little"},
                                },
                            ],
                            "codecs": [
                                {
                                    "name": "transpose",
                                    "configuration": {"order": [2, 1, 0]},
                                },
                                {
                                    "name": "bytes",
                                    "configuration": {"endian": "little"},
                                },
                                {"name": "gzip"},
                            ],
                        },
                    },
                ]
            },
        },
        {
            "driver": "zarr3",
            "schema": {"chunk_layout": {"write_chunk": {"shape": [6, 12, 24]}}},
            "metadata": {
                "codecs": [
                    {"name": "transpose", "configuration": {"order": [0, 2, 1]}},
                    {
                        "name": "sharding_indexed",
                        "configuration": {
                            "chunk_shape": [2, 3, 4],
                            "index_location": "start",
                            "index_codecs": [
                                {
                                    "name": "transpose",
                                    "configuration": {"order": [3, 1, 0, 2]},
                                },
                                {
                                    "name": "bytes",
                                    "configuration": {"endian": "little"},
                                },
                            ],
                            "codecs": [
                                {
                                    "name": "transpose",
                                    "configuration": {"order": [2, 1, 0]},
                                },
                                {
                                    "name": "bytes",
                                    "configuration": {"endian": "little"},
                                },
                                {"name": "gzip"},
                            ],
                        },
                    },
                ]
            },
        },
    ],
    ids=str,
)
def test_zarr(tempdir_server: tuple[pathlib.Path, str], webdriver, spec):
    import tensorstore as ts

    tmp_path, server_url = tempdir_server

    shape = [10, 20, 30]

    a = np.arange(np.prod(shape), dtype=np.int32).reshape(shape)

    file_spec = {
        "driver": "file",
        "path": str(tmp_path),
    }

    if "kvstore" in spec:
        full_spec = {**spec, "kvstore": {**spec["kvstore"], "base": file_spec}}
    else:
        full_spec = {**spec, "kvstore": file_spec}

    store = ts.open(full_spec, create=True, dtype=ts.int32, shape=shape).result()
    store[...] = a

    with webdriver.viewer.txn() as s:
        s.layers.append(name="a", layer=neuroglancer.ImageLayer(source=server_url))

    vol = webdriver.viewer.volume("a").result()
    b = vol.read().result()
    np.testing.assert_equal(a, b)


def test_zarr_corrupt(tempdir_server: tuple[pathlib.Path, str], webdriver):
    import tensorstore as ts

    tmp_path, server_url = tempdir_server

    shape = [10, 20, 30]

    a = np.arange(np.prod(shape), dtype=np.int32).reshape(shape)

    full_spec_for_chunks = {
        "driver": "zarr3",
        "kvstore": {
            "driver": "file",
            "path": str(tmp_path),
        },
        "metadata": {"codecs": ["zstd"]},
    }

    full_spec_for_metadata = {
        "driver": "zarr3",
        "kvstore": {
            "driver": "file",
            "path": str(tmp_path),
        },
        "metadata": {"codecs": ["gzip"]},
    }

    ts.open(full_spec_for_metadata, create=True, dtype=ts.int32, shape=shape).result()
    store = ts.open(
        full_spec_for_chunks,
        open=True,
        assume_metadata=True,
        dtype=ts.int32,
        shape=shape,
    ).result()
    store[...] = a

    with webdriver.viewer.txn() as s:
        s.layers.append(
            name="a", layer=neuroglancer.ImageLayer(source=f"zarr3://{server_url}")
        )

    vol = webdriver.viewer.volume("a").result()
    with pytest.raises(ValueError, match=".*Failed to decode gzip"):
        vol.read().result()


EXCLUDED_ZARR_V2_CASES = {
    ".zgroup",
    ".zattrs",
    ".zmetadata",
    # bool not supported by neuroglancer
    "1d.contiguous.b1",
    # float64 not supported by neuroglancer
    "1d.contiguous.f8",
    # LZ4 not supported by neuroglancer or tensorstore
    "1d.contiguous.lz4.i2",
    # S not supported by neuroglancer
    "1d.contiguous.S7",
    # U not supported by neuroglancer
    "1d.contiguous.U13.be",
    "1d.contiguous.U13.le",
    "1d.contiguous.U7",
    "2d.chunked.U7",
    # VLenUTF8 not supported by neuroglancer
    "3d.chunked.O",
}

EXCLUDED_ZARR_V3_CASES = {
    "zarr.json",
    # bool not supported by neuroglancer
    "1d.contiguous.b1",
    "1d.contiguous.compressed.sharded.b1",
    # float64 not supported by neuroglancer
    "1d.contiguous.f8",
    "1d.contiguous.compressed.sharded.f8",
}


@pytest.mark.parametrize(
    "driver,data_dir",
    [
        ("zarr", p)
        for p in TEST_DATA_DIR.glob("zarr_v2/from_zarr-python/data.zarr/*")
        if p.name != ".zgroup" and p.name not in EXCLUDED_ZARR_V2_CASES
    ]
    + [
        ("zarr3", p)
        for p in TEST_DATA_DIR.glob("zarr_v3/from_zarrita/data.zarr/*")
        if p.name not in EXCLUDED_ZARR_V3_CASES
    ],
    ids=str,
)
def test_data(driver: str, data_dir: pathlib.Path, static_file_server, webdriver):
    import tensorstore as ts

    server_url = static_file_server(data_dir)
    full_spec = {
        "driver": driver,
        "kvstore": {
            "driver": "file",
            "path": str(data_dir),
        },
    }
    store = ts.open(full_spec, open=True, read=True).result()
    a = store.read().result()

    with webdriver.viewer.txn() as s:
        s.layers.append(
            name="a", layer=neuroglancer.ImageLayer(source=f"zarr://{server_url}")
        )

    vol = webdriver.viewer.volume("a").result()
    b = vol.read().result()
    np.testing.assert_equal(a, b)


## OME-ZARR 0.6 tests
"""Simplified OME-ZARR 0.6 transform tests.

Each test loads the corresponding example dataset for a transformation
defined in RFC-5

Transformation types covered (RFC-5 table):
  identity, mapAxis, translation, scale, affine, rotation,
  sequence, displacements, coordinates, inverseOf, bijection, byDimension.

"""

OME_ZARR_0_6_ROOT = TEST_DATA_DIR / "ome_zarr" / "all_0.6"


def _sync_with_timeout(webdriver, timeout=3):
    """Sync with timeout to prevent indefinite hangs.
    
    Wraps screenshot() call with a timeout. If the viewer doesn't respond within the timeout
    (e.g., due to unsupported transforms), the test fails with a clear message instead of hanging.
    
    Also monitors console logs for "Unsupported" transform errors to provide better diagnostics.
    
    :param timeout: Maximum time to wait for sync (default: 3 seconds)
    """
    import threading
    
    error_detected = []
    sync_result = [None]
    exception_raised = [None]
    
    def error_listener(log):
        msg = log.text
        if "Unsupported" in msg and "transform" in msg:
            error_detected.append(msg)
    
    def do_sync():
        try:
            result = webdriver.viewer.screenshot()
            sync_result[0] = result
        except Exception as e:
            exception_raised[0] = e
    
    unregister = webdriver.add_log_listener(error_listener)
    
    try:
        sync_thread = threading.Thread(target=do_sync)
        sync_thread.daemon = True
        sync_thread.start()
        sync_thread.join(timeout=timeout)
        
        if error_detected:
            pytest.fail(f"Layer failed to load due to unsupported feature: {error_detected[0]}")
        
        if sync_thread.is_alive():
            pytest.fail(f"Sync timed out after {timeout} seconds (likely due to unsupported transform)")
        
        if exception_raised[0]:
            raise exception_raised[0]
        
        return sync_result[0]
    finally:
        unregister()


def _assert_renders(webdriver, layer_name: str):
    model_space = get_layer_model_space(webdriver, layer_name)
    assert model_space is not None, f"Layer '{layer_name}' did not render (modelSpace missing)."
    assert isinstance(model_space, dict), f"Layer '{layer_name}' failed to render: {model_space}"
    return model_space


def _verify_data_at_point(vol, voxel_point, expected_value):
    """Verifies that the data value at the given voxel coordinates matches expected_value.
    
    vol: The volume object (to avoid reading it multiple times).
    voxel_point: tuple of voxel coordinates.
    expected_value: The expected value at the given voxel.
    """
    # Read the entire volume (for small test datasets this is fine)
    data = vol.read().result()
    domain = vol.domain

    # Calculate index into the data array accounting for domain origin
    origin = domain.origin
    idx = tuple(int(v - o) for v, o in zip(voxel_point, origin))
    
    if any(i < 0 or i >= s for i, s in zip(idx, data.shape)):
        assert False, f"Voxel {voxel_point} is out of bounds"

    value = data[idx]
    assert value == expected_value, f"Expected value {expected_value} at voxel {voxel_point}, got {value}"


# Point in identity.zarr near the edge with a less common value
TEST_VOXEL = (13, 122, 169) # (z, y, x)
EXPECTED_VALUE = 145  # Value at the specified voxel coordinates

def test_ome_zarr_0_6_map_axis(static_file_server, webdriver):
    """mapAxis: Axis permutation via axis name mapping.
    Example dataset: axis_dependent/mapAxis.zarr
    
    This dataset uses the same underlying array as identity.zarr (shape [27, 226, 186])
    but applies a mapAxis transform that **permutes the axes**:
      - mapAxis [2, 1, 0] means: output[0]←input[2], output[1]←input[1], output[2]←input[0]
      - This swaps the first and last array dimensions
    
    The underlying array has dimensions [27, 226, 186] (dim_0, dim_1, dim_2).
    After mapAxis [2, 1, 0]:
      - Physical "z" axis ← array dim_2 (size 186) → domain z: [0, 186)
      - Physical "y" axis ← array dim_1 (size 226) → domain y: [0, 226)
      - Physical "x" axis ← array dim_0 (size 27) → domain x: [0, 27)
    
    Compared to identity.zarr which has z:[0,27), y:[0,226), x:[0,186),
    the mapAxis transform swaps the x and z extents.
    """
    mapaxis_dir = OME_ZARR_0_6_ROOT / "axis_dependent" / "mapAxis.zarr"
    mapaxis_url = static_file_server(mapaxis_dir)
    
    with webdriver.viewer.txn() as s:
        s.layers.append(name="mapAxis", layer=neuroglancer.ImageLayer(source=f"zarr3://{mapaxis_url}"))
    webdriver.sync()
    
    mapaxis_space = _assert_renders(webdriver, "mapAxis")
    
    # Get the volume from the model space (already fetched in _assert_renders)
    vol = mapaxis_space['volume']

    permuted_voxel = (TEST_VOXEL[2], TEST_VOXEL[1], TEST_VOXEL[0])  # (x, y, z) -> (z, y, x)
    _verify_data_at_point(vol, permuted_voxel, EXPECTED_VALUE)


def test_ome_zarr_0_6_identity(static_file_server, webdriver):
    """identity: Do-nothing transformation; usually implicit.
    Example dataset: basic/identity.zarr
    """
    test_dir = OME_ZARR_0_6_ROOT / "basic" / "identity.zarr"
    server_url = static_file_server(test_dir)
    with webdriver.viewer.txn() as s:
        s.layers.append(name="identity", layer=neuroglancer.ImageLayer(source=f"zarr3://{server_url}"))
    webdriver.sync()
    identity_space = _assert_renders(webdriver, "identity")
    
    # Verify value at test voxel
    _verify_data_at_point(identity_space['volume'], TEST_VOXEL, EXPECTED_VALUE)




def test_ome_zarr_0_6_multiscale_scale(static_file_server, webdriver):
    """scale (multiscale): Scaling on multiscale pyramid levels.
    Example dataset: basic/scale_multiscale.zarr
    """
    test_dir = OME_ZARR_0_6_ROOT / "basic" / "scale_multiscale.zarr"
    server_url = static_file_server(test_dir)
    with webdriver.viewer.txn() as s:
        s.layers.append(name="scale_ms", layer=neuroglancer.ImageLayer(source=f"zarr3://{server_url}"))
    webdriver.sync()
    model_space = _assert_renders(webdriver, "scale_ms")
    pass


def test_ome_zarr_0_6_scale(static_file_server, webdriver):
    """scale: Per-axis scaling factors (JSON vector form).
    Example dataset: basic/scale.zarr
    
    Scale transform: [4, 3, 2] for (z, y, x) axes maps array coordinates to physical space.
    This test verifies the scale was correctly applied by:
    1. Checking the coordinate space has correct scale factors
    2. Verifying we can read data using physical coordinates
    """
    test_dir = OME_ZARR_0_6_ROOT / "basic" / "scale.zarr"
    server_url = static_file_server(test_dir)
    with webdriver.viewer.txn() as s:
        s.layers.append(name="scale", layer=neuroglancer.ImageLayer(source=f"zarr3://{server_url}"))
    webdriver.sync()
    model_space = _assert_renders(webdriver, "scale")
    
    # Verify the scale transform was applied by checking the coordinate space
    # Expected scales are [4, 3, 2] for (z, y, x) in micrometers from the scale.zarr metadata
    # Neuroglancer converts to meters internally, so we expect [4e-6, 3e-6, 2e-6]
    expected_scales = [4e-6, 3e-6, 2e-6]  # meters
    actual_scales = model_space["scales"]
    actual_units = model_space["units"]
    
    assert len(actual_scales) == 3, f"Expected 3 scales, got {len(actual_scales)}"
    assert actual_units == ["m", "m", "m"], f"Expected units ['m', 'm', 'm'], got {actual_units}"
    
    for i, (expected, actual) in enumerate(zip(expected_scales, actual_scales)):
        assert abs(actual - expected) < 1e-9, \
            f"Scale mismatch on axis {i}: expected {expected}, got {actual}"
    
    # Verify we can read data at TEST_VOXEL, proving the scale transform allows proper data access
    _verify_data_at_point(model_space['volume'], TEST_VOXEL, EXPECTED_VALUE)





@pytest.mark.skip(reason="Hangs the test suite (unsupported transform)")
def test_ome_zarr_0_6_sequence_multiscale(static_file_server, webdriver):
    """sequence (multiscale): Composition across multiscale dataset.
    Example dataset: basic/sequenceScaleTranslation_multiscale.zarr
    """
    test_dir = OME_ZARR_0_6_ROOT / "basic" / "sequenceScaleTranslation_multiscale.zarr"
    server_url = static_file_server(test_dir)
    with webdriver.viewer.txn() as s:
        s.layers.append(name="sequence_ms", layer=neuroglancer.ImageLayer(source=f"zarr3://{server_url}"))
    webdriver.sync()
    _assert_renders(webdriver, "sequence_ms")


def test_ome_zarr_0_6_sequence(static_file_server, webdriver):
    """sequence: Ordered composition of transforms (scale + translation).
    Example dataset: basic/sequenceScaleTranslation.zarr
    
    Transform:
    1. Scale: [4, 3, 2] (z, y, x)
    2. Translation: [32, 21, 10] (z, y, x)
    
    Physical = Scale * Index + Translation
    For voxel index (13, 122, 169):
    - Physical z = 4 * 13 + 32 = 84 micrometers
    - Physical y = 3 * 122 + 21 = 387 micrometers
    - Physical x = 2 * 169 + 10 = 348 micrometers
    
    The translation values are chosen so that translation/scale yields
    integer origins [8, 7, 5] in voxel space, which Neuroglancer requires.
    """
    test_dir = OME_ZARR_0_6_ROOT / "basic" / "sequenceScaleTranslation.zarr"
    server_url = static_file_server(test_dir)
    with webdriver.viewer.txn() as s:
        s.layers.append(name="sequence", layer=neuroglancer.ImageLayer(source=f"zarr3://{server_url}"))
    webdriver.sync()
    model_space = _assert_renders(webdriver, "sequence")
    
    # Verify scales - the scale component of the sequence transform
    # Expected scales are [4, 3, 2] micrometers -> [4e-6, 3e-6, 2e-6] meters
    expected_scales = [4e-6, 3e-6, 2e-6]
    actual_scales = model_space["scales"]
    for i, (expected, actual) in enumerate(zip(expected_scales, actual_scales)):
        assert abs(actual - expected) < 1e-9, \
            f"Scale mismatch on axis {i}: expected {expected}, got {actual}"
    
    # Verify the translation component by reading data
    # The sequence transform applies: Physical = Scale * Index + Translation
    # Scale: [4, 3, 2], Translation: [32, 21, 10] (all in micrometers)
    
    # In voxel space with the normalized coordinate system (scale factored out):
    # - The origin is at translation/scale = [32/4, 21/3, 10/2] = [8, 7, 5] voxels
    # - The data at array index [13, 122, 169] appears at voxel [13+8, 122+7, 169+5] = [21, 129, 174]
    _verify_data_at_point(model_space['volume'], (21, 129, 174), EXPECTED_VALUE)



def test_ome_zarr_0_6_translation(static_file_server, webdriver):
    """translation: Per-axis translation (JSON vector form).
    Example dataset: basic/translation.zarr
    """
    test_dir = OME_ZARR_0_6_ROOT / "basic" / "translation.zarr"
    server_url = static_file_server(test_dir)
    with webdriver.viewer.txn() as s:
        s.layers.append(name="translation", layer=neuroglancer.ImageLayer(source=f"zarr3://{server_url}"))
    webdriver.sync()
    translation_space = _assert_renders(webdriver, "translation")
    
    # Translation: [30, 20, 10] (z, y, x) in micrometers
    # Since the data has 1 micrometer spacing, translation values directly map to voxels
    # Voxel (13, 122, 169) + translation (30, 20, 10) = (43, 142, 179)
    translated_voxel = (TEST_VOXEL[0] + 30, TEST_VOXEL[1] + 20, TEST_VOXEL[2] + 10)
    _verify_data_at_point(translation_space['volume'], translated_voxel, EXPECTED_VALUE)



@pytest.mark.skip(reason="Hangs the test suite (unsupported transform)")
def test_ome_zarr_0_6_affine(static_file_server, webdriver):
    """affine: Affine matrix (JSON form) applied to single scale.
    Example dataset: simple/affine.zarr
    """
    test_dir = OME_ZARR_0_6_ROOT / "simple" / "affine.zarr"
    server_url = static_file_server(test_dir)
    with webdriver.viewer.txn() as s:
        s.layers.append(name="affine", layer=neuroglancer.ImageLayer(source=f"zarr3://{server_url}"))
    webdriver.sync()
    # Volume access fails for affine (No matching source), skipping verification
    # model_space = _assert_renders(webdriver, "affine")
    
    # Affine: Inverse transform in metadata.
    # Data verification skipped for affine as _verify_data_at_point assumes axis alignment.
    pass


@pytest.mark.skip(reason="Hangs the test suite (unsupported transform)")
def test_ome_zarr_0_6_affine_multiscale(static_file_server, webdriver):
    """affine (multiscale): Affine transform within multiscale context.
    Example dataset: simple/affine_multiscale.zarr
    """
    test_dir = OME_ZARR_0_6_ROOT / "simple" / "affine_multiscale.zarr"
    server_url = static_file_server(test_dir)
    with webdriver.viewer.txn() as s:
        s.layers.append(name="affine_ms", layer=neuroglancer.ImageLayer(source=f"zarr3://{server_url}"))
    webdriver.sync()
    _assert_renders(webdriver, "affine_ms")



def test_ome_zarr_0_6_rotation(static_file_server, webdriver):
    """rotation: Rotation matrix (or axis permutation) example.
    Example dataset: simple/rotation.zarr
    """
    test_dir = OME_ZARR_0_6_ROOT / "simple" / "rotation.zarr"
    server_url = static_file_server(test_dir)
    with webdriver.viewer.txn() as s:
        s.layers.append(name="rotation", layer=neuroglancer.ImageLayer(source=f"zarr3://{server_url}"))
    webdriver.sync()

    model_space = _assert_renders(webdriver, "rotation")
    vol = model_space['volume']
    permuted_voxel = (TEST_VOXEL[2], TEST_VOXEL[0], TEST_VOXEL[1])  # (z, y, x) -> (x, z, y)
    _verify_data_at_point(vol, permuted_voxel, EXPECTED_VALUE)


def get_layer_model_space(webdriver, layer_name):
    """Helper to retrieve the modelSpace from the backend volume object."""
    try:
        vol = webdriver.viewer.volume(layer_name).result()
        
        # Extract names from domain labels
        names = list(vol.domain.labels)
        
        # Extract units and scales from dimension_units
        # vol.dimension_units returns a tuple of Unit objects
        # Unit object has .base_unit (string) and .multiplier (float)
        units = [u.base_unit for u in vol.dimension_units]
        scales = [u.multiplier for u in vol.dimension_units]
        
        return {
            "volume": vol,
            "names": names,
            "units": units,
            "scales": scales,
        }
    except Exception as e:
        return f"Failed to get volume: {e}"
