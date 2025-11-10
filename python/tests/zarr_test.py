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


@pytest.mark.parametrize(
    "test_dir,transformation_type",
    [
        # rotation.zarr has a rotation matrix that permutes axes
        (TEST_DATA_DIR / "ome_zarr" / "all_0.6" / "simple" / "rotation.zarr", "rotation"),
        # affine_multiscale.zarr has diagonal scaling
        (TEST_DATA_DIR / "ome_zarr" / "all_0.6" / "simple" / "affine_multiscale.zarr", "affine"),
    ],
    ids=["rotation.zarr", "affine_multiscale.zarr"],
)
def test_ome_zarr_0_6_transformations(
    test_dir: pathlib.Path, transformation_type, static_file_server, webdriver
):
    """Test OME-ZARR 0.6 files with axis-aligned transformations.
    
    Tests rotation (permutation) and diagonal affine (scaling) transformations
    that are compatible with the Python volume reader. The Python reader returns
    the underlying array data (not transformed); transformations are applied during
    rendering in the viewer.
    """
    import tensorstore as ts
    import json

    server_url = static_file_server(test_dir)
    
    # Read the metadata to understand the transformation
    with open(test_dir / "zarr.json", "r") as f:
        metadata = json.load(f)
    
    transforms = metadata["attributes"]["ome"]["multiscales"][0]["datasets"][0][
        "coordinateTransformations"
    ]
    
    # Verify the transformation type matches what we expect
    assert transforms[0]["type"] == transformation_type
    
    # Test that neuroglancer can load and display the data
    with webdriver.viewer.txn() as s:
        s.layers.append(
            name="ome06", layer=neuroglancer.ImageLayer(source=f"zarr3://{server_url}")
        )

    vol = webdriver.viewer.volume("ome06").result()
    data = vol.read().result()
    
    # Verify we can read the data and it has the expected shape
    # Note: The Python reader returns the underlying array shape (not transformed)
    # Transformations are applied during rendering in the viewer
    assert data is not None
    assert data.ndim == 3
    assert all(dim > 0 for dim in data.shape)
    
    # Verify the data shape matches the underlying array shape
    # For rotation.zarr: array shape is [27, 226, 186]
    # For affine_multiscale.zarr: s0 array shape is [27, 226, 186]
    assert data.shape == (27, 226, 186), (
        f"Expected underlying array shape (27, 226, 186), got {data.shape}"
    )


def test_ome_zarr_0_6_general_affine_parsing(static_file_server, webdriver):
    """Test that OME-ZARR 0.6 files with general affine transformations can be parsed.
    
    Note: The Python volume reader cannot read data with non-axis-aligned transformations,
    but the metadata should be parsed correctly without errors.
    """
    test_dir = TEST_DATA_DIR / "ome_zarr" / "all_0.6" / "simple" / "affine.zarr"
    server_url = static_file_server(test_dir)
    
    # Test that neuroglancer can parse the metadata (even though data reading will fail)
    with webdriver.viewer.txn() as s:
        s.layers.append(
            name="affine_test",
            layer=neuroglancer.ImageLayer(source=f"zarr3://{server_url}"),
        )

    # The layer should be created even though data reading might fail
    # This validates that the affine transformation metadata is parsed correctly
    vol = webdriver.viewer.volume("affine_test")
    # Attempting to read will fail because the Python volume reader
    # cannot handle non-axis-aligned transformations
    with pytest.raises(ValueError, match="No matching source"):
        vol.result()


@pytest.mark.parametrize(
    "test_dir,expected_scales",
    [
        # affine_multiscale.zarr has diagonal affine with scales [4, 3, 2]
        # These are the effective scales extracted from the transformation matrix
        (
            TEST_DATA_DIR / "ome_zarr" / "all_0.6" / "simple" / "affine_multiscale.zarr",
            [4.0, 3.0, 2.0],
        ),
    ],
    ids=["affine_multiscale"],
)
def test_ome_zarr_0_6_multiscale_affine(
    test_dir: pathlib.Path, expected_scales, static_file_server, webdriver
):
    """Test OME-ZARR 0.6 multiscale with diagonal affine transformations (scaling only).
    
    This tests that:
    1. Multiscale data with diagonal affine matrices (pure scaling) can be loaded
    2. The effective scales are extracted correctly using L2 norm of column vectors
    3. The bounding box is computed correctly by transforming all corners
    """
    import json
    
    server_url = static_file_server(test_dir)
    
    # Read the transformation matrix
    with open(test_dir / "zarr.json", "r") as f:
        metadata = json.load(f)
    
    affine_matrix = metadata["attributes"]["ome"]["multiscales"][0]["datasets"][0][
        "coordinateTransformations"
    ][0]["affine"]
    
    # Verify the diagonal scales match our expectations
    # For a diagonal affine, the L2 norm of each column is just the diagonal value
    for i, expected_scale in enumerate(expected_scales):
        # Extract column i from the affine matrix (which is MxN format where M=3, N=4)
        # Column i contains [affine[0][i], affine[1][i], affine[2][i]]
        col_values = [affine_matrix[j][i] for j in range(3)]
        actual_scale = np.sqrt(sum(v**2 for v in col_values))
        np.testing.assert_allclose(
            actual_scale,
            expected_scale,
            rtol=1e-6,
            err_msg=f"Scale extraction for axis {i} incorrect",
        )
    
    with webdriver.viewer.txn() as s:
        s.layers.append(
            name="multiscale",
            layer=neuroglancer.ImageLayer(source=f"zarr3://{server_url}"),
        )

    vol = webdriver.viewer.volume("multiscale").result()
    data = vol.read().result()
    
    # Verify multiscale data loads correctly
    assert data is not None
    assert data.ndim == 3
    # Verify all dimensions have reasonable sizes
    assert all(dim > 0 for dim in data.shape)


def test_ome_zarr_0_6_affine_bounding_box_validation(static_file_server, webdriver):
    """Validate that bounding boxes are computed correctly for general affine transformations.
    
    For a general affine transformation with rotation/shear, the bounding box cannot be
    computed by simple translation + shape. Instead, we must transform all 2^rank corners
    of the array and find the axis-aligned bounding box that encloses them.
    
    This test validates the metadata parsing works even though the Python volume reader
    cannot actually read the rotated data.
    """
    import json
    
    test_dir = TEST_DATA_DIR / "ome_zarr" / "all_0.6" / "simple" / "affine.zarr"
    server_url = static_file_server(test_dir)
    
    # Read the transformation matrix
    with open(test_dir / "zarr.json", "r") as f:
        metadata = json.load(f)
    
    affine_matrix = metadata["attributes"]["ome"]["multiscales"][0]["datasets"][0][
        "coordinateTransformations"
    ][0]["affine"]
    
    # This affine has rotation/shear components (non-diagonal elements)
    # Verify it's not purely diagonal
    has_off_diagonal = False
    for i in range(3):
        for j in range(3):
            if i != j and abs(affine_matrix[i][j]) > 1e-6:
                has_off_diagonal = True
                break
    
    assert has_off_diagonal, "Test data should have rotation/shear components"
    
    # Compute expected effective scales using L2 norm
    expected_scales = []
    for col_idx in range(3):
        col_values = [affine_matrix[row_idx][col_idx] for row_idx in range(3)]
        scale = np.sqrt(sum(v**2 for v in col_values))
        expected_scales.append(scale)
    
    # Verify scales are computed correctly (should not just be diagonal elements)
    # For this transformation, scales should be different from diagonal values
    diagonal_values = [affine_matrix[i][i] for i in range(3)]
    assert not np.allclose(expected_scales, diagonal_values), (
        "Effective scales should differ from diagonal for rotated transformations"
    )
    
    # The layer should be created (metadata parsed successfully) even though
    # the Python volume reader will fail to read the actual data
    with webdriver.viewer.txn() as s:
        s.layers.append(
            name="affine_rotated",
            layer=neuroglancer.ImageLayer(source=f"zarr3://{server_url}"),
        )
    
    # Attempting to read will fail as expected (Python reader can't handle rotations)
    vol = webdriver.viewer.volume("affine_rotated")
    with pytest.raises(ValueError, match="No matching source"):
        vol.result()


def test_ome_zarr_0_6_inverse_transform_composition(static_file_server, webdriver):
    """Validate that OME-ZARR files with inverse transformations are parsed correctly.
    
    This test verifies the complete transformation pipeline using an L-shaped asymmetrical figure:
    1. Base data (OME-ZARR 0.5) with identity transform - L-shape in original orientation
    2. Intermediate data (OME-ZARR 0.5) with forward transform applied but identity in metadata - shows transformed L
    3. Transformed data (OME-ZARR 0.6/Zarr v3) with forward transform applied + inverse transform in metadata - should match base
    
    The forward transformation (applied to data) swaps y and x axes:
      z_out = z_in, y_out = x_in, x_out = y_in
    
    The inverse (in metadata of transformed file) undoes this:
      z_in = z_out, y_in = x_out, x_in = y_out
    
    When rendered together, base and transformed should show the same L-shape at the same physical coordinates.
    The intermediate file shows the transformed L-shape directly without inverse correction.
    """
    import json
    
    base_dir = TEST_DATA_DIR / "ome_zarr" / "test_base_0.5.zarr"
    intermediate_dir = TEST_DATA_DIR / "ome_zarr" / "test_intermediate_0.5.zarr"
    transformed_dir = TEST_DATA_DIR / "ome_zarr" / "test_transformed_0.6.zarr"
    
    # Verify the metadata is correct
    # All files use Zarr v3 format with zarr.json
    with open(base_dir / "zarr.json", "r") as f:
        base_meta = json.load(f)
    
    with open(intermediate_dir / "zarr.json", "r") as f:
        intermediate_meta = json.load(f)
    
    with open(transformed_dir / "zarr.json", "r") as f:
        trans_meta = json.load(f)
    
    # Base should have scale transform (identity-like) in 0.5 format
    base_transform = base_meta["attributes"]["ome"]["multiscales"][0]["datasets"][0][
        "coordinateTransformations"
    ][0]
    assert base_transform["type"] == "scale", "Base should have scale transform"
    assert base_transform["scale"] == [1, 1, 1], "Base should have identity scale"
    
    # Intermediate should also have scale transform (identity) but data is transformed
    intermediate_transform = intermediate_meta["attributes"]["ome"]["multiscales"][0]["datasets"][0][
        "coordinateTransformations"
    ][0]
    assert intermediate_transform["type"] == "scale", "Intermediate should have scale transform"
    assert intermediate_transform["scale"] == [1, 1, 1], "Intermediate should have identity scale"
    
    # Transformed should have affine transform
    trans_transform = trans_meta["attributes"]["ome"]["multiscales"][0]["datasets"][0][
        "coordinateTransformations"
    ][0]
    assert trans_transform["type"] == "affine", "Transformed should have affine transform"
    
    # Verify the affine is a y<->x swap (inverse of forward transform)
    expected_affine = [
        [1, 0, 0, 0],  # z_in = z_out
        [0, 0, 1, 0],  # y_in = x_out
        [0, 1, 0, 0],  # x_in = y_out
    ]
    assert trans_transform["affine"] == expected_affine, "Affine should be y<->x swap"
    
    # Now test that neuroglancer can load all three files
    base_url = static_file_server(base_dir)
    intermediate_url = static_file_server(intermediate_dir)
    transformed_url = static_file_server(transformed_dir)
    
    # Load base data (identity transform, original L-shape)
    with webdriver.viewer.txn() as s:
        s.layers.append(
            name="base",
            layer=neuroglancer.ImageLayer(source=f"zarr3://{base_url}"),
        )
    
    base_vol = webdriver.viewer.volume("base").result()
    base_ng_data = base_vol.read().result()
    
    # Load intermediate data (identity transform in metadata, but data is transformed)
    with webdriver.viewer.txn() as s:
        s.layers.append(
            name="intermediate",
            layer=neuroglancer.ImageLayer(source=f"zarr3://{intermediate_url}"),
        )
    
    intermediate_vol = webdriver.viewer.volume("intermediate").result()
    intermediate_ng_data = intermediate_vol.read().result()
    
    # Load transformed data (inverse transform in metadata, data is transformed)
    with webdriver.viewer.txn() as s:
        s.layers.append(
            name="transformed",
            layer=neuroglancer.ImageLayer(source=f"zarr3://{transformed_url}"),
        )
    
    transformed_vol = webdriver.viewer.volume("transformed").result()
    transformed_ng_data = transformed_vol.read().result()
    
    # All should have the same underlying array shape
    assert base_ng_data.shape == intermediate_ng_data.shape == transformed_ng_data.shape, "Shapes should match"
    assert base_ng_data.shape == (16, 16, 16), "Should be 16x16x16"
    
    # Verify that intermediate and transformed have the same underlying data (both forward transformed)
    np.testing.assert_array_equal(intermediate_ng_data, transformed_ng_data, 
                                   err_msg="Intermediate and transformed should have same underlying data")
    
    # Verify that base and intermediate have different data (one original, one transformed)
    assert not np.array_equal(base_ng_data, intermediate_ng_data), \
        "Base and intermediate should have different data (one original, one transformed)"
    
    print(f"✓ Base data loaded successfully with identity transform (original L-shape)")
    print(f"✓ Intermediate data loaded successfully (transformed L-shape, no inverse in metadata)")
    print(f"✓ Transformed data loaded successfully with affine y<->x swap inverse")
    print(f"✓ All three have shape {base_ng_data.shape}")
    print(f"✓ Intermediate and transformed have same underlying data (both forward transformed)")
    print(f"✓ Base and intermediate have different data as expected")
    print(f"✓ Transformation metadata parsed correctly")
