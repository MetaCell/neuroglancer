import neuroglancer
import numpy as np
import pytest


def test_numpy_data():
    shape = (8, 16, 32)
    data = np.zeros(shape)
    local_volume = neuroglancer.LocalVolume(data)
    assert local_volume.rank == len(shape)
    assert local_volume.shape == shape
    dimensions = neuroglancer.CoordinateSpace(
        names=local_volume.data._default_labels,
        units=local_volume.data._default_units,
        scales=local_volume.data._default_scales,
    )
    assert local_volume.dimensions.to_json() == dimensions.to_json()


def test_tensorstore_defaults():
    ts = pytest.importorskip("tensorstore")
    shape = (8, 16, 32)
    data = ts.open(
        {
            "driver": "n5",
            "kvstore": {
                "driver": "memory",
            },
            "metadata": {
                "dataType": "uint8",
                "dimensions": shape,
            },
            "create": True,
            "delete_existing": True,
        }
    ).result()
    local_volume = neuroglancer.LocalVolume(data)
    assert local_volume.rank == len(shape)
    assert local_volume.shape == shape
    dimensions = neuroglancer.CoordinateSpace(
        names=local_volume.data._default_labels,
        units=local_volume.data._default_units,
        scales=local_volume.data._default_scales,
    )
    assert local_volume.dimensions.to_json() == dimensions.to_json()


def test_tensorstore_features():
    ts = pytest.importorskip("tensorstore")
    shape = (8, 16, 32)
    offset = (2, 4, 8)
    labels = ["x", "y", ""]
    units = ["m", "m", "m"]
    scales = [1.0, 1.0, 0.5]
    data = ts.open(
        {
            "driver": "n5",
            "kvstore": {
                "driver": "memory",
            },
            "metadata": {
                "dataType": "uint8",
                "dimensions": shape,
                "units": units,
                "resolution": scales,
            },
            "transform": {
                "input_labels": labels,
                "input_inclusive_min": offset,
            },
            "create": True,
            "delete_existing": True,
        }
    ).result()
    local_volume = neuroglancer.LocalVolume(data)
    assert local_volume.rank == len(shape)
    assert local_volume.shape == tuple([s - o for s, o in zip(shape, offset)])
    dimensions = neuroglancer.CoordinateSpace(
        names=["x", "y", "d2"], units=units, scales=scales
    )
    assert local_volume.dimensions.to_json() == dimensions.to_json()


def test_volume_info_preserves_padded_dimensions(webdriver):
    viewer = webdriver.viewer
    padded_dimensions = neuroglancer.CoordinateSpace(
        names=["y", "x", "z"], units=["nm"] * 3, scales=[3, 5, 2]
    )
    with viewer.txn() as state:
        state.layout = "3d"
        state.show_slices = False
        state.layers["image"] = neuroglancer.ImageLayer(
            source=neuroglancer.LayerDataSource(
                url=neuroglancer.LocalVolume(
                    np.zeros((3, 5), dtype=np.uint8),
                    dimensions=neuroglancer.CoordinateSpace(
                        names=["y", "x"], units=["nm", "nm"], scales=[3, 5]
                    ),
                    voxel_offset=[2, 4],
                ),
                transform=neuroglancer.CoordinateSpaceTransform(
                    input_dimensions=padded_dimensions,
                    output_dimensions=padded_dimensions,
                    source_rank=2,
                    matrix=[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 7]],
                ),
            ),
            volume_rendering_mode="off",
        )
    webdriver.sync()
    info = viewer.volume_info("image").result(timeout=15)
    assert info.dimensions.rank == 3
    indices = [list(info.dimensions.names).index(axis) for axis in "xyz"]
    np.testing.assert_array_equal(np.array(info.lower_bound)[indices], [4, 2, 7])
    np.testing.assert_array_equal(np.array(info.upper_bound)[indices], [9, 5, 8])
