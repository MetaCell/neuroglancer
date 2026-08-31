# @license
# Copyright 2020 Google Inc.
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
"""Tests that skeleton rendering can be controlled via ViewerState."""

import neuroglancer
import neuroglancer.skeleton
import numpy as np

dimensions = neuroglancer.CoordinateSpace(
    names=["x", "y", "z"], units="nm", scales=[1, 1, 1]
)

USER_SHADER = """
#uicontrol vec3 color color(default="white")
void main () {
  emitRGB(color);
}
"""


class SinglePointSkeletonSource(neuroglancer.skeleton.SkeletonSource):
    def __init__(self):
        super().__init__(dimensions=dimensions)

    def get_skeleton(self, object_id):
        return neuroglancer.skeleton.Skeleton(
            vertex_positions=[[0, 0, 0]],
            edges=[[0, 0]],
        )


class TwoNodeSkeletonSource(neuroglancer.skeleton.SkeletonSource):
    def __init__(self):
        super().__init__(dimensions=dimensions)

    def get_skeleton(self, object_id):
        return neuroglancer.skeleton.Skeleton(
            vertex_positions=[[-20, -20, 0], [20, 20, 0]],
            edges=[[0, 1]],
        )


def screenshot_pixels(webdriver, size):
    return webdriver.viewer.screenshot(size=[size, size]).screenshot.image_pixels


def render_skeleton(webdriver, source, *, layout, line_width, size, mode=None):
    """Draws one red skeleton on black and returns the screenshot pixels."""
    with webdriver.viewer.txn() as s:
        s.dimensions = dimensions
        s.position = [0, 0, 0]
        s.layout = layout
        s.projection_scale = 120
        s.cross_section_scale = 0.6
        # Otherwise grey in the slice view, and these tests read black as not drawn.
        s.cross_section_background_color = "#000000"
        s.show_axis_lines = False
        s.show_scale_bar = False
        s.layers.clear()
        s.layers.append(
            name="a",
            layer=neuroglancer.SegmentationLayer(source=source, segments=[1]),
        )
        rendering = s.layers[0].skeleton_rendering
        rendering.line_width2d = line_width
        rendering.line_width3d = line_width
        if mode is not None:
            if layout == "3d":
                rendering.mode3d = mode
            else:
                rendering.mode2d = mode
        rendering.shader = USER_SHADER
        rendering.shader_controls["color"] = "#f00"
    return screenshot_pixels(webdriver, size)


def assert_solid_color(image, color):
    np.testing.assert_array_equal(
        image, np.tile(np.array(color, dtype=np.uint8), image.shape[:2] + (1,))
    )


def test_skeleton_options(webdriver):
    # A marker wider than the viewport, so the colour can be checked exactly.
    image = render_skeleton(
        webdriver,
        SinglePointSkeletonSource(),
        layout="xy",
        line_width=100,
        size=10,
    )
    assert_solid_color(image, [255, 0, 0, 255])

    with webdriver.viewer.txn() as s:
        s.layout = "3d"
    assert_solid_color(screenshot_pixels(webdriver, 10), [255, 0, 0, 255])

    with webdriver.viewer.txn() as s:
        s.layers[0].source[0].subsources["default"] = False
    assert_solid_color(screenshot_pixels(webdriver, 10), [0, 0, 0, 255])


# Each entry pairs a mode with the only shading signature it produces.
FEATHERED = "feathered"  # slice view feathers the line edge
FLAT = "flat"  # no feather outside the slice view
LIT = "lit"  # shaded by the surface normal

RENDER_MODES = [
    ("xy", "lines", FEATHERED),
    ("xy", "lines_and_points", FEATHERED),
    ("3d", "lines", FLAT),
    ("3d", "lines_and_points", FLAT),
    ("3d", "cylinders", LIT),
    ("3d", "cylinders_and_balls", LIT),
]

ENLARGED_PAIRS = [
    ("xy", "lines", "lines_and_points"),
    ("3d", "lines", "lines_and_points"),
    ("3d", "cylinders", "cylinders_and_balls"),
]


def test_skeleton_render_mode(webdriver):
    drawn_counts = {}
    for layout, mode, shading in RENDER_MODES:
        case = f"{layout}/{mode}"
        image = render_skeleton(
            webdriver,
            TwoNodeSkeletonSource(),
            layout=layout,
            line_width=10,
            size=100,
            mode=mode,
        )
        red, green, blue = (image[..., i].astype(int) for i in range(3))
        # A pure red shader leaves the other channels untouched in every mode.
        np.testing.assert_array_equal(green, 0, err_msg=case)
        np.testing.assert_array_equal(blue, 0, err_msg=case)
        drawn_red = red[red != 0]
        assert len(drawn_red) > 200, f"{case} drew nothing recognisable"
        drawn_counts[(layout, mode)] = len(drawn_red)

        if shading is LIT:
            assert drawn_red.min() < 250, f"{case} is flat, so it is not lit"
            assert drawn_red.max() == 255, case
        elif shading is FLAT:
            np.testing.assert_array_equal(drawn_red, 255, err_msg=case)
        else:
            assert drawn_red.min() < 255, f"{case} has no feathered edge"
            assert drawn_red.max() == 255, case

    for layout, plain, enlarged in ENLARGED_PAIRS:
        assert drawn_counts[(layout, enlarged)] > drawn_counts[(layout, plain)], (
            f"{layout}/{enlarged} covers no more than {layout}/{plain}"
        )
