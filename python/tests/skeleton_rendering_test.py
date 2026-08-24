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
import pytest

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
    """One diagonal edge, so that a cylinder has a real axis to be oriented along."""

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


@pytest.mark.parametrize(
    "layout,mode",
    [
        ("xy", "lines"),
        ("xy", "lines_and_points"),
        ("3d", "lines"),
        ("3d", "lines_and_points"),
        ("3d", "cylinders"),
        ("3d", "cylinders_and_spheres"),
    ],
)
def test_skeleton_render_mode(webdriver, layout, mode):
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
    np.testing.assert_array_equal(green, 0)
    np.testing.assert_array_equal(blue, 0)
    drawn_red = red[red != 0]
    assert len(drawn_red) > 200, "nothing recognisable was drawn"
    if mode in ("cylinders", "cylinders_and_spheres"):
        # Lit by the surface normal, so the red varies; a billboard would be flat.
        assert drawn_red.min() < 250
        assert drawn_red.max() == 255
    elif layout == "3d":
        # No feather outside the slice view, so every drawn pixel is the full colour.
        np.testing.assert_array_equal(drawn_red, 255)


@pytest.mark.parametrize(
    "plain,enlarged",
    [("lines", "lines_and_points"), ("cylinders", "cylinders_and_spheres")],
)
def test_skeleton_enlarged_nodes_cover_more(webdriver, plain, enlarged):
    def drawn_pixel_count(mode):
        image = render_skeleton(
            webdriver,
            TwoNodeSkeletonSource(),
            layout="3d",
            line_width=10,
            size=100,
            mode=mode,
        )
        return int((image[..., 0] != 0).sum())

    assert drawn_pixel_count(enlarged) > drawn_pixel_count(plain)
