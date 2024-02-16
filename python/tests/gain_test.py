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
"""Tests basic screenshot functionality."""

import neuroglancer
import numpy as np
import cv2
from skimage.metrics import structural_similarity as ssim
from time import sleep
import os
from PIL import Image
import pytest



URL = r"zarr://s3://aind-open-data/exaSPIM_653980_2023-08-10_20-08-29_fusion_2023-08-24/fused.zarr/"
image_path = 'image_comparison_assets/gain.png'
image_path_nogain = 'image_comparison_assets/nogain.png'

def add_render_panel(side="left", row=0, col=0):
    return neuroglancer.LayerSidePanelState(
        side=side,
        col=col,
        row=row,
        tab="rendering",
        tabs=["rendering", "source"],
    )

    
no_gain_screenshot = None

@pytest.mark.timeout(600)
def test_no_gain(webdriver):
    a = np.array([[[255]]], dtype=np.uint8)
    with webdriver.viewer.txn() as s:
        s.dimensions = neuroglancer.CoordinateSpace(
            names=["x", "y", "z", "t"], units=["nm", "nm", "um", "ms"], scales=[748, 748, 1, 1]
        )
        s.layers.append(
            name="a",
            layer=neuroglancer.ImageLayer(
                source = URL,
                panels=[add_render_panel()],
                shader="""
#uicontrol invlerp normalized(range=[0, 250], window=[0, 65535], clamp=true)
void main() {
  emitGrayscale(normalized());
}
""",
                volume_rendering=True,
                volumeRenderingDepthSamples=512,
                 tool_bindings={
                    "A": neuroglancer.VolumeRenderingDepthSamplesTool(),
                    "B": neuroglancer.VolumeRenderingGainTool(),  
                    }
                 ),
        )
        s.cross_section_scale = 1e-6
        s.show_axis_lines = False
        s.position = [0.5, 0.5, 0.5]
        s.layers["brain"].volumeRenderingGain = 0
        # s.layers["brain"].shaderControl
        # webdriver.sync()
    sleep(5)
    global no_gain_screenshot
    screenshot = webdriver.viewer.screenshot().screenshot
    no_gain_screenshot = screenshot.image_pixels
    assert screenshot.image_pixels.size != 0, "Image is empty"
    # Check if the image contains valid pixel values
    assert np.all(screenshot.image_pixels >= 0) and np.all(screenshot.image_pixels <= 255), "Image contains invalid pixel values"
    # webdriver.quit()
    
@pytest.mark.timeout(600)
def test_gain(webdriver):
    global no_gain_screenshot
    a = np.array([[[255]]], dtype=np.uint8)
    with webdriver.viewer.txn() as s:
        s.dimensions = neuroglancer.CoordinateSpace(
            names=["x", "y", "z", "t"], units=["nm", "nm", "um", "ms"], scales=[748, 748, 1, 1]
        )
        s.layers.append(
            name="a",
            layer=neuroglancer.ImageLayer(
                source = URL,
                panels=[add_render_panel()],
                shader="""
#uicontrol invlerp normalized(range=[0, 250], window=[0, 65535], clamp=true)
void main() {
  emitGrayscale(normalized());
}
""",
                volume_rendering=True,
                volumeRenderingDepthSamples=512,
                 tool_bindings={
                    "A": neuroglancer.VolumeRenderingDepthSamplesTool(),
                    "B": neuroglancer.VolumeRenderingGainTool(),  
                    }
                 ),
        )
        s.cross_section_scale = 1e-6
        s.show_axis_lines = False
        s.position = [0.5, 0.5, 0.5]
        s.layers["brain"].volumeRenderingGain = 5
        # s.layers["brain"].shaderControl
        # webdriver.sync()
    sleep(1)
    gain_screenshot = webdriver.viewer.screenshot().screenshot
    assert gain_screenshot.image_pixels.size != 0, "Image is empty"
    # Check if the image contains valid pixel values
    assert np.all(gain_screenshot.image_pixels >= 0) and np.all(gain_screenshot.image_pixels <= 255), "Image contains invalid pixel values"
    # assert np.array_equal(no_gain_screenshot, screenshot.image_pixels), "Screenshots are not equal"
   
   