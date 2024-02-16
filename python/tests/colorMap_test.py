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



def setup_nogain_webdriver(webdriver):
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
    

def setup_gain_webdriver(webdriver):
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
        

    
    
def setup_colormappoints_webdriver(webdriver):
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
#uicontrol transferFunction colormap(points=[[26201, "#000000", 0], [31198, "#28e63f", 0.7], [45868, "#ffffff", 1]])
# void main() {
  //emitGrayscale(normalized());
  emitRGBA(colormap());
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
        


# def test_noGain_screenshot( webdriver):
#     setup_only3dnogain_webdriver(webdriver)
#     sleep(6) 
#     screenshot = webdriver.viewer.screenshot().screenshot
#     sleep(2) 
#     if not os.listdir(os.path.dirname(image_path_nogain)):
#         image = Image.fromarray(screenshot.image_pixels)
#         image.save(image_path_nogain)
#         print('Base Image saved')
#     similarity = compare_images(screenshot.image_pixels, image_path_nogain)
#     assert similarity > 0.95

# def test_gain_screenshot( webdriver):
#     setup_only3dwithgain_webdriver(webdriver)
#     sleep(6) 
#     screenshot = webdriver.viewer.screenshot().screenshot
#     sleep(2) 
#     if not os.listdir(os.path.dirname(image_path)):
#         image = Image.fromarray(screenshot.image_pixels)
#         image.save(image_path)
#         print('Base Image saved')
#     similarity = compare_images(screenshot.image_pixels, screenshot.image_pixels)
#     assert similarity > 0.95

def test_colormap_screenshot( webdriver):
    setup_colormappoints_webdriver(webdriver)
    sleep(6) 
    screenshot = webdriver.viewer.screenshot().screenshot
    sleep(2) 
    if not os.listdir(os.path.dirname(image_path)):
        image = Image.fromarray(screenshot.image_pixels)
        image.save(image_path)
        print('Base Image saved')
    similarity = compare_images(screenshot.image_pixels, screenshot.image_pixels)
    assert similarity > 0.95