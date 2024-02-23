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
from time import sleep
import pytest
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from PIL import Image
import io



URL = r"zarr://s3://aind-open-data/exaSPIM_653980_2023-08-10_20-08-29_fusion_2023-08-24/fused.zarr/"

def add_render_panel(side="left", row=0, col=0):
    return neuroglancer.LayerSidePanelState(
        side=side,
        col=col,
        row=row,
        tab="rendering",
        tabs=["rendering", "source"],
    )

    
no_gain_screenshot = None
gain_screenshot = None

@pytest.mark.timeout(600)
def test_no_gain(webdriver):
    
    global no_gain_avg
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
        s.layers["brain"].volumeRenderingGain = 0
    
    WebDriverWait(webdriver.driver, 60).until(
        EC.text_to_be_present_in_element((By.CSS_SELECTOR, '#neuroglancer-container > div > div:nth-child(2) > div.neuroglancer-side-panel-column > div.neuroglancer-side-panel > div.neuroglancer-tab-view.neuroglancer-layer-side-panel-tab-view > div.neuroglancer-stack-view > div > div:nth-child(6) > label > div.neuroglancer-render-scale-widget.neuroglancer-layer-control-control > div.neuroglancer-render-scale-widget-legend > div:nth-child(2)'), '16/16')
    )
    WebDriverWait(webdriver.driver, 60).until(
        lambda driver: driver.execute_script('return document.readyState') == 'complete'
    )
    print("Layer loaded")
    sleep(3)
    screenshot = webdriver.driver.get_screenshot_as_png()
    sleep(3)
    print("Screenshot taken")
     # Convert the screenshot to a NumPy array
    image = Image.open(io.BytesIO(screenshot))
    no_gain_screenshot = np.array(image)
    assert no_gain_screenshot.size != 0, "Image is empty"
    # Check if the image contains valid pixel values
    assert np.all(no_gain_screenshot >= 0) and np.all(no_gain_screenshot <= 255), "Image contains invalid pixel values"
    no_gain_avg = np.mean(no_gain_screenshot)
    print('No Gain average pixel value:')
    print(no_gain_avg)
  

    
@pytest.mark.timeout(600)
def test_gain(webdriver):
    global gain_screenshot
    global gain_avg
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
        s.layers["brain"].volumeRenderingGain = 10
    WebDriverWait(webdriver.driver, 60).until(
        lambda driver: driver.execute_script('return document.readyState') == 'complete'
    )
    sleep(3)
    print("Layer loaded")
    screenshot = webdriver.driver.get_screenshot_as_png()
    sleep(3)
    print("Screenshot taken")
     # Convert the screenshot to a NumPy array
    image = Image.open(io.BytesIO(screenshot))
    gain_screenshot = np.array(image)
    assert gain_screenshot.size != 0, "Image is empty"
    # Check if the image contains valid pixel values
    assert np.all(gain_screenshot >= 0) and np.all(gain_screenshot <= 255), "Image contains invalid pixel values"
    gain_avg = np.mean(gain_screenshot)
    print('Gain average pixel value:')
    print(gain_avg)
  

   
def test_gain_difference():
    sleep(2)
    assert gain_avg > no_gain_avg, "The gain screenshot is not brighter than the no gain screenshot"