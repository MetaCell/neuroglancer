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



def add_render_panel(side="left", row=0, col=0):
    return neuroglancer.LayerSidePanelState(
        side=side,
        col=col,
        row=row,
        tab="rendering",
        tabs=["rendering", "source"],
    )

def add_image_layer(state, **kwargs):
    shape = (50,) * 3
    data = np.full(shape=shape, fill_value=255, dtype=np.uint8)
    dimensions = neuroglancer.CoordinateSpace(
        names=["x", "y", "z"], units="nm", scales=[400, 400, 400]
    )
    local_volume = neuroglancer.LocalVolume(data, dimensions)
    state.layers["image"] = neuroglancer.ImageLayer(
        source=local_volume,
        volume_rendering=True,
        tool_bindings={
            "A": neuroglancer.VolumeRenderingGainTool(),
        },
        panels=[add_render_panel()],
        **kwargs,
    )
    state.layout = "3d"

def get_shader():
    return """
void main() {
    emitRGBA(vec4(1.0, 1.0, 1.0, 0.001));
    }
    """

@pytest.fixture()
def shared_webdriver(request, webdriver):
    gainValue = request.node.get_closest_marker("gain_value").args[0]
    with webdriver.viewer.txn() as s:
        add_image_layer(s, shader=get_shader())
        s.layers["image"].volumeRenderingGain = gainValue
    yield webdriver

no_gain_screenshot = None
gain_screenshot = None

@pytest.mark.timeout(600)
@pytest.mark.gain_value(10)
def test_gain(shared_webdriver):
    global gain_screenshot
    global gain_avg
    shared_webdriver.sync()
    sleep(2)
    WebDriverWait(shared_webdriver.driver, 60).until(
        lambda driver: driver.execute_script('return document.readyState') == 'complete'
    )
    sleep(3)
    print("Layer loaded")
    canvas_element = WebDriverWait(shared_webdriver.driver, 10).until(
        EC.presence_of_element_located((By.CLASS_NAME, 'neuroglancer-layer-group-viewer'))
    )
    sleep(3)
    screenshot = canvas_element.screenshot_as_png
    with open('gain_screenshot.png', 'wb') as file:
        file.write(screenshot)
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
  
@pytest.mark.timeout(600)
@pytest.mark.gain_value(0)
def test_no_gain(shared_webdriver):
    
    global no_gain_avg
    global no_gain_screenshot
   
    shared_webdriver.sync()
    sleep(2)
    WebDriverWait(shared_webdriver.driver, 60).until(
        lambda driver: driver.execute_script('return document.readyState') == 'complete'
    )
    sleep(2)
    WebDriverWait(shared_webdriver.driver, 60).until(
        EC.text_to_be_present_in_element((By.CSS_SELECTOR, '#neuroglancer-container > div > div:nth-child(2) > div.neuroglancer-side-panel-column > div.neuroglancer-side-panel > div.neuroglancer-tab-view.neuroglancer-layer-side-panel-tab-view > div.neuroglancer-stack-view > div > div:nth-child(6) > label > div.neuroglancer-render-scale-widget.neuroglancer-layer-control-control > div.neuroglancer-render-scale-widget-legend > div:nth-child(2)'), '8/8')
    )
    
    print("Layer loaded")
    sleep(3)
    canvas_element = WebDriverWait(shared_webdriver.driver, 10).until(
        EC.presence_of_element_located((By.CLASS_NAME, 'neuroglancer-layer-group-viewer'))
    )
    sleep(3)
    screenshot = canvas_element.screenshot_as_png
    with open('no_gain_screenshot.png', 'wb') as file:
        file.write(screenshot)
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
  

    


@pytest.mark.timeout(10)
def test_gain_difference():
    sleep(2)
    assert gain_avg > no_gain_avg, "The gain screenshot is not brighter than the no gain screenshot"
    
    