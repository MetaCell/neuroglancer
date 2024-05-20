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

# import io
# from time import sleep

import neuroglancer
import numpy as np

# import pytest
# from PIL import Image

# from selenium.webdriver.common.by import By
# from selenium.webdriver.support import expected_conditions as EC
# from selenium.webdriver.support.ui import WebDriverWait


def add_image_layer(state):
    data = np.full(shape=(10,) * 3, fill_value=255, dtype=np.uint8)
    dimensions = neuroglancer.CoordinateSpace(
        names=["x", "y", "z"], units="nm", scales=[1, 1, 1]
    )
    local_volume = neuroglancer.LocalVolume(data, dimensions)
    state.layers["image"] = neuroglancer.ImageLayer(
        source=local_volume,
        volume_rendering=True,
        shader="""
void main() {
    emitRGBA(vec4(1.0, 1.0, 1.0, 0.001));
    }
    """,
    )
    state.show_axis_lines = False
    state.projection_scale = 1e-8
    state.position = [5, 5, 5]
    state.layout = "3d"
    state.show_default_annotations = False


def test_gain(webdriver):
    with webdriver.viewer.txn() as s:
        add_image_layer(s)
        s.layers["image"].volumeRenderingGain = 10
    webdriver.viewer.ready.wait()
    webdriver.sync()
    screenshot_response = webdriver.viewer.screenshot(size=[10, 10])
    gain_screenshot = screenshot_response.screenshot
    print(gain_screenshot.image_pixels)
    assert gain_screenshot.image_pixels.shape == (10, 10, 4)
    with webdriver.viewer.txn() as s:
        s.layers["image"].volumeRenderingGain = 0
    webdriver.viewer.ready.wait()
    webdriver.sync()
    screenshot_response = webdriver.viewer.screenshot(size=[10, 10])
    no_gain_screenshot = screenshot_response.screenshot
    print(no_gain_screenshot.image_pixels)
    assert no_gain_screenshot.image_pixels.shape == (10, 10, 4)
    assert np.mean(gain_screenshot.image_pixels) > np.mean(
        no_gain_screenshot.image_pixels
    )
    assert np.all(gain_screenshot >= no_gain_screenshot)
    
    # Check if the image contains valid pixel values
    # assert np.all(gain_screenshot >= 0) and np.all(
    #     gain_screenshot <= 255
    # ), "Image contains invalid pixel values"


# @pytest.mark.timeout(600)
# @pytest.mark.gain_value(0)
# def test_no_gain(shared_webdriver):

#     global no_gain_avg
#     global no_gain_screenshot

#     # shared_webdriver.sync()
#     sleep(2)
#     WebDriverWait(shared_webdriver.driver, 60).until(
#         lambda driver: driver.execute_script("return document.readyState") == "complete"
#     )
#     sleep(2)
#     # WebDriverWait(shared_webdriver.driver, 60).until(
#     #      EC.text_to_be_present_in_element((By.CSS_SELECTOR, '#neuroglancer-container > div > div:nth-child(2) > div.neuroglancer-side-panel-column > div.neuroglancer-side-panel > div.neuroglancer-tab-view.neuroglancer-layer-side-panel-tab-view > div.neuroglancer-stack-view > div > div:nth-child(6) > label > div.neuroglancer-render-scale-widget.neuroglancer-layer-control-control > div.neuroglancer-render-scale-widget-legend > div:nth-child(2)'), '8/8')
#     # )

#     print("Layer loaded")
#     sleep(3)
#     canvas_element = WebDriverWait(shared_webdriver.driver, 10).until(
#         EC.presence_of_element_located(
#             (By.CLASS_NAME, "neuroglancer-layer-group-viewer")
#         )
#     )
#     screenshot = canvas_element.screenshot_as_png
#     with open("no_gain_screenshot.png", "wb") as file:
#         file.write(screenshot)
#     sleep(3)
#     print("Screenshot taken")
#     # Convert the screenshot to a NumPy array
#     image = Image.open(io.BytesIO(screenshot))
#     no_gain_screenshot = np.array(image)
#     assert no_gain_screenshot.size != 0, "Image is empty"
#     # Check if the image contains valid pixel values
#     assert np.all(no_gain_screenshot >= 0) and np.all(
#         no_gain_screenshot <= 255
#     ), "Image contains invalid pixel values"
#     no_gain_avg = np.mean(no_gain_screenshot)
#     print("No Gain average pixel value:")
#     print(no_gain_avg)


# @pytest.mark.timeout(10)
# def test_gain_difference():
#     sleep(2)
#     assert (
#         gain_avg > no_gain_avg
#     ), "The gain screenshot is not brighter than the no gain screenshot"
