"""Tests for the picking indicator overlay.

Toggling the indicator from the command palette must show or hide the ring
immediately, without a mouse move after the palette closes."""

import time

import neuroglancer
import numpy as np
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support import expected_conditions
from selenium.webdriver.support.ui import WebDriverWait

RING_STATE_SCRIPT = (
    "const ring = document.querySelector('.neuroglancer-picking-indicator');"
    "return ring === null ? 'absent' : (ring.hidden || ring.parentElement.hidden ? 'hidden' : 'shown');"
)


def _ring_state(driver):
    return driver.execute_script(RING_STATE_SCRIPT)


def _wait_for_ring_state(driver, wanted, timeout=5.0):
    deadline = time.time() + timeout
    seen = None
    while time.time() < deadline:
        seen = _ring_state(driver)
        if seen == wanted:
            return seen
        time.sleep(0.05)
    return seen


def test_palette_toggle_shows_ring_without_mouse_move(webdriver):
    with webdriver.viewer.txn() as s:
        s.dimensions = neuroglancer.CoordinateSpace(
            names=["x", "y", "z"], units="nm", scales=[1, 1, 1]
        )
        s.layers.append(
            name="image",
            layer=neuroglancer.ImageLayer(
                source=neuroglancer.LocalVolume(
                    data=np.full((64, 64, 64), 128, dtype=np.uint8),
                    dimensions=s.dimensions,
                )
            ),
        )
        s.layout = "xy"
        s.position = [32, 32, 32]
        s.cross_section_scale = 0.5
    with webdriver.viewer.config_state.txn() as cs:
        cs.show_ui_controls = False
        cs.show_panel_borders = False
    webdriver.sync()
    driver = webdriver.driver

    # Hovering focuses the panel so the palette key binding reaches it.
    ActionChains(driver).move_to_element(webdriver.root_element).perform()
    time.sleep(1.0)
    assert _ring_state(driver) in ("absent", "hidden")

    ActionChains(driver).key_down(Keys.CONTROL).send_keys("p").key_up(
        Keys.CONTROL
    ).perform()
    search_input = WebDriverWait(driver, 5).until(
        expected_conditions.presence_of_element_located(
            (By.CSS_SELECTOR, ".neuroglancer-command-palette-input")
        )
    )
    search_input.send_keys("toggle picking indicator")
    time.sleep(0.3)
    search_input.send_keys(Keys.ENTER)

    # No mouse movement from here on.
    state = _wait_for_ring_state(driver, "shown")
    webdriver.sync()
    assert webdriver.viewer.state.to_json().get("showPickingIndicator") is True
    assert state == "shown", f"ring state after toggle: {state}"

    ActionChains(driver).key_down(Keys.CONTROL).send_keys("p").key_up(
        Keys.CONTROL
    ).perform()
    search_input = WebDriverWait(driver, 5).until(
        expected_conditions.presence_of_element_located(
            (By.CSS_SELECTOR, ".neuroglancer-command-palette-input")
        )
    )
    search_input.send_keys("toggle picking indicator")
    time.sleep(0.3)
    search_input.send_keys(Keys.ENTER)
    state = _wait_for_ring_state(driver, "hidden")
    assert state == "hidden", f"ring state after second toggle: {state}"
