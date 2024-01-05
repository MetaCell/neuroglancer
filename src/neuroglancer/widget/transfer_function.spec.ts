/**
 * @license
 * Copyright 2024 Google Inc.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

import { lerpBetweenControlPoints, TRANSFER_FUNCTION_LENGTH, NUM_COLOR_CHANNELS, ControlPoint } from "neuroglancer/widget/transfer_function";
import {vec4} from 'neuroglancer/util/geom';

describe("lerpBetweenControlPoints", () => {
    const output = new Uint8Array(NUM_COLOR_CHANNELS * TRANSFER_FUNCTION_LENGTH);
    it("returns transparent black when given no control points", () => {
        const controlPoints: ControlPoint[] = [];
        lerpBetweenControlPoints(output, controlPoints);
        expect(output.every(value => value === 0)).toBeTruthy();
    });
    it("returns transparent black up to the first control point, and the last control point value after", () => {
        const controlPoints: ControlPoint[] = [
            { position: 120, color: vec4.fromValues(21, 22, 254, 210) },
        ];
        lerpBetweenControlPoints(output, controlPoints);
        expect(output.slice(0, NUM_COLOR_CHANNELS * 120).every(value => value === 0)).toBeTruthy();
        const endPiece = output.slice(NUM_COLOR_CHANNELS * 120);
        const color = controlPoints[0].color;
        expect(endPiece.every((value, index) => value === color[index % NUM_COLOR_CHANNELS])).toBeTruthy()
    });
    it("correctly interpolates between three control points", () => {
        const controlPoints: ControlPoint[] = [
            { position: 120, color: vec4.fromValues(21, 22, 254, 210) },
            { position: 140, color: vec4.fromValues(0, 0, 0, 0) },
            { position: 200, color: vec4.fromValues(255, 255, 255, 255) },
        ];
        lerpBetweenControlPoints(output, controlPoints);
        expect(output.slice(0, NUM_COLOR_CHANNELS * 120).every(value => value === 0)).toBeTruthy();
        expect(output.slice(NUM_COLOR_CHANNELS * 200).every(value => value === 255)).toBeTruthy();

        const firstColor = controlPoints[0].color;
        const secondColor = controlPoints[1].color;
        for (let i = 120 * NUM_COLOR_CHANNELS; i < 140 * NUM_COLOR_CHANNELS; i++) {
            const difference = Math.floor((i - (120 * NUM_COLOR_CHANNELS)) / 4);
            const expectedValue = firstColor[i % NUM_COLOR_CHANNELS] + ((secondColor[i % NUM_COLOR_CHANNELS] - firstColor[i % NUM_COLOR_CHANNELS]) * difference / 20);
            const decimalPart = expectedValue - Math.floor(expectedValue);
            // If the decimal part is 0.5, it could be rounded up or down depending on precision.
            if (Math.abs(decimalPart - 0.5) < 0.001) {
                expect([Math.floor(expectedValue), Math.ceil(expectedValue)]).toContain(output[i]);
            }
            else {
                expect(output[i]).toBe(Math.round(expectedValue));
            }
        }

        const thirdColor = controlPoints[2].color;
        for (let i = 140 * NUM_COLOR_CHANNELS; i < 200 * NUM_COLOR_CHANNELS; i++) {
            const difference = Math.floor((i - (140 * NUM_COLOR_CHANNELS)) / 4);
            const expectedValue = secondColor[i % NUM_COLOR_CHANNELS] + ((thirdColor[i % NUM_COLOR_CHANNELS] - secondColor[i % NUM_COLOR_CHANNELS]) * difference / 60);
            const decimalPart = expectedValue - Math.floor(expectedValue);
            // If the decimal part is 0.5, it could be rounded up or down depending on precision.
            if (Math.abs(decimalPart - 0.5) < 0.001) {
                expect([Math.floor(expectedValue), Math.ceil(expectedValue)]).toContain(output[i]);
            }
            else {
                expect(output[i]).toBe(Math.round(expectedValue));
            }
        }
    });

});