/**
 * @license
 * Copyright 2016 Google Inc.
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

import { describe, it, expect } from "vitest";
import {
  SegmentColorHash,
  SegmentColorShaderManager,
} from "#src/segment_color.js";
import { randomUint64 } from "#src/util/bigint.js";
import { DataType } from "#src/util/data_type.js";
import { fragmentShaderTest } from "#src/webgl/shader_testing.js";

describe("segment_color", () => {
  it("the JavaScript implementation matches the WebGL shader implementation", () => {
    fragmentShaderTest(
      { inputValue: DataType.UINT64 },
      { outR: "float", outG: "float", outB: "float" },
      (tester) => {
        const shaderManager = new SegmentColorShaderManager("getColor");
        const { builder } = tester;
        shaderManager.defineShader(builder);
        const colorHash = SegmentColorHash.getDefault();
        builder.setFragmentMain(`
highp vec3 color = getColor(inputValue);
outR = color.r;
outG = color.g;
outB = color.b;
`);
        tester.build();
        const { gl, shader } = tester;
        shader.bind();
        shaderManager.enable(gl, shader, colorHash.value);

        function testValue(x: bigint) {
          tester.execute({ inputValue: x });
          const expected = new Float32Array(3);
          colorHash.compute(expected, x);
          const { values } = tester;
          expect(values.outR).toBeCloseTo(expected[0]);
          expect(values.outG).toBeCloseTo(expected[1]);
          expect(values.outB).toBeCloseTo(expected[2]);
        }

        testValue(0n);
        testValue(8n);
        const COUNT = 100;
        for (let iter = 0; iter < COUNT; ++iter) {
          const x = randomUint64();
          testValue(x);
        }
      },
    );
  });
});
