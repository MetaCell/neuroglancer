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
import { hashCombine } from "#src/gpu_hash/hash_function.js";
import { HashMapUint64, HashSetUint64 } from "#src/gpu_hash/hash_table.js";
import {
  GPUHashTable,
  HashMapShaderManager,
  HashSetShaderManager,
} from "#src/gpu_hash/shader.js";
import { randomUint64 } from "#src/util/bigint.js";
import { DataType } from "#src/util/data_type.js";
import { getRandomUint32 } from "#src/util/random.js";
import { fragmentShaderTest } from "#src/webgl/shader_testing.js";

const COUNT = 100;

describe("gpu_hash.shader", () => {
  it("hashCombineUint32", () => {
    fragmentShaderTest(
      { inputValue: "uint", hashSeed: "uint" },
      { outputValue: "uint" },
      (tester) => {
        const { builder } = tester;
        const hashTableShaderManager = new HashSetShaderManager("h");
        hashTableShaderManager.defineShader(builder);
        builder.setFragmentMain(
          "outputValue = hashCombine(hashSeed, inputValue);",
        );
        const testHash = (hashSeed: number, inputValue: number) => {
          tester.execute({ hashSeed, inputValue });
          const expected = hashCombine(hashSeed, inputValue);
          expect(tester.values.outputValue).toEqual(expected);
        };
        for (let k = 0; k < 50; ++k) {
          testHash(getRandomUint32(), getRandomUint32());
        }
      },
    );
  });

  it("hashCombine", () => {
    fragmentShaderTest(
      { inputValue: DataType.UINT64, hashSeed: "uint" },
      { outputValue: "uint" },
      (tester) => {
        const { builder } = tester;
        const hashTableShaderManager = new HashSetShaderManager("h");
        hashTableShaderManager.defineShader(builder);
        builder.setFragmentMain(
          "outputValue = hashCombine(hashSeed, inputValue);",
        );
        for (let k = 0; k < 20; ++k) {
          const inputValue = randomUint64();
          const hashSeed = getRandomUint32();
          tester.execute({ hashSeed, inputValue });
          let expected = hashCombine(
            hashSeed,
            Number(inputValue & 0xffffffffn),
          );
          expected = hashCombine(expected, Number(inputValue >> 32n));
          expect(tester.values.outputValue).toEqual(expected);
        }
      },
    );
  });

  it("GPUHashTable:HashSetUint64", () => {
    fragmentShaderTest(
      { inputValue: DataType.UINT64 },
      { outputValue: "bool" },
      (tester) => {
        const { gl, builder } = tester;
        const hashTableShaderManager = new HashSetShaderManager("h");
        hashTableShaderManager.defineShader(builder);
        builder.setFragmentMain("outputValue = h_has(inputValue);");
        const { shader } = tester;

        const hashTable = new HashSetUint64();
        const gpuHashTable = tester.registerDisposer(
          GPUHashTable.get(gl, hashTable),
        );
        const testValues = new Array<bigint>();
        while (testValues.length < COUNT) {
          const x = randomUint64();
          if (hashTable.has(x)) {
            continue;
          }
          testValues.push(x);
          hashTable.add(x);
        }
        const notPresentValues = new Array<bigint>();
        notPresentValues.push(hashTable.empty);
        while (notPresentValues.length < COUNT) {
          const x = randomUint64();
          if (hashTable.has(x)) {
            continue;
          }
          notPresentValues.push(x);
        }
        function checkPresent(x: bigint) {
          hashTableShaderManager.enable(gl, shader, gpuHashTable);
          tester.execute({ inputValue: x });
          return tester.values.outputValue;
        }
        testValues.forEach((x, i) => {
          expect(hashTable.has(x), `cpu: i = ${i}, x = ${x}`).toBe(true);
          expect(
            checkPresent(x),
            `gpu: i = ${i}, x = ${x}, index = ${hashTable.indexOf(x)}`,
          ).toBe(true);
        });
        notPresentValues.forEach((x, i) => {
          expect(hashTable.has(x), `cpu: i = ${i}, x = ${x}`).toBe(false);
          expect(checkPresent(x), `gpu: i = ${i}, x = ${x}`).toBe(false);
        });
      },
    );
  });

  it("GPUHashTable:HashMapUint64", () => {
    fragmentShaderTest(
      { key: DataType.UINT64 },
      { isPresent: "bool", outputValue: DataType.UINT64 },
      (tester) => {
        const { gl, builder } = tester;
        const shaderManager = new HashMapShaderManager("h");
        shaderManager.defineShader(builder);
        builder.setFragmentMain("isPresent = h_get(key, outputValue);");
        const { shader } = tester;
        const hashTable = new HashMapUint64();
        const gpuHashTable = tester.registerDisposer(
          GPUHashTable.get(gl, hashTable),
        );
        const testValues = new Array<bigint>();
        while (testValues.length < COUNT) {
          const x = randomUint64();
          if (hashTable.has(x)) {
            continue;
          }
          testValues.push(x);
          hashTable.set(x, randomUint64());
        }
        const notPresentValues = new Array<bigint>();
        notPresentValues.push(hashTable.empty);
        while (notPresentValues.length < COUNT) {
          const x = randomUint64();
          if (hashTable.has(x)) {
            continue;
          }
          notPresentValues.push(x);
        }
        function checkPresent(x: bigint) {
          shaderManager.enable(gl, shader, gpuHashTable);
          tester.execute({ key: x });
          const { values } = tester;
          const value = hashTable.get(x);
          const has = values.isPresent;
          expect(value !== undefined, `x=${x}`).toBe(values.isPresent);
          if (has) {
            expect(values.outputValue, `x=${x}`).toBe(value);
          }
        }
        testValues.forEach((x, i) => {
          expect(hashTable.has(x), `cpu: i = ${i}, x = ${x}`).toBe(true);
          checkPresent(x);
        });
        notPresentValues.forEach((x) => {
          checkPresent(x);
        });
      },
    );
  });
});
