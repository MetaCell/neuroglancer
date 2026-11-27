/**
 * @license
 * Copyright 2026 Google Inc.
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

import { fetchZstdDictionaryWasm } from "#src/async_computation/zstd_dictionary/fetch_wasm.js";

const libraryEnv = {
  emscripten_notify_memory_growth: () => {},
};

interface ZstdDictionaryExports {
  memory: WebAssembly.Memory;
  _initialize(): void;
  malloc(size: number): number;
  free(pointer: number): void;
  ZSTD_createDCtx(): number;
  ZSTD_freeDCtx(context: number): number;
  ZSTD_createDDict(dictionary: number, dictionarySize: number): number;
  ZSTD_freeDDict(dictionary: number): number;
  ZSTD_decompress_usingDDict(
    context: number,
    output: number,
    outputCapacity: number,
    input: number,
    inputSize: number,
    dictionary: number,
  ): number;
  ZSTD_isError(code: number): number;
}

let zstdDictionaryModulePromise: Promise<ZstdDictionaryExports> | undefined;

function getZstdDictionaryModulePromise(): Promise<ZstdDictionaryExports> {
  if (zstdDictionaryModulePromise === undefined) {
    zstdDictionaryModulePromise = (async () => {
      const { instance } = await WebAssembly.instantiateStreaming(
        fetchZstdDictionaryWasm(),
        { env: libraryEnv },
      );
      const exports = instance.exports as unknown as ZstdDictionaryExports;
      exports._initialize();
      return exports;
    })();
    // A failed load is retried by the next call instead of failing forever.
    zstdDictionaryModulePromise.catch(() => {
      zstdDictionaryModulePromise = undefined;
    });
  }
  return zstdDictionaryModulePromise;
}

export async function decompressZstdStringsWithDictionary(
  dictionary: Uint8Array<ArrayBuffer>,
  frames: Uint8Array<ArrayBuffer>,
  frameEnds: Uint32Array<ArrayBuffer>,
  maxDecodedLength: number,
): Promise<string[]> {
  const zstd = await getZstdDictionaryModulePromise();
  let dictionaryPointer = 0;
  let framesPointer = 0;
  let outputPointer = 0;
  let decompressionDictionary = 0;
  let decompressionContext = 0;
  try {
    dictionaryPointer = zstd.malloc(dictionary.length);
    framesPointer = zstd.malloc(frames.length);
    outputPointer = zstd.malloc(maxDecodedLength);
    if (dictionaryPointer === 0 || framesPointer === 0 || outputPointer === 0) {
      throw new Error("Failed to allocate zstd buffers");
    }
    // Each allocation can grow the memory, which detaches older views of it.
    const heap = new Uint8Array(zstd.memory.buffer);
    heap.set(dictionary, dictionaryPointer);
    heap.set(frames, framesPointer);
    decompressionDictionary = zstd.ZSTD_createDDict(
      dictionaryPointer,
      dictionary.length,
    );
    decompressionContext = zstd.ZSTD_createDCtx();
    if (decompressionDictionary === 0 || decompressionContext === 0) {
      throw new Error("Failed to create zstd decompression state");
    }
    const textDecoder = new TextDecoder("utf-8", { fatal: true });
    const strings = new Array<string>(frameEnds.length);
    let frameStart = 0;
    for (let i = 0; i < frameEnds.length; ++i) {
      const frameEnd = frameEnds[i];
      const decodedLength = zstd.ZSTD_decompress_usingDDict(
        decompressionContext,
        outputPointer,
        maxDecodedLength,
        framesPointer + frameStart,
        frameEnd - frameStart,
        decompressionDictionary,
      );
      if (zstd.ZSTD_isError(decodedLength)) {
        throw new Error(`Failed to decompress zstd frame ${i}`);
      }
      strings[i] = textDecoder.decode(
        new Uint8Array(zstd.memory.buffer, outputPointer, decodedLength),
      );
      frameStart = frameEnd;
    }
    return strings;
  } finally {
    zstd.ZSTD_freeDCtx(decompressionContext);
    zstd.ZSTD_freeDDict(decompressionDictionary);
    zstd.free(outputPointer);
    zstd.free(framesPointer);
    zstd.free(dictionaryPointer);
  }
}
