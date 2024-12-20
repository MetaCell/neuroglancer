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

/**
 * @file
 * Support for compressing uint32/uint64 segment label chunks.
 */

import {encodeChannel as encodeChannelCommon, encodeChannels as encodeChannelsCommon, writeBlock} from '#src/sliceview/compressed_segmentation/encode_common.js';
import {getFortranOrderStrides} from '#src/util/array.js';
import {Uint32ArrayBuilder} from '#src/util/uint32array_builder.js';

export {newCache} from '#src/sliceview/compressed_segmentation/encode_common.js';

let tempEncodingBuffer: Uint32Array;
let tempValuesBuffer1: Uint32Array;
let tempValuesBuffer2: Uint32Array;
let tempIndexBuffer1: Uint32Array;
let tempIndexBuffer2: Uint32Array;

const uint32sPerElement = /*@ strideMultiplier @*/;

export function encodeBlock(
    rawData: Uint32Array, inputOffset: number, inputStrides: ArrayLike<number>,
    blockSize: ArrayLike<number>, actualSize: ArrayLike<number>, baseOffset: number,
    cache: Map<string, number>, output: Uint32ArrayBuilder): [number, number] {
  const ax = actualSize[0], ay = actualSize[1], az = actualSize[2];
  const bx = blockSize[0], by = blockSize[1], bz = blockSize[2];
  const sx = inputStrides[0];
  let sy = inputStrides[1], sz = inputStrides[2];
  sz -= sy * ay;
  sy -= sx * ax;
  if (ax * ay * az === 0) {
    return [0, 0];
  }

  let numBlockElements = bx * by * bz + 31; // Add padding elements.
  if (tempEncodingBuffer === undefined || tempEncodingBuffer.length < numBlockElements) {
    tempEncodingBuffer = new Uint32Array(numBlockElements);
    tempValuesBuffer1 = new Uint32Array(numBlockElements * uint32sPerElement);
    tempValuesBuffer2 = new Uint32Array(numBlockElements * uint32sPerElement);
    tempIndexBuffer1 = new Uint32Array(numBlockElements);
    tempIndexBuffer2 = new Uint32Array(numBlockElements);
  }

  const encodingBuffer = tempEncodingBuffer.subarray(0, numBlockElements);
  encodingBuffer.fill(0);
  const valuesBuffer1 = tempValuesBuffer1;
  const valuesBuffer2 = tempValuesBuffer2;
  const indexBuffer1 = tempIndexBuffer1;
  const indexBuffer2 = tempIndexBuffer2;

  let noAdjacentDuplicateIndex = 0;
  {
    let prevLow = ((rawData[inputOffset] + 1) >>> 0);
    /*% if dataType == 'uint64' %*/
    let prevHigh = 0;
    /*% endif %*/
    let curInputOff = inputOffset;
    let blockElementIndex = 0;
    let bsy = bx - ax;
    let bsz = bx * by - bx * ay;
    for (let z = 0; z < az; ++z, curInputOff += sz, blockElementIndex += bsz) {
      for (let y = 0; y < ay; ++y, curInputOff += sy, blockElementIndex += bsy) {
        for (let x = 0; x < ax; ++x, curInputOff += sx) {
          let valueLow = rawData[curInputOff];
          /*% if dataType == 'uint64' %*/
          let valueHigh = rawData[curInputOff + 1];
          /*% endif %*/
          if (valueLow !== prevLow
              /*% if dataType == 'uint64' %*/
              || valueHigh !== prevHigh
              /*% endif %*/
             ) {
            prevLow = valuesBuffer1[noAdjacentDuplicateIndex * /*@ strideMultiplier @*/] = valueLow;
            /*% if dataType == 'uint64' %*/
            prevHigh = valuesBuffer1[noAdjacentDuplicateIndex * 2 + 1] = valueHigh;
            /*% endif %*/
            indexBuffer1[noAdjacentDuplicateIndex] = noAdjacentDuplicateIndex++;
          }
          encodingBuffer[blockElementIndex++] = noAdjacentDuplicateIndex;
        }
      }
    }
  }

  indexBuffer1.subarray(0, noAdjacentDuplicateIndex).sort((a, b) => {
    /*% if dataType == 'uint64' %*/
    let aHigh = valuesBuffer1[2 * a + 1];
    let bHigh = valuesBuffer1[2 * b + 1];
    let aLow = valuesBuffer1[2 * a];
    let bLow = valuesBuffer1[2 * b];
    return (aHigh - bHigh) || (aLow - bLow);
    /*% else %*/
    return valuesBuffer1[a] - valuesBuffer1[b];
    /*% endif %*/
  });

  let numUniqueValues = -1;
  {
    let prevLow = (valuesBuffer1[indexBuffer1[0] * uint32sPerElement] + 1) >>> 0;
    /*% if dataType == 'uint64' %*/
    let prevHigh = 0;
    /*% endif %*/
    for (let i = 0; i < noAdjacentDuplicateIndex; ++i) {
      let index = indexBuffer1[i];
      let valueIndex = index * uint32sPerElement;
      let valueLow = valuesBuffer1[valueIndex];
      /*% if dataType == 'uint64' %*/
      let valueHigh = valuesBuffer1[valueIndex + 1];
      /*% endif %*/
      if (valueLow !== prevLow
          /*% if dataType == 'uint64' %*/
          || valueHigh !== prevHigh
          /*% endif %*/
         ) {
        ++numUniqueValues;
        let outputIndex2 = numUniqueValues * uint32sPerElement;
        prevLow = valuesBuffer2[outputIndex2] = valueLow;
        /*% if dataType == 'uint64' %*/
        prevHigh = valuesBuffer2[outputIndex2 + 1] = valueHigh;
        /*% endif %*/
      }
      indexBuffer2[index + 1] = numUniqueValues;
    }
    ++numUniqueValues;
  }

  return writeBlock(output, baseOffset, cache, bx * by * bz, numUniqueValues, valuesBuffer2, encodingBuffer,
                    indexBuffer2, uint32sPerElement);
}

export function encodeChannel(
  output: Uint32ArrayBuilder, blockSize: ArrayLike<number>,
  rawData: Uint32Array, volumeSize: ArrayLike<number>,
  baseInputOffset: number = 0, inputStrides = getFortranOrderStrides(volumeSize, /*@ strideMultiplier @*/)) {
  return encodeChannelCommon(output, blockSize, rawData, volumeSize, baseInputOffset, inputStrides, encodeBlock);
}

export function encodeChannels(
    output: Uint32ArrayBuilder, blockSize: ArrayLike<number>, rawData: Uint32Array,
    volumeSize: ArrayLike<number>, baseInputOffset: number = 0,
    inputStrides = getFortranOrderStrides(volumeSize, /*@ strideMultiplier @*/)) {
  return encodeChannelsCommon(
      output, blockSize, rawData, volumeSize, baseInputOffset, inputStrides, encodeBlock);
}
