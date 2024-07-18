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

 * @file Defines facilities for manipulation of empirical cumulative distribution functions.
 */
import { DataType } from "#src/util/data_type.js";
import {
  clampToInterval,
  defaultDataTypeRange,
  type DataTypeInterval,
} from "#src/util/lerp.js";
import { Uint64 } from "#src/util/uint64.js";

function calculateEmpiricalCdf(histogram: Float32Array): Float32Array {
  const totalSamples = histogram.reduce((a, b) => a + b, 0);
  let cumulativeCount = 0;
  const empiricalCdf = histogram.map((count) => {
    cumulativeCount += count;
    return cumulativeCount / totalSamples;
  });
  return empiricalCdf;
}

function calculateBinSize(
  histogram: Float32Array,
  previousRange: DataTypeInterval,
  inputDataType: DataType,
): number {
  const totalBins = histogram.length - 2; // Exclude the first and last bins.
  if (inputDataType === DataType.UINT64) {
    const numerator64 = new Uint64();
    const denominator64 = Uint64.fromNumber(totalBins);
    const min = previousRange[0] as Uint64;
    const max = previousRange[1] as Uint64;
    Uint64.subtract(numerator64, max, min);
    return numerator64.toNumber() / denominator64.toNumber();
  } else {
    const min = previousRange[0] as number;
    const max = previousRange[1] as number;
    return (max - min) / totalBins;
  }
}

function decreaseBound(
  bound: number | Uint64,
  dataType: DataType,
  change: number,
): number | Uint64 {
  if (dataType !== DataType.FLOAT32) {
    const minBound = defaultDataTypeRange[dataType][0];
    if (minBound === bound) {
      return bound;
    }
  }
  const delta = dataType === DataType.FLOAT32 ? change : Math.round(change);
  const temp = new Uint64();
  const decreasedBound =
    dataType === DataType.UINT64
      ? Uint64.subtract(temp, bound as Uint64, Uint64.fromNumber(delta))
      : (bound as number) - delta;
  if (dataType === DataType.FLOAT32) {
    return decreasedBound;
  }
  const maxDataRange = defaultDataTypeRange[dataType];
  return clampToInterval(maxDataRange, decreasedBound);
}

function increaseBound(
  bound: number | Uint64,
  dataType: DataType,
  change: number,
): number | Uint64 {
  if (dataType !== DataType.FLOAT32) {
    const maxBound = defaultDataTypeRange[dataType][1];
    if (maxBound === bound) {
      return bound;
    }
  }
  const delta = dataType === DataType.FLOAT32 ? change : Math.round(change);
  const temp = new Uint64();
  const increasedBound =
    dataType === DataType.UINT64
      ? Uint64.add(temp, bound as Uint64, Uint64.fromNumber(delta))
      : (bound as number) + delta;
  if (dataType === DataType.FLOAT32) {
    return increasedBound;
  }
  const maxDataRange = defaultDataTypeRange[dataType];
  return clampToInterval(maxDataRange, increasedBound);
}

export function computeRangeForCdf(
  histogram: Float32Array,
  lowerPercentile: number = 0.05,
  upperPercentile: number = 0.95,
  previousRange: DataTypeInterval,
  inputDataType: DataType,
): DataTypeInterval {
  // 256 bins total. First and last bin are below lower bound/above upper.
  let lowerBound = previousRange[0];
  let upperBound = previousRange[1];
  const cdf = calculateEmpiricalCdf(histogram);
  const binSize = calculateBinSize(histogram, previousRange, inputDataType);

  // Find the indices of the percentiles.
  let lowerIndex = 0;
  for (let i = 0; i < cdf.length; i++) {
    lowerIndex = i;
    if (cdf[i] > lowerPercentile) {
      break;
    }
  }

  let upperIndex = cdf.findIndex((cdfValue) => cdfValue >= upperPercentile);
  upperIndex = upperIndex === -1 ? histogram.length - 1 : upperIndex;

  // Find new bounds based on the indices, either by trimming or expanding.
  if (lowerIndex === 0) {
    let shiftAmount = binSize / 2;
    if (inputDataType === DataType.FLOAT32) {
      shiftAmount = Math.max(
        shiftAmount,
        Math.max(1, Math.abs((lowerBound as number) / 2)),
      );
    }
    lowerBound = decreaseBound(lowerBound, inputDataType, shiftAmount);
  } else {
    const shiftAmount = lowerIndex - 1; // Exclude the first bin.
    lowerBound = increaseBound(
      lowerBound,
      inputDataType,
      binSize * shiftAmount,
    );
  }
  if (upperIndex === histogram.length - 1) {
    let shiftAmount = binSize / 2;
    if (inputDataType === DataType.FLOAT32) {
      shiftAmount = Math.max(
        shiftAmount,
        Math.max(1, Math.abs((upperBound as number) / 2)),
      );
    }
    upperBound = increaseBound(upperBound, inputDataType, shiftAmount);
  } else {
    const shiftAmount = histogram.length - 2 - upperIndex; // Exclude the first bin.
    upperBound = decreaseBound(
      upperBound,
      inputDataType,
      binSize * shiftAmount,
    );
  }

  return [lowerBound, upperBound] as DataTypeInterval;
}

export function makeAutoRangeButtons(
  parent: HTMLDivElement,
  minMaxHandler: () => void,
  oneTo99Handler: () => void,
  fiveTo95Handler: () => void,
) {
  const minMaxButton = document.createElement("button");
  minMaxButton.textContent = "Min-Max";
  minMaxButton.title = "Set range to the minimum and maximum values.";
  minMaxButton.classList.add("neuroglancer-auto-range-button");
  minMaxButton.addEventListener("click", minMaxHandler);
  parent.appendChild(minMaxButton);

  const midButton = document.createElement("button");
  midButton.textContent = "1-99%";
  midButton.title = "Set range to the 1st and 99th percentiles.";
  midButton.classList.add("neuroglancer-auto-range-button");
  midButton.addEventListener("click", oneTo99Handler);
  parent.appendChild(midButton);

  const highButton = document.createElement("button");
  highButton.textContent = "5-95%";
  highButton.title = "Set range to the 5th and 95th percentiles.";
  highButton.classList.add("neuroglancer-auto-range-button");
  highButton.addEventListener("click", fiveTo95Handler);
  parent.appendChild(highButton);
}
