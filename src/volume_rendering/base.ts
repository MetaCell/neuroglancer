/**
 * @license
 * Copyright 2020 Google Inc.
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

import type { ProjectionParameters } from "#src/projection_parameters.js";
import type {
  MultiscaleVolumetricDataRenderLayer,
  TransformedSource,
} from "#src/sliceview/base.js";
import { forEachVisibleVolumetricChunk } from "#src/sliceview/base.js";
import type { VolumeChunkSource } from "#src/sliceview/volume/base.js";
import type { vec3 } from "#src/util/geom.js";
import {
  getViewFrustumDepthRange,
  mat3,
  mat3FromMat4,
  prod3,
} from "#src/util/geom.js";

export const VOLUME_RENDERING_RENDER_LAYER_RPC_ID =
  "volume_rendering/VolumeRenderingRenderLayer";
export const VOLUME_RENDERING_RENDER_LAYER_UPDATE_SOURCES_RPC_ID =
  "volume_rendering/VolumeRenderingRenderLayer/update";

const DEBUG_CHUNK_LEVEL = false;

const tempMat3 = mat3.create();
// const tempMat4 = mat4.create();
// const tempVisibleVolumetricClippingPlanes = new Float32Array(24);

export interface HistogramInformation {
  spatialScales: Map<number, number>;
  activeIndex: number;
}

export function getVolumeRenderingNearFarBounds(
  clippingPlanes: Float32Array,
  displayLowerBound: Float32Array,
  displayUpperBound: Float32Array,
) {
  let volumeMinZ = 0;
  let volumeMaxZ = 0;
  for (let i = 0; i < 3; ++i) {
    const planeCoeff = clippingPlanes[16 + i];
    const a = planeCoeff * displayLowerBound[i];
    const b = planeCoeff * displayUpperBound[i];
    volumeMinZ += Math.min(a, b);
    volumeMaxZ += Math.max(a, b);
  }
  const near = -clippingPlanes[19];
  const adjustedNear = Math.max(near, volumeMinZ);
  const far = clippingPlanes[23];
  const adjustedFar = Math.min(far, volumeMaxZ);
  return { near, far, adjustedNear, adjustedFar };
}

// Returns target volume in "world" space.
// function getTargetVolume(
//     tsource: TransformedSource<MultiscaleVolumetricDataRenderLayer>,
//     projectionParameters: ProjectionParameters) {
//   const modelViewProjection = mat4.multiply(
//       tempMat4, projectionParameters.viewProjectionMat, tsource.chunkLayout.transform);
//   const clippingPlanes = tempVisibleVolumetricClippingPlanes;
//   getFrustumPlanes(clippingPlanes, modelViewProjection);
//   const {near, far} = getVolumeRenderingNearFarBounds(
//       clippingPlanes, tsource.lowerClipDisplayBound, tsource.upperClipDisplayBound);
//   if (near === far) return -1;
//   const depthRange = (far - near);
//   const targetSpacing = depthRange / volumeRenderingDepthSamples;
//   const targetVolume = targetSpacing ** 3;
//   return targetVolume * tsource.chunkLayout.detTransform;
// }

export function forEachVisibleVolumeRenderingChunk<
  RLayer extends MultiscaleVolumetricDataRenderLayer,
  Source extends VolumeChunkSource,
  Transformed extends TransformedSource<RLayer, Source>,
>(
  projectionParameters: ProjectionParameters,
  localPosition: Float32Array,
  volumeRenderingDepthSamples: number,
  transformedSources: readonly Transformed[],
  // Per-scale nesting: entry `i` reflects whether scale
  // `i` nests inside its immediate finer neighbor `i - 1`, and
  // entry `0` is unused but set to true.
  // `undefined` means single-scale and can be used to reflect only the target scale
  // this can be useful for a ready check for example
  nestedOctreeLevels: readonly boolean[] | undefined,
  beginScale: (
    source: Transformed,
    index: number,
    physicalSpacing: number,
    optimalSamples: number,
    clippingPlanes: Float32Array,
    histogramInformation: HistogramInformation,
  ) => void,
  callback: (
    source: Transformed,
    index: number,
    positionInChunks: vec3,
  ) => void,
): { targetScaleIndex: number; coarsestNestedScaleIndex: number } | undefined {
  if (transformedSources.length === 0) return undefined;
  const { viewMatrix, projectionMat, displayDimensionRenderInfo } =
    projectionParameters;
  const { voxelPhysicalScales } = displayDimensionRenderInfo;
  const canonicalToPhysicalScale = prod3(voxelPhysicalScales);

  const depthRange = getViewFrustumDepthRange(projectionMat);
  // Target voxel spacing in view space
  const targetViewSpacing = depthRange / volumeRenderingDepthSamples;
  // Target voxel volume in view space.
  const targetViewVolume = targetViewSpacing ** 3;
  const viewDet = mat3.determinant(mat3FromMat4(tempMat3, viewMatrix));

  // Target voxel volume in view space.
  // const targetViewVolume = getTargetVolume(transformedSources[0], projectionParameters)
  // *physicalSpacing viewDet;

  const histogramInformation: HistogramInformation = {
    spatialScales: new Map(),
    activeIndex: -1,
  };

  // Returns volume of a single voxel of source `scaleIndex` in "view" space.
  const getViewVolume = (scaleIndex: number) => {
    const tsource = transformedSources[scaleIndex];
    return Math.abs(tsource.chunkLayout.detTransform * viewDet);
  };
  // Index of high resolution source with voxel volume greater than `targetViewVolume`.
  // This allows to find the highest resolution source that is not greatly under-sampled.
  let targetScaleIndex = transformedSources.length - 1;
  for (let scaleIndex = targetScaleIndex; scaleIndex >= 0; --scaleIndex) {
    const viewVolume = getViewVolume(scaleIndex);
    const physicalSpacing = Math.cbrt(
      (viewVolume * canonicalToPhysicalScale) / viewDet,
    );
    const optimalSamples = depthRange / Math.cbrt(viewVolume);
    histogramInformation.spatialScales.set(physicalSpacing, optimalSamples);
    if (viewVolume - targetViewVolume >= 0) {
      targetScaleIndex = scaleIndex;
    }
    histogramInformation.activeIndex = targetScaleIndex;
  }

  if (DEBUG_CHUNK_LEVEL) {
    console.log(transformedSources);
    for (
      let scaleIndex = 0;
      scaleIndex < transformedSources.length;
      ++scaleIndex
    ) {
      const viewVolume = getViewVolume(scaleIndex);
      const desiredSamples = depthRange / Math.cbrt(viewVolume);
      console.log(
        `scaleIndex=${scaleIndex} viewVolume=${viewVolume} targetScaleIndex=${targetScaleIndex} actualViewVolume=${targetViewVolume}, desiredSamples=${desiredSamples}, difference=${
          viewVolume - targetViewVolume
        }`,
      );
    }
  }

  // Walk forward from targetScaleIndex through coarser scales as long as
  // each next scale nests cleanly inside its immediate finer neighbor,
  // stopping at the first `false` -- or, once there are no coarser scales
  // left, the out-of-bounds array read is `undefined` (falsy), which stops
  // the walk the same way. If nestedOctreeLevels is undefined, no fallback
  // walking is performed (single-scale mode).
  let coarsestNestedScaleIndex = targetScaleIndex;
  if (nestedOctreeLevels !== undefined) {
    while (nestedOctreeLevels[coarsestNestedScaleIndex + 1]) {
      ++coarsestNestedScaleIndex;
    }
  }

  for (
    let scaleIndex = targetScaleIndex;
    scaleIndex <= coarsestNestedScaleIndex;
    ++scaleIndex
  ) {
    const viewVolume = getViewVolume(scaleIndex);
    const physicalSpacing = Math.cbrt(
      (viewVolume * canonicalToPhysicalScale) / viewDet,
    );
    const optimalSamples = depthRange / Math.cbrt(viewVolume);
    const tsource = transformedSources[scaleIndex];
    let firstChunk = true;
    forEachVisibleVolumetricChunk(
      projectionParameters,
      localPosition,
      tsource,
      (positionInChunks, clippingPlanes) => {
        if (firstChunk) {
          beginScale(
            tsource,
            scaleIndex,
            physicalSpacing,
            optimalSamples,
            clippingPlanes,
            histogramInformation,
          );
          firstChunk = false;
        }
        callback(tsource, scaleIndex, positionInChunks);
      },
    );
  }

  return { targetScaleIndex, coarsestNestedScaleIndex };
}
