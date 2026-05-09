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

import type { ProjectionParameters } from "#src/projection_parameters.js";
import type {
  MultiscaleVolumetricDataRenderLayer,
  SliceViewChunkSource,
  SliceViewChunkSpecification,
  TransformedSource,
} from "#src/sliceview/base.js";
import { forEachVisibleVolumetricChunk } from "#src/sliceview/base.js";
import {
  getViewFrustrumVolume,
  mat3,
  mat3FromMat4,
  prod3,
} from "#src/util/geom.js";

export interface SpatiallyIndexedChunkSpecification
  extends SliceViewChunkSpecification {
  /**
   * Specifies the maximum density of items provided by this chunk source, as
   * `limit` per chunk volume.  A value of 0 disables density-based stopping for
   * that level and selects it with a draw fraction of 1.
   */
  limit: number;
}

const tempMat3 = mat3.create();

export function forEachSpatiallyIndexedScale<
  RLayer extends MultiscaleVolumetricDataRenderLayer,
  Source extends SliceViewChunkSource<SpatiallyIndexedChunkSpecification>,
  Transformed extends TransformedSource<RLayer, Source>,
>(
  projectionParameters: ProjectionParameters,
  renderScaleTarget: number,
  transformedSources: readonly Transformed[],
  callback: (
    source: Transformed,
    index: number,
    drawFraction: number,
    physicalSpacing: number,
    pixelSpacing: number,
  ) => void,
) {
  const {
    displayDimensionRenderInfo,
    viewMatrix,
    projectionMat,
    width,
    height,
  } = projectionParameters;
  const { voxelPhysicalScales } = displayDimensionRenderInfo;
  const viewDet = Math.abs(
    mat3.determinant(mat3FromMat4(tempMat3, viewMatrix)),
  );
  const canonicalToPhysicalScale = prod3(voxelPhysicalScales);
  const viewFrustrumVolume =
    (getViewFrustrumVolume(projectionMat) / viewDet) * canonicalToPhysicalScale;

  if (transformedSources.length === 0) return;
  const baseSource = transformedSources[0];
  let sourceVolume =
    Math.abs(baseSource.chunkLayout.detTransform) * canonicalToPhysicalScale;
  const { lowerClipDisplayBound, upperClipDisplayBound } = baseSource;
  for (let i = 0; i < 3; ++i) {
    sourceVolume *= upperClipDisplayBound[i] - lowerClipDisplayBound[i];
  }

  const effectiveVolume = Math.min(sourceVolume, viewFrustrumVolume);
  const viewportArea = width * height;
  const targetNumItems = viewportArea / renderScaleTarget ** 2;
  const physicalDensityTarget = targetNumItems / effectiveVolume;

  let totalPhysicalDensity = 0;
  for (
    let scaleIndex = transformedSources.length - 1;
    scaleIndex >= 0 && totalPhysicalDensity < physicalDensityTarget;
    --scaleIndex
  ) {
    const transformedSource = transformedSources[scaleIndex];
    const spec = transformedSource.source.spec;
    const { chunkLayout } = transformedSource;
    const physicalVolume =
      prod3(chunkLayout.size) *
      Math.abs(chunkLayout.detTransform) *
      canonicalToPhysicalScale;
    const { limit, rank } = spec;
    const { nonDisplayLowerClipBound, nonDisplayUpperClipBound } =
      transformedSource;
    let sliceFraction = 1;
    for (let i = 0; i < rank; ++i) {
      const b = nonDisplayUpperClipBound[i] - nonDisplayLowerClipBound[i];
      if (Number.isFinite(b)) sliceFraction /= b;
    }
    const physicalDensity = (limit * sliceFraction) / physicalVolume;
    const newTotalPhysicalDensity = totalPhysicalDensity + physicalDensity;
    const totalPhysicalSpacing = (1 / newTotalPhysicalDensity) ** (1 / 3);
    const totalPixelSpacing = Math.sqrt(
      viewportArea / (newTotalPhysicalDensity * effectiveVolume),
    );
    const desiredCount =
      ((physicalDensityTarget - totalPhysicalDensity) * physicalVolume) /
      sliceFraction;
    const drawFraction = Math.min(1, desiredCount / limit);
    callback(
      transformedSource,
      scaleIndex,
      drawFraction,
      totalPhysicalSpacing,
      totalPixelSpacing,
    );
    totalPhysicalDensity = newTotalPhysicalDensity;
  }
}

export function forEachVisibleSpatiallyIndexedChunk<
  RLayer extends MultiscaleVolumetricDataRenderLayer,
  Source extends SliceViewChunkSource<SpatiallyIndexedChunkSpecification>,
  Transformed extends TransformedSource<RLayer, Source>,
>(
  projectionParameters: ProjectionParameters,
  localPosition: Float32Array,
  renderScaleTarget: number,
  transformedSources: readonly Transformed[],
  beginScale: (source: Transformed, index: number) => void,
  callback: (
    source: Transformed,
    index: number,
    drawFraction: number,
    physicalSpacing: number,
    pixelSpacing: number,
  ) => void,
) {
  forEachSpatiallyIndexedScale(
    projectionParameters,
    renderScaleTarget,
    transformedSources,
    (
      transformedSource,
      scaleIndex,
      drawFraction,
      totalPhysicalSpacing,
      totalPixelSpacing,
    ) => {
      let firstChunk = true;
      forEachVisibleVolumetricChunk(
        projectionParameters,
        localPosition,
        transformedSource,
        () => {
          if (firstChunk) {
            beginScale(transformedSource, scaleIndex);
            firstChunk = false;
          }
          callback(
            transformedSource,
            scaleIndex,
            drawFraction,
            totalPhysicalSpacing,
            totalPixelSpacing,
          );
        },
      );
    },
  );
}
