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

import "#src/layer/segmentation/style.css";

import type { CoordinateTransformSpecification } from "#src/coordinate_transform.js";
import { emptyValidCoordinateSpace } from "#src/coordinate_transform.js";
import type { DataSourceSpecification } from "#src/datasource/index.js";
import {
  LocalDataSource,
  localEquivalencesUrl,
} from "#src/datasource/local.js";
import type { LayerActionContext, ManagedUserLayer } from "#src/layer/index.js";
import {
  LinkedLayerGroup,
  registerLayerType,
  registerLayerTypeDetector,
  registerVolumeLayerType,
  UserLayer,
} from "#src/layer/index.js";
import type { LoadedDataSubsource } from "#src/layer/layer_data_source.js";
import { layerDataSourceSpecificationFromJson } from "#src/layer/layer_data_source.js";
import * as json_keys from "#src/layer/segmentation/json_keys.js";
import { registerLayerControls } from "#src/layer/segmentation/layer_controls.js";
import {
  MeshLayer,
  MeshSource,
  MultiscaleMeshLayer,
  MultiscaleMeshSource,
} from "#src/mesh/frontend.js";
import {
  RenderScaleHistogram,
  numRenderScaleHistogramBins,
  renderScaleHistogramBinSize,
  renderScaleHistogramOrigin,
  trackableRenderScaleTarget,
} from "#src/render_scale_statistics.js";
import { getCssColor, SegmentColorHash } from "#src/segment_color.js";
import type {
  SegmentationColorGroupState,
  SegmentationDisplayState,
  SegmentationGroupState,
} from "#src/segmentation_display_state/frontend.js";
import {
  augmentSegmentId,
  bindSegmentListWidth,
  getBaseObjectColor,
  makeSegmentWidget,
  maybeAugmentSegmentId,
  registerCallbackWhenSegmentationDisplayStateChanged,
  SegmentSelectionState,
  Uint64MapEntry,
} from "#src/segmentation_display_state/frontend.js";
import type {
  PreprocessedSegmentPropertyMap,
  SegmentPropertyMap,
} from "#src/segmentation_display_state/property_map.js";
import { getPreprocessedSegmentPropertyMap } from "#src/segmentation_display_state/property_map.js";
import { LocalSegmentationGraphSource } from "#src/segmentation_graph/local.js";
import { VisibleSegmentEquivalencePolicy } from "#src/segmentation_graph/segment_id.js";
import type {
  SegmentationGraphSource,
  SegmentationGraphSourceConnection,
} from "#src/segmentation_graph/source.js";
import { SegmentationGraphSourceTab } from "#src/segmentation_graph/source.js";
import { SharedDisjointUint64Sets } from "#src/shared_disjoint_sets.js";
import { SharedWatchableValue } from "#src/shared_watchable_value.js";
import {
  PerspectiveViewSkeletonLayer,
  SkeletonLayer,
  SkeletonRenderingOptions,
  SliceViewPanelSkeletonLayer,
  PerspectiveViewSpatiallyIndexedSkeletonLayer,
  SliceViewPanelSpatiallyIndexedSkeletonLayer,
  SliceViewSpatiallyIndexedSkeletonLayer,
  SpatiallyIndexedSkeletonLayer,
  SpatiallyIndexedSkeletonSource,
  MultiscaleSpatiallyIndexedSkeletonSource,
  MultiscaleSliceViewSpatiallyIndexedSkeletonLayer,
} from "#src/skeleton/frontend.js";
import { DataType, VolumeType } from "#src/sliceview/volume/base.js";
import { MultiscaleVolumeChunkSource } from "#src/sliceview/volume/frontend.js";
import { SegmentationRenderLayer } from "#src/sliceview/volume/segmentation_renderlayer.js";
import { StatusMessage } from "#src/status.js";
import { trackableAlphaValue } from "#src/trackable_alpha.js";
import { TrackableBoolean } from "#src/trackable_boolean.js";
import { trackableFiniteFloat } from "#src/trackable_finite_float.js";
import type {
  TrackableValueInterface,
  WatchableValueInterface,
} from "#src/trackable_value.js";
import {
  IndirectTrackableValue,
  IndirectWatchableValue,
  makeCachedDerivedWatchableValue,
  makeCachedLazyDerivedWatchableValue,
  observeWatchable,
  registerNestedSync,
  TrackableValue,
  WatchableValue,
} from "#src/trackable_value.js";
import { UserLayerWithAnnotationsMixin } from "#src/ui/annotations.js";
import { SegmentDisplayTab } from "#src/ui/segment_list.js";
import { registerSegmentSelectTools } from "#src/ui/segment_select_tools.js";
import { registerSegmentSplitMergeTools } from "#src/ui/segment_split_merge_tools.js";
import { DisplayOptionsTab } from "#src/ui/segmentation_display_options_tab.js";
import { Uint64Map } from "#src/uint64_map.js";
import { Uint64OrderedSet } from "#src/uint64_ordered_set.js";
import { Uint64Set } from "#src/uint64_set.js";
import { gatherUpdate } from "#src/util/array.js";
import {
  packColor,
  parseRGBColorSpecification,
  serializeColor,
  TrackableOptionalRGB,
  unpackRGB,
} from "#src/util/color.js";
import type { Borrowed, Owned } from "#src/util/disposable.js";
import { RefCounted } from "#src/util/disposable.js";
import type { vec3, vec4 } from "#src/util/geom.js";
import {
  parseArray,
  parseUint64,
  verifyFiniteNonNegativeFloat,
  verifyNonnegativeInt,
  verifyObjectAsMap,
  verifyOptionalObjectProperty,
  verifyString,
} from "#src/util/json.js";
import { Signal } from "#src/util/signal.js";
import { makeWatchableShaderError } from "#src/webgl/dynamic_shader.js";
import type { DependentViewContext } from "#src/widget/dependent_view_widget.js";
import { registerLayerShaderControlsTool } from "#src/widget/shader_controls.js";

const MAX_LAYER_BAR_UI_INDICATOR_COLORS = 6;

export class SegmentationUserLayerGroupState
  extends RefCounted
  implements SegmentationGroupState {
  specificationChanged = new Signal();
  constructor(public layer: SegmentationUserLayer) {
    super();
    const { specificationChanged } = this;
    this.hideSegmentZero.changed.add(specificationChanged.dispatch);
    this.segmentQuery.changed.add(specificationChanged.dispatch);

    const { selectedSegments } = this;
    const visibleSegments = (this.visibleSegments = this.registerDisposer(
      Uint64Set.makeWithCounterpart(layer.manager.rpc),
    ));
    this.segmentEquivalences = this.registerDisposer(
      SharedDisjointUint64Sets.makeWithCounterpart(
        layer.manager.rpc,
        layer.registerDisposer(
          makeCachedDerivedWatchableValue(
            (x) =>
              x?.visibleSegmentEquivalencePolicy ||
              VisibleSegmentEquivalencePolicy.MIN_REPRESENTATIVE,
            [this.graph],
          ),
        ),
      ),
    );

    this.temporaryVisibleSegments = layer.registerDisposer(
      Uint64Set.makeWithCounterpart(layer.manager.rpc),
    );
    this.temporarySegmentEquivalences = layer.registerDisposer(
      SharedDisjointUint64Sets.makeWithCounterpart(
        layer.manager.rpc,
        this.segmentEquivalences.disjointSets.visibleSegmentEquivalencePolicy,
      ),
    );
    this.useTemporaryVisibleSegments = layer.registerDisposer(
      SharedWatchableValue.make(layer.manager.rpc, false),
    );
    this.useTemporarySegmentEquivalences = layer.registerDisposer(
      SharedWatchableValue.make(layer.manager.rpc, false),
    );

    visibleSegments.changed.add(specificationChanged.dispatch);
    selectedSegments.changed.add(specificationChanged.dispatch);
    selectedSegments.changed.add((x, add) => {
      if (!add) {
        if (x) {
          visibleSegments.delete(x);
        } else {
          visibleSegments.clear();
        }
      }
    });
    visibleSegments.changed.add((x, add) => {
      if (add) {
        if (x) {
          selectedSegments.add(x);
        }
      }
    });
  }

  restoreState(specification: unknown) {
    verifyOptionalObjectProperty(
      specification,
      json_keys.HIDE_SEGMENT_ZERO_JSON_KEY,
      (value) => this.hideSegmentZero.restoreState(value),
    );
    verifyOptionalObjectProperty(
      specification,
      json_keys.EQUIVALENCES_JSON_KEY,
      (value) => {
        this.localGraph.restoreState(value);
      },
    );

    verifyOptionalObjectProperty(
      specification,
      json_keys.SEGMENTS_JSON_KEY,
      (segmentsValue) => {
        const { segmentEquivalences, selectedSegments, visibleSegments } = this;
        parseArray(segmentsValue, (value) => {
          let stringValue = String(value);
          const hidden = stringValue.startsWith("!");
          if (hidden) {
            stringValue = stringValue.substring(1);
          }
          const id = parseUint64(stringValue);
          const segmentId = segmentEquivalences.get(id);
          selectedSegments.add(segmentId);
          if (!hidden) {
            visibleSegments.add(segmentId);
          }
        });
      },
    );
    verifyOptionalObjectProperty(
      specification,
      json_keys.SEGMENT_QUERY_JSON_KEY,
      (value) => this.segmentQuery.restoreState(value),
    );
  }

  toJSON() {
    const x: any = {};
    x[json_keys.HIDE_SEGMENT_ZERO_JSON_KEY] = this.hideSegmentZero.toJSON();
    const { selectedSegments, visibleSegments } = this;
    if (selectedSegments.size > 0) {
      x[json_keys.SEGMENTS_JSON_KEY] = [...selectedSegments].map((segment) => {
        if (visibleSegments.has(segment)) {
          return segment.toString();
        }
        return "!" + segment.toString();
      });
    } else {
      x[json_keys.SEGMENTS_JSON_KEY] = [];
    }
    const { segmentEquivalences } = this;
    if (this.localSegmentEquivalences && segmentEquivalences.size > 0) {
      x[json_keys.EQUIVALENCES_JSON_KEY] = segmentEquivalences.toJSON();
    }
    x[json_keys.SEGMENT_QUERY_JSON_KEY] = this.segmentQuery.toJSON();
    return x;
  }

  assignFrom(other: SegmentationUserLayerGroupState) {
    this.maxIdLength.value = other.maxIdLength.value;
    this.hideSegmentZero.value = other.hideSegmentZero.value;
    this.selectedSegments.assignFrom(other.selectedSegments);
    this.visibleSegments.assignFrom(other.visibleSegments);
    this.segmentEquivalences.assignFrom(other.segmentEquivalences);
  }

  localGraph = new LocalSegmentationGraphSource();
  visibleSegments: Uint64Set;
  selectedSegments = this.registerDisposer(new Uint64OrderedSet());

  segmentPropertyMap = new WatchableValue<
    PreprocessedSegmentPropertyMap | undefined
  >(undefined);
  graph = new WatchableValue<SegmentationGraphSource | undefined>(undefined);
  segmentEquivalences: SharedDisjointUint64Sets;
  localSegmentEquivalences = false;
  maxIdLength = new WatchableValue(1);
  hideSegmentZero = new TrackableBoolean(true, true);
  segmentQuery = new TrackableValue<string>("", verifyString);

  temporaryVisibleSegments: Uint64Set;
  temporarySegmentEquivalences: SharedDisjointUint64Sets;
  useTemporaryVisibleSegments: SharedWatchableValue<boolean>;
  useTemporarySegmentEquivalences: SharedWatchableValue<boolean>;
}

export class SegmentationUserLayerColorGroupState
  extends RefCounted
  implements SegmentationColorGroupState {
  specificationChanged = new Signal();
  constructor(public layer: SegmentationUserLayer) {
    super();
    const { specificationChanged } = this;
    this.segmentColorHash.changed.add(specificationChanged.dispatch);
    this.segmentStatedColors.changed.add(specificationChanged.dispatch);
    this.tempSegmentStatedColors2d.changed.add(specificationChanged.dispatch);
    this.segmentDefaultColor.changed.add(specificationChanged.dispatch);
    this.tempSegmentDefaultColor2d.changed.add(specificationChanged.dispatch);
    this.highlightColor.changed.add(specificationChanged.dispatch);
  }

  restoreState(specification: unknown) {
    verifyOptionalObjectProperty(
      specification,
      json_keys.COLOR_SEED_JSON_KEY,
      (value) => this.segmentColorHash.restoreState(value),
    );
    verifyOptionalObjectProperty(
      specification,
      json_keys.SEGMENT_DEFAULT_COLOR_JSON_KEY,
      (value) => this.segmentDefaultColor.restoreState(value),
    );
    verifyOptionalObjectProperty(
      specification,
      json_keys.SEGMENT_STATED_COLORS_JSON_KEY,
      (y) => {
        const result = verifyObjectAsMap(y, (x) =>
          parseRGBColorSpecification(String(x)),
        );
        for (const [idStr, colorVec] of result) {
          const id = parseUint64(idStr);
          const color = BigInt(packColor(colorVec));
          this.segmentStatedColors.set(id, color);
        }
      },
    );
  }

  toJSON() {
    const x: any = {};
    x[json_keys.COLOR_SEED_JSON_KEY] = this.segmentColorHash.toJSON();
    x[json_keys.SEGMENT_DEFAULT_COLOR_JSON_KEY] =
      this.segmentDefaultColor.toJSON();
    const { segmentStatedColors } = this;
    if (segmentStatedColors.size > 0) {
      const j: any = (x[json_keys.SEGMENT_STATED_COLORS_JSON_KEY] = {});
      for (const [key, value] of segmentStatedColors) {
        j[key.toString()] = serializeColor(unpackRGB(Number(value)));
      }
    }
    return x;
  }

  assignFrom(other: SegmentationUserLayerColorGroupState) {
    this.segmentColorHash.value = other.segmentColorHash.value;
    this.segmentStatedColors.assignFrom(other.segmentStatedColors);
    this.tempSegmentStatedColors2d.assignFrom(other.tempSegmentStatedColors2d);
    this.segmentDefaultColor.value = other.segmentDefaultColor.value;
    this.highlightColor.value = other.highlightColor.value;
  }

  segmentColorHash = SegmentColorHash.getDefault();
  segmentStatedColors = this.registerDisposer(new Uint64Map());
  tempSegmentStatedColors2d = this.registerDisposer(new Uint64Map());
  segmentDefaultColor = new TrackableOptionalRGB();
  tempSegmentDefaultColor2d = new WatchableValue<vec3 | vec4 | undefined>(
    undefined,
  );
  highlightColor = new WatchableValue<vec4 | undefined>(undefined);
}

class LinkedSegmentationGroupState<
  State extends
  | SegmentationUserLayerGroupState
  | SegmentationUserLayerColorGroupState,
>
  extends RefCounted
  implements WatchableValueInterface<State> {
  private curRoot: SegmentationUserLayer | undefined;
  private curGroupState: Owned<State> | undefined;
  get changed() {
    return this.linkedGroup.root.changed;
  }
  get value() {
    const root = this.linkedGroup.root.value as SegmentationUserLayer;
    if (root !== this.curRoot) {
      this.curRoot = root;
      const groupState = root.displayState[this.propertyName] as State;
      if (root === this.linkedGroup.layer) {
        const { curGroupState } = this;
        if (curGroupState !== undefined) {
          groupState.assignFrom(curGroupState as any);
          curGroupState.dispose();
        }
      }
      this.curGroupState = groupState.addRef() as State;
    }
    return this.curGroupState!;
  }
  disposed() {
    this.curGroupState?.dispose();
  }
  constructor(
    public linkedGroup: LinkedLayerGroup,
    private propertyName: State extends SegmentationUserLayerGroupState
      ? "originalSegmentationGroupState"
      : "originalSegmentationColorGroupState",
  ) {
    super();
    this.value;
  }
}

type SpatialSkeletonGridSize = { x: number; y: number; z: number };
type SpatialSkeletonGridLevel = { size: SpatialSkeletonGridSize; lod: number };

function getSpatialSkeletonGridSpacing(size: SpatialSkeletonGridSize) {
  return Math.min(size.x, size.y, size.z);
}

function buildSpatialSkeletonGridLevels(
  gridSizes: SpatialSkeletonGridSize[],
): SpatialSkeletonGridLevel[] {
  if (gridSizes.length === 0) return [];
  const lastIndex = gridSizes.length - 1;
  return gridSizes.map((size, index) => ({
    size,
    lod: lastIndex === 0 ? 0 : index / lastIndex,
  }));
}

function findClosestSpatialSkeletonGridLevel(
  levels: SpatialSkeletonGridLevel[],
  lod: number,
): number {
  let bestIndex = 0;
  let bestDistance = Number.POSITIVE_INFINITY;
  for (let i = 0; i < levels.length; ++i) {
    const distance = Math.abs(levels[i].lod - lod);
    if (distance < bestDistance) {
      bestDistance = distance;
      bestIndex = i;
    }
  }
  return bestIndex;
}

function findClosestSpatialSkeletonGridLevelBySpacing(
  levels: SpatialSkeletonGridLevel[],
  spacing: number,
): number {
  let bestIndex = 0;
  let bestDistance = Number.POSITIVE_INFINITY;
  for (let i = 0; i < levels.length; ++i) {
    const gridSpacing = getSpatialSkeletonGridSpacing(levels[i].size);
    const distance = Math.abs(gridSpacing - spacing);
    if (distance < bestDistance) {
      bestDistance = distance;
      bestIndex = i;
    }
  }
  return bestIndex;
}

function getSpatialSkeletonGridHistogramConfig(
  levels: SpatialSkeletonGridLevel[],
) {
  if (levels.length === 0) {
    return {
      origin: renderScaleHistogramOrigin,
      binSize: renderScaleHistogramBinSize,
    };
  }
  const logSpacings: number[] = [];
  let minLogSpacing = Number.POSITIVE_INFINITY;
  let maxLogSpacing = Number.NEGATIVE_INFINITY;
  for (const level of levels) {
    const spacing = Math.max(getSpatialSkeletonGridSpacing(level.size), 1e-6);
    const logSpacing = Math.log2(spacing);
    logSpacings.push(logSpacing);
    minLogSpacing = Math.min(minLogSpacing, logSpacing);
    maxLogSpacing = Math.max(maxLogSpacing, logSpacing);
  }
  if (!Number.isFinite(minLogSpacing) || !Number.isFinite(maxLogSpacing)) {
    return {
      origin: renderScaleHistogramOrigin,
      binSize: renderScaleHistogramBinSize,
    };
  }
  logSpacings.sort((a, b) => a - b);
  let minDelta = Number.POSITIVE_INFINITY;
  for (let i = 1; i < logSpacings.length; ++i) {
    const delta = logSpacings[i] - logSpacings[i - 1];
    if (delta > 0) minDelta = Math.min(minDelta, delta);
  }
  const span = maxLogSpacing - minLogSpacing;
  const minBinSizeForCoverage = span / Math.max(numRenderScaleHistogramBins - 4, 1);
  const lowerBound = Math.max(minBinSizeForCoverage, 0.05);
  let binSize = lowerBound;
  if (Number.isFinite(minDelta)) {
    const maxBinSizeForDistinctBars = minDelta * 0.9;
    if (maxBinSizeForDistinctBars >= lowerBound) {
      binSize = maxBinSizeForDistinctBars;
    }
  }
  if (!Number.isFinite(binSize) || binSize <= 0) {
    binSize = renderScaleHistogramBinSize;
  }

  const range = numRenderScaleHistogramBins * binSize;
  const desiredPadding = binSize * 2;
  const minOrigin = maxLogSpacing + desiredPadding - range;
  const maxOrigin = minLogSpacing - desiredPadding;
  const centeredOrigin = (minLogSpacing + maxLogSpacing - range) / 2;
  const clampedOrigin = Math.min(
    Math.max(centeredOrigin, minOrigin),
    maxOrigin,
  );
  const roundedBinSize = Math.max(binSize, 1e-3);
  const roundedOrigin =
    Math.round(clampedOrigin / roundedBinSize) * roundedBinSize;
  return { origin: roundedOrigin, binSize: roundedBinSize };
}

class SegmentationUserLayerDisplayState implements SegmentationDisplayState {
  constructor(public layer: SegmentationUserLayer) {
    // Even though `SegmentationUserLayer` assigns this to its `displayState` property, redundantly
    // assign it here first in order to allow it to be accessed by `segmentationGroupState`.
    layer.displayState = this;

    this.linkedSegmentationGroup = layer.registerDisposer(
      new LinkedLayerGroup(
        layer.manager.rootLayers,
        layer,
        (userLayer) => userLayer instanceof SegmentationUserLayer,
        (userLayer: SegmentationUserLayer) =>
          userLayer.displayState.linkedSegmentationGroup,
      ),
    );

    this.linkedSegmentationColorGroup = this.layer.registerDisposer(
      new LinkedLayerGroup(
        layer.manager.rootLayers,
        layer,
        (userLayer) => userLayer instanceof SegmentationUserLayer,
        (userLayer: SegmentationUserLayer) =>
          userLayer.displayState.linkedSegmentationColorGroup,
      ),
    );

    this.originalSegmentationGroupState = layer.registerDisposer(
      new SegmentationUserLayerGroupState(layer),
    );

    this.originalSegmentationColorGroupState = layer.registerDisposer(
      new SegmentationUserLayerColorGroupState(layer),
    );

    this.transparentPickEnabled = layer.pick;

    this.useTempSegmentStatedColors2d = layer.registerDisposer(
      SharedWatchableValue.make(layer.manager.rpc, false),
    );

    this.segmentationGroupState = this.layer.registerDisposer(
      new LinkedSegmentationGroupState<SegmentationUserLayerGroupState>(
        this.linkedSegmentationGroup,
        "originalSegmentationGroupState",
      ),
    );
    this.segmentationColorGroupState = this.layer.registerDisposer(
      new LinkedSegmentationGroupState<SegmentationUserLayerColorGroupState>(
        this.linkedSegmentationColorGroup,
        "originalSegmentationColorGroupState",
      ),
    );

    this.selectSegment = layer.selectSegment;
    this.filterBySegmentLabel = layer.filterBySegmentLabel;

    this.hideSegmentZero = this.layer.registerDisposer(
      new IndirectWatchableValue(
        this.segmentationGroupState,
        (group) => group.hideSegmentZero,
      ),
    );
    this.segmentColorHash = this.layer.registerDisposer(
      new IndirectTrackableValue(
        this.segmentationColorGroupState,
        (group) => group.segmentColorHash,
      ),
    );
    this.segmentStatedColors = this.layer.registerDisposer(
      new IndirectTrackableValue(
        this.segmentationColorGroupState,
        (group) => group.segmentStatedColors,
      ),
    );
    this.tempSegmentStatedColors2d = this.layer.registerDisposer(
      new IndirectTrackableValue(
        this.segmentationColorGroupState,
        (group) => group.tempSegmentStatedColors2d,
      ),
    );
    this.segmentDefaultColor = this.layer.registerDisposer(
      new IndirectTrackableValue(
        this.segmentationColorGroupState,
        (group) => group.segmentDefaultColor,
      ),
    );
    this.tempSegmentDefaultColor2d = this.layer.registerDisposer(
      new IndirectTrackableValue(
        this.segmentationColorGroupState,
        (group) => group.tempSegmentDefaultColor2d,
      ),
    );
    this.highlightColor = this.layer.registerDisposer(
      new IndirectTrackableValue(
        this.segmentationColorGroupState,
        (group) => group.highlightColor,
      ),
    );
    this.segmentQuery = this.layer.registerDisposer(
      new IndirectWatchableValue(
        this.segmentationGroupState,
        (group) => group.segmentQuery,
      ),
    );
    this.segmentPropertyMap = this.layer.registerDisposer(
      new IndirectWatchableValue(
        this.segmentationGroupState,
        (group) => group.segmentPropertyMap,
      ),
    );

    this.spatialSkeletonGridResolutionTarget2d.changed.add(() => {
      if (this.suppressSpatialSkeletonGridResolutionTarget2d) return;
      this.spatialSkeletonGridResolutionTarget2dExplicit = true;
      this.applySpatialSkeletonGridResolutionTarget("2d");
    });
    this.spatialSkeletonGridResolutionTarget3d.changed.add(() => {
      if (this.suppressSpatialSkeletonGridResolutionTarget3d) return;
      this.spatialSkeletonGridResolutionTarget3dExplicit = true;
      this.applySpatialSkeletonGridResolutionTarget("3d");
    });
    this.spatialSkeletonGridResolutionRelative2d.changed.add(() => {
      const nextRelative = this.spatialSkeletonGridResolutionRelative2d.value;
      if (nextRelative !== this.lastSpatialSkeletonGridResolutionRelative2d) {
        const pixelSize = Math.max(this.spatialSkeletonGridPixelSize2d.value, 1e-6);
        const currentTarget = this.spatialSkeletonGridResolutionTarget2d.value;
        const adjustedTarget = nextRelative
          ? currentTarget / pixelSize
          : currentTarget * pixelSize;
        this.suppressSpatialSkeletonGridResolutionTarget2d = true;
        this.spatialSkeletonGridResolutionTarget2d.value = adjustedTarget;
        this.suppressSpatialSkeletonGridResolutionTarget2d = false;
        this.spatialSkeletonGridResolutionTarget2dExplicit = true;
        this.lastSpatialSkeletonGridResolutionRelative2d = nextRelative;
      }
      this.applySpatialSkeletonGridResolutionTarget("2d");
    });
    this.spatialSkeletonGridResolutionRelative3d.changed.add(() => {
      const nextRelative = this.spatialSkeletonGridResolutionRelative3d.value;
      if (nextRelative !== this.lastSpatialSkeletonGridResolutionRelative3d) {
        const pixelSize = Math.max(this.spatialSkeletonGridPixelSize3d.value, 1e-6);
        const currentTarget = this.spatialSkeletonGridResolutionTarget3d.value;
        const adjustedTarget = nextRelative
          ? currentTarget / pixelSize
          : currentTarget * pixelSize;
        this.suppressSpatialSkeletonGridResolutionTarget3d = true;
        this.spatialSkeletonGridResolutionTarget3d.value = adjustedTarget;
        this.suppressSpatialSkeletonGridResolutionTarget3d = false;
        this.spatialSkeletonGridResolutionTarget3dExplicit = true;
        this.lastSpatialSkeletonGridResolutionRelative3d = nextRelative;
      }
      this.applySpatialSkeletonGridResolutionTarget("3d");
    });
    this.spatialSkeletonGridPixelSize2d.changed.add(() => {
      if (this.spatialSkeletonGridResolutionRelative2d.value) {
        this.applySpatialSkeletonGridResolutionTarget("2d");
      }
    });
    this.spatialSkeletonGridPixelSize3d.changed.add(() => {
      if (this.spatialSkeletonGridResolutionRelative3d.value) {
        this.applySpatialSkeletonGridResolutionTarget("3d");
      }
    });
    this.spatialSkeletonGridLevel2d.changed.add(() => {
      if (this.suppressSpatialSkeletonGridLevel2d) return;
      if (this.spatialSkeletonGridLevels.value.length === 0) return;
      this.setSpatialSkeletonGridLevel(
        "2d",
        this.spatialSkeletonGridLevel2d.value,
        true,
      );
    });
    this.spatialSkeletonGridLevel3d.changed.add(() => {
      if (this.suppressSpatialSkeletonGridLevel3d) return;
      if (this.spatialSkeletonGridLevels.value.length === 0) return;
      this.setSpatialSkeletonGridLevel(
        "3d",
        this.spatialSkeletonGridLevel3d.value,
        true,
      );
    });
  }

  segmentSelectionState = new SegmentSelectionState();
  selectedAlpha = trackableAlphaValue(0.5);
  saturation = trackableAlphaValue(1.0);
  notSelectedAlpha = trackableAlphaValue(0);
  hoverHighlight = new TrackableBoolean(true, true);
  silhouetteRendering = new TrackableValue<number>(
    0,
    verifyFiniteNonNegativeFloat,
    0,
  );
  objectAlpha = trackableAlphaValue(1.0);
  hiddenObjectAlpha = trackableAlphaValue(0.5);
  skeletonLod = trackableFiniteFloat(0.0);
  spatialSkeletonGridLevel2d = new TrackableValue<number>(
    0,
    verifyNonnegativeInt,
    0,
  );
  spatialSkeletonGridLevel3d = new TrackableValue<number>(
    0,
    verifyNonnegativeInt,
    0,
  );
  spatialSkeletonGridLevels = new WatchableValue<SpatialSkeletonGridLevel[]>([]);
  spatialSkeletonGridResolutionTarget2d = new TrackableValue<number>(
    1,
    verifyFiniteNonNegativeFloat,
    1,
  );
  spatialSkeletonGridResolutionTarget3d = new TrackableValue<number>(
    1,
    verifyFiniteNonNegativeFloat,
    1,
  );
  spatialSkeletonGridResolutionRelative2d = new TrackableBoolean(false, false);
  spatialSkeletonGridResolutionRelative3d = new TrackableBoolean(false, false);
  spatialSkeletonGridPixelSize2d = new WatchableValue<number>(1);
  spatialSkeletonGridPixelSize3d = new WatchableValue<number>(1);
  spatialSkeletonGridRenderScaleHistogram2d = new RenderScaleHistogram();
  spatialSkeletonGridRenderScaleHistogram3d = new RenderScaleHistogram();
  spatialSkeletonLod2d = new WatchableValue<number>(0);
  private spatialSkeletonGridResolutionTarget2dExplicit = false;
  private spatialSkeletonGridResolutionTarget3dExplicit = false;
  private spatialSkeletonGridLevel2dExplicit = false;
  private spatialSkeletonGridLevel3dExplicit = false;
  private suppressSpatialSkeletonGridLevel2d = false;
  private suppressSpatialSkeletonGridLevel3d = false;
  private suppressSpatialSkeletonGridResolutionTarget2d = false;
  private suppressSpatialSkeletonGridResolutionTarget3d = false;
  private lastSpatialSkeletonGridResolutionRelative2d = false;
  private lastSpatialSkeletonGridResolutionRelative3d = false;
  ignoreNullVisibleSet = new TrackableBoolean(true, true);
  skeletonRenderingOptions = new SkeletonRenderingOptions();
  shaderError = makeWatchableShaderError();
  renderScaleHistogram = new RenderScaleHistogram();
  renderScaleTarget = trackableRenderScaleTarget(1);
  selectSegment: (id: bigint, pin: boolean | "toggle") => void;
  transparentPickEnabled: TrackableBoolean;
  baseSegmentColoring = new TrackableBoolean(false, false);
  baseSegmentHighlighting = new TrackableBoolean(false, false);
  useTempSegmentStatedColors2d: SharedWatchableValue<boolean>;
  hasVolume = new TrackableBoolean(false, false);

  filterBySegmentLabel: (id: bigint) => void;

  moveToSegment = (id: bigint) => {
    this.layer.moveToSegment(id);
  };

  setSpatialSkeletonGridSizes(gridSizes: SpatialSkeletonGridSize[]) {
    const sortedSizes = [...gridSizes].sort(
      (a, b) => Math.min(b.x, b.y, b.z) - Math.min(a.x, a.y, a.z),
    );
    const levels = buildSpatialSkeletonGridLevels(sortedSizes);
    const { origin: histogramOrigin, binSize: histogramBinSize } =
      getSpatialSkeletonGridHistogramConfig(levels);
    if (
      this.spatialSkeletonGridRenderScaleHistogram2d.logScaleOrigin !==
        histogramOrigin ||
      this.spatialSkeletonGridRenderScaleHistogram2d.logScaleBinSize !==
        histogramBinSize
    ) {
      this.spatialSkeletonGridRenderScaleHistogram2d.logScaleOrigin =
        histogramOrigin;
      this.spatialSkeletonGridRenderScaleHistogram2d.logScaleBinSize =
        histogramBinSize;
      this.spatialSkeletonGridRenderScaleHistogram2d.changed.dispatch();
    }
    if (
      this.spatialSkeletonGridRenderScaleHistogram3d.logScaleOrigin !==
        histogramOrigin ||
      this.spatialSkeletonGridRenderScaleHistogram3d.logScaleBinSize !==
        histogramBinSize
    ) {
      this.spatialSkeletonGridRenderScaleHistogram3d.logScaleOrigin =
        histogramOrigin;
      this.spatialSkeletonGridRenderScaleHistogram3d.logScaleBinSize =
        histogramBinSize;
      this.spatialSkeletonGridRenderScaleHistogram3d.changed.dispatch();
    }
    this.spatialSkeletonGridLevels.value = levels;
    if (levels.length === 0) return;
    const target3dIndex = this.spatialSkeletonGridResolutionTarget3dExplicit
      ? findClosestSpatialSkeletonGridLevelBySpacing(
          levels,
          this.getSpatialSkeletonGridTargetSpacing("3d"),
        )
      : this.spatialSkeletonGridLevel3dExplicit
        ? this.spatialSkeletonGridLevel3d.value
        : findClosestSpatialSkeletonGridLevel(levels, this.skeletonLod.value);
    const resolved3dIndex = this.setSpatialSkeletonGridLevel(
      "3d",
      target3dIndex,
      this.spatialSkeletonGridResolutionTarget3dExplicit ||
        this.spatialSkeletonGridLevel3dExplicit,
    );
    const target2dIndex = this.spatialSkeletonGridResolutionTarget2dExplicit
      ? findClosestSpatialSkeletonGridLevelBySpacing(
          levels,
          this.getSpatialSkeletonGridTargetSpacing("2d"),
        )
      : this.spatialSkeletonGridLevel2dExplicit
        ? this.spatialSkeletonGridLevel2d.value
        : resolved3dIndex;
    this.setSpatialSkeletonGridLevel(
      "2d",
      target2dIndex,
      this.spatialSkeletonGridResolutionTarget2dExplicit ||
        this.spatialSkeletonGridLevel2dExplicit,
    );
    if (!this.spatialSkeletonGridResolutionTarget3dExplicit) {
      const spacing = getSpatialSkeletonGridSpacing(
        levels[Math.min(resolved3dIndex, levels.length - 1)].size,
      );
      const targetValue = this.spatialSkeletonGridResolutionRelative3d.value
        ? spacing / Math.max(this.spatialSkeletonGridPixelSize3d.value, 1e-6)
        : spacing;
      this.suppressSpatialSkeletonGridResolutionTarget3d = true;
      this.spatialSkeletonGridResolutionTarget3d.value = targetValue;
      this.suppressSpatialSkeletonGridResolutionTarget3d = false;
    }
    if (!this.spatialSkeletonGridResolutionTarget2dExplicit) {
      const resolved2dIndex = Math.min(
        Math.max(target2dIndex, 0),
        levels.length - 1,
      );
      const spacing = getSpatialSkeletonGridSpacing(
        levels[resolved2dIndex].size,
      );
      const targetValue = this.spatialSkeletonGridResolutionRelative2d.value
        ? spacing / Math.max(this.spatialSkeletonGridPixelSize2d.value, 1e-6)
        : spacing;
      this.suppressSpatialSkeletonGridResolutionTarget2d = true;
      this.spatialSkeletonGridResolutionTarget2d.value = targetValue;
      this.suppressSpatialSkeletonGridResolutionTarget2d = false;
    }
  }

  applySpatialSkeletonGridLevel2dFromSpec(value: any) {
    if (value !== undefined && !this.spatialSkeletonGridResolutionTarget2dExplicit) {
      this.spatialSkeletonGridLevel2d.restoreState(value);
      this.spatialSkeletonGridLevel2dExplicit = true;
      if (this.spatialSkeletonGridLevels.value.length > 0) {
        this.setSpatialSkeletonGridLevel(
          "2d",
          this.spatialSkeletonGridLevel2d.value,
          true,
        );
        if (!this.spatialSkeletonGridResolutionTarget2dExplicit) {
          const spacing = getSpatialSkeletonGridSpacing(
            this.spatialSkeletonGridLevels.value[
              Math.min(
                this.spatialSkeletonGridLevel2d.value,
                this.spatialSkeletonGridLevels.value.length - 1,
              )
            ].size,
          );
          const targetValue = this.spatialSkeletonGridResolutionRelative2d.value
            ? spacing / Math.max(this.spatialSkeletonGridPixelSize2d.value, 1e-6)
            : spacing;
          this.suppressSpatialSkeletonGridResolutionTarget2d = true;
          this.spatialSkeletonGridResolutionTarget2d.value = targetValue;
          this.suppressSpatialSkeletonGridResolutionTarget2d = false;
        }
      }
    }
  }

  applySpatialSkeletonGridLevel3dFromSpec(value: any) {
    if (value !== undefined && !this.spatialSkeletonGridResolutionTarget3dExplicit) {
      this.spatialSkeletonGridLevel3d.restoreState(value);
      this.spatialSkeletonGridLevel3dExplicit = true;
      if (this.spatialSkeletonGridLevels.value.length > 0) {
        this.setSpatialSkeletonGridLevel(
          "3d",
          this.spatialSkeletonGridLevel3d.value,
          true,
        );
        if (!this.spatialSkeletonGridResolutionTarget3dExplicit) {
          const spacing = getSpatialSkeletonGridSpacing(
            this.spatialSkeletonGridLevels.value[
              Math.min(
                this.spatialSkeletonGridLevel3d.value,
                this.spatialSkeletonGridLevels.value.length - 1,
              )
            ].size,
          );
          const targetValue = this.spatialSkeletonGridResolutionRelative3d.value
            ? spacing / Math.max(this.spatialSkeletonGridPixelSize3d.value, 1e-6)
            : spacing;
          this.suppressSpatialSkeletonGridResolutionTarget3d = true;
          this.spatialSkeletonGridResolutionTarget3d.value = targetValue;
          this.suppressSpatialSkeletonGridResolutionTarget3d = false;
        }
      }
    }
  }

  applySpatialSkeletonGridResolutionTarget2dFromSpec(value: any) {
    if (value !== undefined) {
      this.suppressSpatialSkeletonGridResolutionTarget2d = true;
      this.spatialSkeletonGridResolutionTarget2d.restoreState(value);
      this.suppressSpatialSkeletonGridResolutionTarget2d = false;
      this.spatialSkeletonGridResolutionTarget2dExplicit = true;
      if (this.spatialSkeletonGridLevels.value.length > 0) {
        this.applySpatialSkeletonGridResolutionTarget("2d");
      }
    }
  }

  applySpatialSkeletonGridResolutionTarget3dFromSpec(value: any) {
    if (value !== undefined) {
      this.suppressSpatialSkeletonGridResolutionTarget3d = true;
      this.spatialSkeletonGridResolutionTarget3d.restoreState(value);
      this.suppressSpatialSkeletonGridResolutionTarget3d = false;
      this.spatialSkeletonGridResolutionTarget3dExplicit = true;
      if (this.spatialSkeletonGridLevels.value.length > 0) {
        this.applySpatialSkeletonGridResolutionTarget("3d");
      }
    }
  }

  private getSpatialSkeletonGridTargetSpacing(kind: "2d" | "3d") {
    const target =
      kind === "2d"
        ? this.spatialSkeletonGridResolutionTarget2d.value
        : this.spatialSkeletonGridResolutionTarget3d.value;
    const isRelative =
      kind === "2d"
        ? this.spatialSkeletonGridResolutionRelative2d.value
        : this.spatialSkeletonGridResolutionRelative3d.value;
    const pixelSize =
      kind === "2d"
        ? this.spatialSkeletonGridPixelSize2d.value
        : this.spatialSkeletonGridPixelSize3d.value;
    return isRelative ? target * pixelSize : target;
  }

  private applySpatialSkeletonGridResolutionTarget(kind: "2d" | "3d") {
    const levels = this.spatialSkeletonGridLevels.value;
    if (levels.length === 0) return;
    const targetSpacing = this.getSpatialSkeletonGridTargetSpacing(kind);
    const index = findClosestSpatialSkeletonGridLevelBySpacing(
      levels,
      targetSpacing,
    );
    const markExplicit =
      kind === "2d"
        ? this.spatialSkeletonGridResolutionTarget2dExplicit
        : this.spatialSkeletonGridResolutionTarget3dExplicit;
    this.setSpatialSkeletonGridLevel(kind, index, markExplicit);
  }

  private setSpatialSkeletonGridLevel(
    kind: "2d" | "3d",
    index: number,
    markExplicit: boolean,
  ) {
    const levels = this.spatialSkeletonGridLevels.value;
    if (levels.length === 0) return 0;
    const clampedIndex = Math.min(Math.max(index, 0), levels.length - 1);
    if (kind === "2d") {
      if (markExplicit) this.spatialSkeletonGridLevel2dExplicit = true;
      this.suppressSpatialSkeletonGridLevel2d = true;
      this.spatialSkeletonGridLevel2d.value = clampedIndex;
      this.suppressSpatialSkeletonGridLevel2d = false;
      const nextLod = levels[clampedIndex].lod;
      if (this.spatialSkeletonLod2d.value !== nextLod) {
        this.spatialSkeletonLod2d.value = nextLod;
      }
      return clampedIndex;
    }
    if (markExplicit) this.spatialSkeletonGridLevel3dExplicit = true;
    this.suppressSpatialSkeletonGridLevel3d = true;
    this.spatialSkeletonGridLevel3d.value = clampedIndex;
    this.suppressSpatialSkeletonGridLevel3d = false;
    const nextLod = levels[clampedIndex].lod;
    if (this.skeletonLod.value !== nextLod) {
      this.skeletonLod.value = nextLod;
    }
    return clampedIndex;
  }

  linkedSegmentationGroup: LinkedLayerGroup;
  linkedSegmentationColorGroup: LinkedLayerGroup;
  originalSegmentationGroupState: SegmentationUserLayerGroupState;
  originalSegmentationColorGroupState: SegmentationUserLayerColorGroupState;

  segmentationGroupState: WatchableValueInterface<SegmentationUserLayerGroupState>;
  segmentationColorGroupState: WatchableValueInterface<SegmentationUserLayerColorGroupState>;

  // Indirect properties
  hideSegmentZero: WatchableValueInterface<boolean>;
  segmentColorHash: TrackableValueInterface<number>;
  segmentStatedColors: WatchableValueInterface<Uint64Map>;
  tempSegmentStatedColors2d: WatchableValueInterface<Uint64Map>;
  segmentDefaultColor: WatchableValueInterface<vec3 | undefined>;
  tempSegmentDefaultColor2d: WatchableValueInterface<vec3 | vec4 | undefined>;
  highlightColor: WatchableValueInterface<vec4 | undefined>;
  segmentQuery: WatchableValueInterface<string>;
  segmentPropertyMap: WatchableValueInterface<
    PreprocessedSegmentPropertyMap | undefined
  >;
}

interface SegmentationActionContext extends LayerActionContext {
  // Restrict the `select` action not to both toggle on and off segments.  If segment would be
  // toggled on in at least one layer, only toggle segments on.
  segmentationToggleSegmentState?: boolean | undefined;
}

const Base = UserLayerWithAnnotationsMixin(UserLayer);
export class SegmentationUserLayer extends Base {
  sliceViewRenderScaleHistogram = new RenderScaleHistogram();
  sliceViewRenderScaleTarget = trackableRenderScaleTarget(1);
  codeVisible = new TrackableBoolean(true);

  graphConnection = new WatchableValue<
    SegmentationGraphSourceConnection | undefined
  >(undefined);

  bindSegmentListWidth(element: HTMLElement) {
    return bindSegmentListWidth(this.displayState, element);
  }

  segmentQueryFocusTime = new WatchableValue<number>(Number.NEGATIVE_INFINITY);

  selectSegment = (id: bigint, pin: boolean | "toggle") => {
    this.manager.root.selectionState.captureSingleLayerState(
      this,
      (state) => {
        state.value = id;
        return true;
      },
      pin,
    );
  };

  filterBySegmentLabel = (id: bigint) => {
    const augmented = augmentSegmentId(this.displayState, id);
    const { label } = augmented;
    if (!label) return;
    this.filterSegments(label);
  };

  filterSegments = (query: string) => {
    this.displayState.segmentationGroupState.value.segmentQuery.value = query;
    this.segmentQueryFocusTime.value = Date.now();
    this.tabs.value = "segments";
    this.manager.root.selectedLayer.layer = this.managedLayer;
  };

  displayState = new SegmentationUserLayerDisplayState(this);

  anchorSegment = new TrackableValue<bigint | undefined>(undefined, (x) =>
    x === undefined ? undefined : parseUint64(x),
  );

  constructor(managedLayer: Borrowed<ManagedUserLayer>) {
    super(managedLayer);
    this.codeVisible.changed.add(this.specificationChanged.dispatch);
    this.registerDisposer(
      registerNestedSync((context, group) => {
        context.registerDisposer(
          group.specificationChanged.add(this.specificationChanged.dispatch),
        );
        this.specificationChanged.dispatch();
      }, this.displayState.segmentationGroupState),
    );
    this.registerDisposer(
      registerNestedSync((context, group) => {
        context.registerDisposer(
          group.specificationChanged.add(this.specificationChanged.dispatch),
        );
        this.specificationChanged.dispatch();
      }, this.displayState.segmentationColorGroupState),
    );
    this.displayState.segmentSelectionState.bindTo(
      this.manager.layerSelectedValues,
      this,
    );
    this.displayState.selectedAlpha.changed.add(
      this.specificationChanged.dispatch,
    );
    this.displayState.saturation.changed.add(
      this.specificationChanged.dispatch,
    );
    this.displayState.notSelectedAlpha.changed.add(
      this.specificationChanged.dispatch,
    );
    this.displayState.objectAlpha.changed.add(
      this.specificationChanged.dispatch,
    );
    this.displayState.hiddenObjectAlpha.changed.add(
      this.specificationChanged.dispatch,
    );
    this.displayState.skeletonLod.changed.add(
      this.specificationChanged.dispatch,
    );
    this.displayState.spatialSkeletonGridResolutionTarget2d.changed.add(
      this.specificationChanged.dispatch,
    );
    this.displayState.spatialSkeletonGridResolutionTarget3d.changed.add(
      this.specificationChanged.dispatch,
    );
    this.displayState.spatialSkeletonGridResolutionRelative2d.changed.add(
      this.specificationChanged.dispatch,
    );
    this.displayState.spatialSkeletonGridResolutionRelative3d.changed.add(
      this.specificationChanged.dispatch,
    );
    this.displayState.spatialSkeletonGridLevel2d.changed.add(
      this.specificationChanged.dispatch,
    );
    this.displayState.spatialSkeletonGridLevel3d.changed.add(
      this.specificationChanged.dispatch,
    );
    this.displayState.hoverHighlight.changed.add(
      this.specificationChanged.dispatch,
    );
    this.displayState.baseSegmentColoring.changed.add(
      this.specificationChanged.dispatch,
    );
    this.displayState.ignoreNullVisibleSet.changed.add(
      this.specificationChanged.dispatch,
    );
    this.displayState.skeletonRenderingOptions.changed.add(
      this.specificationChanged.dispatch,
    );
    this.displayState.renderScaleTarget.changed.add(
      this.specificationChanged.dispatch,
    );
    this.displayState.silhouetteRendering.changed.add(
      this.specificationChanged.dispatch,
    );
    this.anchorSegment.changed.add(this.specificationChanged.dispatch);
    this.sliceViewRenderScaleTarget.changed.add(
      this.specificationChanged.dispatch,
    );
    this.displayState.originalSegmentationGroupState.localGraph.changed.add(
      this.specificationChanged.dispatch,
    );
    this.displayState.linkedSegmentationGroup.changed.add(() =>
      this.updateDataSubsourceActivations(),
    );
    this.tabs.add("rendering", {
      label: "Render",
      order: -100,
      getter: () => new DisplayOptionsTab(this),
    });
    this.tabs.add("segments", {
      label: "Seg.",
      order: -50,
      getter: () => new SegmentDisplayTab(this),
    });
    const hideGraphTab = this.registerDisposer(
      makeCachedDerivedWatchableValue(
        (x) => x === undefined,
        [this.displayState.segmentationGroupState.value.graph],
      ),
    );
    this.tabs.add("graph", {
      label: "Graph",
      order: -25,
      getter: () => new SegmentationGraphSourceTab(this),
      hidden: hideGraphTab,
    });
    this.tabs.default = "rendering";
  }

  get volumeOptions() {
    return { volumeType: VolumeType.SEGMENTATION };
  }

  readonly has2dLayer = this.registerDisposer(
    makeCachedLazyDerivedWatchableValue(
      (layers) => layers.some((x) => x instanceof SegmentationRenderLayer),
      { changed: this.layersChanged, value: this.renderLayers },
    ),
  );

  readonly has3dLayer = this.registerDisposer(
    makeCachedLazyDerivedWatchableValue(
      (layers) =>
        layers.some(
          (x) =>
            x instanceof MeshLayer ||
            x instanceof MultiscaleMeshLayer ||
            x instanceof PerspectiveViewSkeletonLayer ||
            x instanceof SliceViewPanelSkeletonLayer ||
            x instanceof PerspectiveViewSpatiallyIndexedSkeletonLayer ||
            x instanceof SliceViewPanelSpatiallyIndexedSkeletonLayer,
        ),
      { changed: this.layersChanged, value: this.renderLayers },
    ),
  );

  readonly hasSkeletonsLayer = this.registerDisposer(
    makeCachedLazyDerivedWatchableValue(
      (layers) =>
        layers.some(
          (x) =>
            x instanceof PerspectiveViewSkeletonLayer ||
            x instanceof PerspectiveViewSpatiallyIndexedSkeletonLayer,
        ),
      { changed: this.layersChanged, value: this.renderLayers },
    ),
  );

  readonly hasSpatiallyIndexedSkeletonsLayer = this.registerDisposer(
    makeCachedLazyDerivedWatchableValue(
      (layers) =>
        layers.some(
          (x) =>
            x instanceof PerspectiveViewSpatiallyIndexedSkeletonLayer ||
            x instanceof SliceViewPanelSpatiallyIndexedSkeletonLayer,
        ),
      { changed: this.layersChanged, value: this.renderLayers },
    ),
  );

  readonly getSkeletonLayer = () => {
    for (const layer of this.renderLayers) {
      if (layer instanceof PerspectiveViewSkeletonLayer) {
        return layer.base;
      }
      if (layer instanceof PerspectiveViewSpatiallyIndexedSkeletonLayer) {
        return layer.base;
      }
    }
    return undefined;
  };

  activateDataSubsources(subsources: Iterable<LoadedDataSubsource>) {
    const updatedSegmentPropertyMaps: SegmentPropertyMap[] = [];
    const isGroupRoot =
      this.displayState.linkedSegmentationGroup.root.value === this;
    let updatedGraph: SegmentationGraphSource | undefined;
    let hasVolume = false;
    let spatialSkeletonGridSizes: SpatialSkeletonGridSize[] | undefined;
    for (const loadedSubsource of subsources) {
      if (this.addStaticAnnotations(loadedSubsource)) continue;
      const {
        volume,
        mesh,
        skeleton,
        segmentPropertyMap,
        segmentationGraph,
        local,
      } = loadedSubsource.subsourceEntry.subsource as any;
      if (volume instanceof MultiscaleVolumeChunkSource) {
        switch (volume.dataType) {
          case DataType.FLOAT32:
            loadedSubsource.deactivate(
              "Data type not compatible with segmentation layer",
            );
            continue;
        }
        hasVolume = true;
        loadedSubsource.activate(
          () =>
            loadedSubsource.addRenderLayer(
              new SegmentationRenderLayer(volume, {
                ...this.displayState,
                transform: loadedSubsource.getRenderLayerTransform(),
                renderScaleTarget: this.sliceViewRenderScaleTarget,
                renderScaleHistogram: this.sliceViewRenderScaleHistogram,
                localPosition: this.localPosition,
              }),
            ),
          this.displayState.segmentationGroupState.value,
        );
      } else if (mesh !== undefined) {
        loadedSubsource.activate(() => {
          const displayState = {
            ...this.displayState,
            transform: loadedSubsource.getRenderLayerTransform(),
            localPosition: this.localPosition,
          };
          if (mesh instanceof MeshSource) {
            loadedSubsource.addRenderLayer(
              new MeshLayer(this.manager.chunkManager, mesh, displayState),
            );
          } else if (mesh instanceof MultiscaleMeshSource) {
            loadedSubsource.addRenderLayer(
              new MultiscaleMeshLayer(
                this.manager.chunkManager,
                mesh,
                displayState,
              ),
            );
          } else if (mesh instanceof MultiscaleSpatiallyIndexedSkeletonSource) {
            const base = new MultiscaleSliceViewSpatiallyIndexedSkeletonLayer(
              this.manager.chunkManager,
              mesh,
              displayState,
            );
            loadedSubsource.addRenderLayer(base);

            const perspectiveSources = mesh.getPerspectiveSources();
            const slicePanelSources = mesh.getSliceViewPanelSources();
            if (perspectiveSources.length > 0) {
              const base3d = new SpatiallyIndexedSkeletonLayer(
                this.manager.chunkManager,
                perspectiveSources,
                displayState,
                {
                  gridLevel: displayState.spatialSkeletonGridLevel3d,
                  lod: displayState.skeletonLod,
                },
              );
              loadedSubsource.addRenderLayer(
                new PerspectiveViewSpatiallyIndexedSkeletonLayer(
                  /* transfer ownership */ base3d,
                ),
              );
            }
            if (slicePanelSources.length > 0) {
              const base2d = new SpatiallyIndexedSkeletonLayer(
                this.manager.chunkManager,
                slicePanelSources,
                displayState,
                {
                  gridLevel: displayState.spatialSkeletonGridLevel2d,
                  lod: displayState.spatialSkeletonLod2d,
                },
              );
              loadedSubsource.addRenderLayer(
                new SliceViewPanelSpatiallyIndexedSkeletonLayer(
                  /* transfer ownership */ base2d,
                ),
              );
            }
            spatialSkeletonGridSizes = mesh.getSpatialSkeletonGridSizes();
          } else if (mesh instanceof SpatiallyIndexedSkeletonSource) {
            const base3d = new SpatiallyIndexedSkeletonLayer(
              this.manager.chunkManager,
              mesh,
              displayState,
              {
                gridLevel: displayState.spatialSkeletonGridLevel3d,
                lod: displayState.skeletonLod,
              },
            );
            loadedSubsource.addRenderLayer(
              new PerspectiveViewSpatiallyIndexedSkeletonLayer(
                /* transfer ownership */ base3d,
              ),
            );
            const base2d = new SpatiallyIndexedSkeletonLayer(
              this.manager.chunkManager,
              mesh,
              displayState,
              {
                gridLevel: displayState.spatialSkeletonGridLevel2d,
                lod: displayState.spatialSkeletonLod2d,
              },
            );
            loadedSubsource.addRenderLayer(
              new SliceViewSpatiallyIndexedSkeletonLayer(base2d.addRef()),
            );
            loadedSubsource.addRenderLayer(
              new SliceViewPanelSpatiallyIndexedSkeletonLayer(
                /* transfer ownership */ base2d,
              ),
            );
          } else {
            const base = new SkeletonLayer(
              this.manager.chunkManager,
              mesh,
              displayState,
            );
            loadedSubsource.addRenderLayer(
              new PerspectiveViewSkeletonLayer(base.addRef()),
            );
            loadedSubsource.addRenderLayer(
              new SliceViewPanelSkeletonLayer(/* transfer ownership */ base),
            );
          }
        }, this.displayState.segmentationGroupState.value);
      } else if (skeleton !== undefined) {
        loadedSubsource.activate(() => {
          const displayState = {
            ...this.displayState,
            transform: loadedSubsource.getRenderLayerTransform(),
            localPosition: this.localPosition,
          };
          if (skeleton instanceof SpatiallyIndexedSkeletonSource) {
            const base3d = new SpatiallyIndexedSkeletonLayer(
              this.manager.chunkManager,
              skeleton,
              displayState,
              {
                gridLevel: displayState.spatialSkeletonGridLevel3d,
                lod: displayState.skeletonLod,
              },
            );
            loadedSubsource.addRenderLayer(
              new PerspectiveViewSpatiallyIndexedSkeletonLayer(
                /* transfer ownership */ base3d,
              ),
            );
            const base2d = new SpatiallyIndexedSkeletonLayer(
              this.manager.chunkManager,
              skeleton,
              displayState,
              {
                gridLevel: displayState.spatialSkeletonGridLevel2d,
                lod: displayState.spatialSkeletonLod2d,
              },
            );
            loadedSubsource.addRenderLayer(
              new SliceViewSpatiallyIndexedSkeletonLayer(base2d.addRef()),
            );
            loadedSubsource.addRenderLayer(
              new SliceViewPanelSpatiallyIndexedSkeletonLayer(
                /* transfer ownership */ base2d,
              ),
            );
          } else {
            const base = new SkeletonLayer(
              this.manager.chunkManager,
              skeleton,
              displayState,
            );
            loadedSubsource.addRenderLayer(
              new PerspectiveViewSkeletonLayer(base.addRef()),
            );
            loadedSubsource.addRenderLayer(
              new SliceViewPanelSkeletonLayer(/* transfer ownership */ base),
            );
          }
        }, this.displayState.segmentationGroupState.value);
      } else if (segmentPropertyMap !== undefined) {
        if (!isGroupRoot) {
          loadedSubsource.deactivate(
            "Not supported on non-root linked segmentation layers",
          );
        } else {
          loadedSubsource.activate(() => { });
          updatedSegmentPropertyMaps.push(segmentPropertyMap);
        }
      } else if (segmentationGraph !== undefined) {
        if (!isGroupRoot) {
          loadedSubsource.deactivate(
            "Not supported on non-root linked segmentation layers",
          );
        } else {
          if (updatedGraph !== undefined) {
            loadedSubsource.deactivate(
              "Only one segmentation graph is supported",
            );
          } else {
            updatedGraph = segmentationGraph;
            loadedSubsource.activate((refCounted) => {
              const graphConnection = segmentationGraph.connect(this);
              refCounted.registerDisposer(() => {
                graphConnection.dispose();
                this.graphConnection.value = undefined;
              });
              const displayState = {
                ...this.displayState,
                transform: loadedSubsource.getRenderLayerTransform(),
              };

              const graphRenderLayers = graphConnection.createRenderLayers(
                this.manager.chunkManager,
                displayState,
                this.localPosition,
              );
              this.graphConnection.value = graphConnection;
              for (const renderLayer of graphRenderLayers) {
                loadedSubsource.addRenderLayer(renderLayer);
              }
            });
          }
        }
      } else if (local === LocalDataSource.equivalences) {
        if (!isGroupRoot) {
          loadedSubsource.deactivate(
            "Not supported on non-root linked segmentation layers",
          );
        } else {
          if (updatedGraph !== undefined) {
            loadedSubsource.deactivate(
              "Only one segmentation graph is supported",
            );
          } else {
            updatedGraph =
              this.displayState.originalSegmentationGroupState.localGraph;
            loadedSubsource.activate((refCounted) => {
              this.graphConnection.value = refCounted.registerDisposer(
                updatedGraph!.connect(this),
              );
              refCounted.registerDisposer(() => {
                this.graphConnection.value = undefined;
              });
            });
          }
        }
      } else {
        loadedSubsource.deactivate("Not compatible with segmentation layer");
      }
    }
    this.displayState.originalSegmentationGroupState.segmentPropertyMap.value =
      getPreprocessedSegmentPropertyMap(
        this.manager.chunkManager,
        updatedSegmentPropertyMaps,
      );
    this.displayState.originalSegmentationGroupState.graph.value = updatedGraph;
    this.displayState.setSpatialSkeletonGridSizes(
      spatialSkeletonGridSizes ?? [],
    );
    this.displayState.hasVolume.value = hasVolume;
  }

  getLegacyDataSourceSpecifications(
    sourceSpec: any,
    layerSpec: any,
    legacyTransform: CoordinateTransformSpecification | undefined,
    explicitSpecs: DataSourceSpecification[],
  ): DataSourceSpecification[] {
    const specs = super.getLegacyDataSourceSpecifications(
      sourceSpec,
      layerSpec,
      legacyTransform,
      explicitSpecs,
    );
    const meshPath = verifyOptionalObjectProperty(
      layerSpec,
      json_keys.MESH_JSON_KEY,
      (x) => (x === null ? null : verifyString(x)),
    );
    const skeletonsPath = verifyOptionalObjectProperty(
      layerSpec,
      json_keys.SKELETONS_JSON_KEY,
      (x) => (x === null ? null : verifyString(x)),
    );
    if (meshPath !== undefined || skeletonsPath !== undefined) {
      for (const spec of specs) {
        spec.enableDefaultSubsources = false;
        spec.subsources = new Map([
          ["default", { enabled: true }],
          ["bounds", { enabled: true }],
        ]);
      }
    }
    if (meshPath != null) {
      specs.push(
        layerDataSourceSpecificationFromJson(
          this.manager.dataSourceProviderRegistry.convertLegacyUrl({
            url: meshPath,
            type: "mesh",
          }),
        ),
      );
    }
    if (skeletonsPath != null) {
      specs.push(
        layerDataSourceSpecificationFromJson(
          this.manager.dataSourceProviderRegistry.convertLegacyUrl({
            url: skeletonsPath,
            type: "skeletons",
          }),
        ),
      );
    }
    if (
      layerSpec[json_keys.EQUIVALENCES_JSON_KEY] !== undefined &&
      explicitSpecs.find((spec) => spec.url === localEquivalencesUrl) ===
      undefined
    ) {
      specs.push({
        url: localEquivalencesUrl,
        enableDefaultSubsources: true,
        transform: {
          outputSpace: emptyValidCoordinateSpace,
          sourceRank: 0,
          transform: undefined,
          inputSpace: emptyValidCoordinateSpace,
        },
        subsources: new Map(),
      });
    }
    return specs;
  }

  restoreState(specification: any) {
    super.restoreState(specification);
    this.displayState.selectedAlpha.restoreState(
      specification[json_keys.SELECTED_ALPHA_JSON_KEY],
    );
    this.displayState.saturation.restoreState(
      specification[json_keys.SATURATION_JSON_KEY],
    );
    this.displayState.notSelectedAlpha.restoreState(
      specification[json_keys.NOT_SELECTED_ALPHA_JSON_KEY],
    );
    this.displayState.hoverHighlight.restoreState(
      specification[json_keys.HOVER_HIGHLIGHT_JSON_KEY],
    );
    this.displayState.objectAlpha.restoreState(
      specification[json_keys.OBJECT_ALPHA_JSON_KEY],
    );
    this.displayState.hiddenObjectAlpha.restoreState(
      specification[json_keys.HIDDEN_OPACITY_3D_JSON_KEY],
    );
    this.displayState.skeletonLod.restoreState(
      specification[json_keys.SKELETON_LOD_JSON_KEY],
    );
    this.displayState.spatialSkeletonGridResolutionRelative2d.restoreState(
      specification[
        json_keys.SPATIAL_SKELETON_GRID_RESOLUTION_RELATIVE_2D_JSON_KEY
      ],
    );
    this.displayState.spatialSkeletonGridResolutionRelative3d.restoreState(
      specification[
        json_keys.SPATIAL_SKELETON_GRID_RESOLUTION_RELATIVE_3D_JSON_KEY
      ],
    );
    this.displayState.applySpatialSkeletonGridResolutionTarget2dFromSpec(
      specification[
        json_keys.SPATIAL_SKELETON_GRID_RESOLUTION_TARGET_2D_JSON_KEY
      ],
    );
    this.displayState.applySpatialSkeletonGridResolutionTarget3dFromSpec(
      specification[
        json_keys.SPATIAL_SKELETON_GRID_RESOLUTION_TARGET_3D_JSON_KEY
      ],
    );
    this.displayState.applySpatialSkeletonGridLevel2dFromSpec(
      specification[json_keys.SPATIAL_SKELETON_GRID_LEVEL_2D_JSON_KEY],
    );
    this.displayState.applySpatialSkeletonGridLevel3dFromSpec(
      specification[json_keys.SPATIAL_SKELETON_GRID_LEVEL_3D_JSON_KEY],
    );
    this.displayState.baseSegmentColoring.restoreState(
      specification[json_keys.BASE_SEGMENT_COLORING_JSON_KEY],
    );
    this.displayState.silhouetteRendering.restoreState(
      specification[json_keys.MESH_SILHOUETTE_RENDERING_JSON_KEY],
    );
    this.displayState.ignoreNullVisibleSet.restoreState(
      specification[json_keys.IGNORE_NULL_VISIBLE_SET_JSON_KEY],
    );

    const { skeletonRenderingOptions } = this.displayState;
    skeletonRenderingOptions.restoreState(
      specification[json_keys.SKELETON_RENDERING_JSON_KEY],
    );
    const skeletonShader = specification[json_keys.SKELETON_SHADER_JSON_KEY];
    if (skeletonShader !== undefined) {
      skeletonRenderingOptions.shader.restoreState(skeletonShader);
    }
    this.codeVisible.restoreState(json_keys.SKELETON_CODE_VISIBLE_KEY);
    this.displayState.renderScaleTarget.restoreState(
      specification[json_keys.MESH_RENDER_SCALE_JSON_KEY],
    );
    this.anchorSegment.restoreState(
      specification[json_keys.ANCHOR_SEGMENT_JSON_KEY],
    );
    this.sliceViewRenderScaleTarget.restoreState(
      specification[json_keys.CROSS_SECTION_RENDER_SCALE_JSON_KEY],
    );
    const linkedSegmentationGroupName = verifyOptionalObjectProperty(
      specification,
      json_keys.LINKED_SEGMENTATION_GROUP_JSON_KEY,
      verifyString,
    );
    if (linkedSegmentationGroupName !== undefined) {
      this.displayState.linkedSegmentationGroup.linkByName(
        linkedSegmentationGroupName,
      );
    }
    const linkedSegmentationColorGroupName = verifyOptionalObjectProperty(
      specification,
      json_keys.LINKED_SEGMENTATION_COLOR_GROUP_JSON_KEY,
      (x) => (x === false ? undefined : verifyString(x)),
      linkedSegmentationGroupName,
    );
    if (linkedSegmentationColorGroupName !== undefined) {
      this.displayState.linkedSegmentationColorGroup.linkByName(
        linkedSegmentationColorGroupName,
      );
    }
    this.displayState.segmentationGroupState.value.restoreState(specification);
    this.displayState.segmentationColorGroupState.value.restoreState(
      specification,
    );
  }

  toJSON() {
    const x = super.toJSON();
    x[json_keys.SELECTED_ALPHA_JSON_KEY] =
      this.displayState.selectedAlpha.toJSON();
    x[json_keys.NOT_SELECTED_ALPHA_JSON_KEY] =
      this.displayState.notSelectedAlpha.toJSON();
    x[json_keys.SATURATION_JSON_KEY] = this.displayState.saturation.toJSON();
    x[json_keys.OBJECT_ALPHA_JSON_KEY] = this.displayState.objectAlpha.toJSON();
    x[json_keys.HIDDEN_OPACITY_3D_JSON_KEY] = this.displayState.hiddenObjectAlpha.toJSON();
    x[json_keys.SKELETON_LOD_JSON_KEY] = this.displayState.skeletonLod.toJSON();
    x[json_keys.SPATIAL_SKELETON_GRID_RESOLUTION_TARGET_2D_JSON_KEY] =
      this.displayState.spatialSkeletonGridResolutionTarget2d.toJSON();
    x[json_keys.SPATIAL_SKELETON_GRID_RESOLUTION_TARGET_3D_JSON_KEY] =
      this.displayState.spatialSkeletonGridResolutionTarget3d.toJSON();
    x[json_keys.SPATIAL_SKELETON_GRID_RESOLUTION_RELATIVE_2D_JSON_KEY] =
      this.displayState.spatialSkeletonGridResolutionRelative2d.toJSON();
    x[json_keys.SPATIAL_SKELETON_GRID_RESOLUTION_RELATIVE_3D_JSON_KEY] =
      this.displayState.spatialSkeletonGridResolutionRelative3d.toJSON();
    x[json_keys.SPATIAL_SKELETON_GRID_LEVEL_2D_JSON_KEY] =
      this.displayState.spatialSkeletonGridLevel2d.toJSON();
    x[json_keys.SPATIAL_SKELETON_GRID_LEVEL_3D_JSON_KEY] =
      this.displayState.spatialSkeletonGridLevel3d.toJSON();
    x[json_keys.HOVER_HIGHLIGHT_JSON_KEY] =
      this.displayState.hoverHighlight.toJSON();
    x[json_keys.BASE_SEGMENT_COLORING_JSON_KEY] =
      this.displayState.baseSegmentColoring.toJSON();
    x[json_keys.IGNORE_NULL_VISIBLE_SET_JSON_KEY] =
      this.displayState.ignoreNullVisibleSet.toJSON();
    x[json_keys.MESH_SILHOUETTE_RENDERING_JSON_KEY] =
      this.displayState.silhouetteRendering.toJSON();
    x[json_keys.ANCHOR_SEGMENT_JSON_KEY] = this.anchorSegment.toJSON();
    x[json_keys.SKELETON_RENDERING_JSON_KEY] =
      this.displayState.skeletonRenderingOptions.toJSON();
    x[json_keys.SKELETON_CODE_VISIBLE_KEY] = this.codeVisible.toJSON();
    x[json_keys.MESH_RENDER_SCALE_JSON_KEY] =
      this.displayState.renderScaleTarget.toJSON();
    x[json_keys.CROSS_SECTION_RENDER_SCALE_JSON_KEY] =
      this.sliceViewRenderScaleTarget.toJSON();

    const { linkedSegmentationGroup, linkedSegmentationColorGroup } =
      this.displayState;
    x[json_keys.LINKED_SEGMENTATION_GROUP_JSON_KEY] =
      linkedSegmentationGroup.toJSON();
    if (
      linkedSegmentationColorGroup.root.value !==
      linkedSegmentationGroup.root.value
    ) {
      x[json_keys.LINKED_SEGMENTATION_COLOR_GROUP_JSON_KEY] =
        linkedSegmentationColorGroup.toJSON() ?? false;
    }
    x[json_keys.EQUIVALENCES_JSON_KEY] =
      this.displayState.originalSegmentationGroupState.localGraph.toJSON();
    if (linkedSegmentationGroup.root.value === this) {
      Object.assign(x, this.displayState.segmentationGroupState.value.toJSON());
    }
    if (linkedSegmentationColorGroup.root.value === this) {
      Object.assign(
        x,
        this.displayState.segmentationColorGroupState.value.toJSON(),
      );
    }
    return x;
  }

  transformPickedValue(value: any) {
    if (value == null) {
      return value;
    }
    return maybeAugmentSegmentId(this.displayState, value);
  }

  handleAction(action: string, context: SegmentationActionContext) {
    switch (action) {
      case "recolor": {
        this.displayState.segmentationColorGroupState.value.segmentColorHash.randomize();
        break;
      }
      case "clear-segments": {
        if (!this.pick.value) break;
        this.displayState.segmentationGroupState.value.visibleSegments.clear();
        break;
      }
      case "select":
      case "star": {
        if (!this.pick.value) break;
        const { segmentSelectionState } = this.displayState;
        if (segmentSelectionState.hasSelectedSegment) {
          const segment = segmentSelectionState.selectedSegment;
          const group = this.displayState.segmentationGroupState.value;
          const segmentSet =
            action === "select"
              ? group.visibleSegments
              : group.selectedSegments;
          const newValue = !segmentSet.has(segment);
          if (
            newValue ||
            context.segmentationToggleSegmentState === undefined
          ) {
            context.segmentationToggleSegmentState = newValue;
          }
          context.defer(() => {
            if (context.segmentationToggleSegmentState === newValue) {
              segmentSet.set(segment, newValue);
            }
          });
        }
        break;
      }
    }
  }
  selectionStateFromJson(state: this["selectionState"], json: any) {
    super.selectionStateFromJson(state, json);
    let { value } = state;
    if (typeof value === "number") value = value.toString();
    try {
      state.value = parseUint64(value);
    } catch {
      state.value = undefined;
    }
  }
  selectionStateToJson(state: this["selectionState"], forPython: boolean): any {
    const json = super.selectionStateToJson(state, forPython);
    const { value } = state;
    if (value instanceof Uint64MapEntry) {
      if (forPython) {
        json.value = {
          key: value.key.toString(),
          value: value.value ? value.value.toString() : undefined,
          label: value.label,
        };
      } else {
        json.value = (value.value || value.key).toString();
      }
    } else if (typeof value === "bigint") {
      json.value = value.toString();
    }
    return json;
  }

  private displaySegmentationSelection(
    state: this["selectionState"],
    parent: HTMLElement,
    context: DependentViewContext,
  ): boolean {
    const { value } = state;
    let id: bigint;
    if (typeof value === "number" || typeof value === "string") {
      try {
        id = parseUint64(value);
      } catch {
        return false;
      }
    }
    if (typeof value === "bigint") {
      id = value;
    } else if (value instanceof Uint64MapEntry) {
      id = value.key;
    } else {
      return false;
    }
    const { displayState } = this;
    const normalizedId = augmentSegmentId(displayState, id);
    const {
      segmentEquivalences,
      segmentPropertyMap: { value: segmentPropertyMap },
    } = this.displayState.segmentationGroupState.value;
    const mapped = segmentEquivalences.get(id);
    const row = makeSegmentWidget(this.displayState, normalizedId);
    registerCallbackWhenSegmentationDisplayStateChanged(
      displayState,
      context,
      context.redraw,
    );
    context.registerDisposer(bindSegmentListWidth(displayState, row));
    row.classList.add("neuroglancer-selection-details-segment");
    parent.appendChild(row);

    if (segmentPropertyMap !== undefined) {
      const { inlineProperties } = segmentPropertyMap.segmentPropertyMap;
      if (inlineProperties !== undefined) {
        const index = segmentPropertyMap.getSegmentInlineIndex(mapped);
        if (index !== -1) {
          for (const property of inlineProperties.properties) {
            if (property.type === "label") continue;
            if (property.type === "description") {
              const value = property.values[index];
              if (!value) continue;
              const descriptionElement = document.createElement("div");
              descriptionElement.classList.add(
                "neuroglancer-selection-details-segment-description",
              );
              descriptionElement.textContent = value;
              parent.appendChild(descriptionElement);
            } else if (
              property.type === "number" ||
              property.type === "string"
            ) {
              const value = property.values[index];
              if (
                property.type === "number"
                  ? Number.isNaN(value as number)
                  : !value
              )
                continue;
              const propertyElement = document.createElement("div");
              propertyElement.classList.add(
                "neuroglancer-selection-details-segment-property",
              );
              const nameElement = document.createElement("div");
              nameElement.classList.add(
                "neuroglancer-selection-details-segment-property-name",
              );
              nameElement.textContent = property.id;
              if (property.description) {
                nameElement.title = property.description;
              }
              const valueElement = document.createElement("div");
              valueElement.classList.add(
                "neuroglancer-selection-details-segment-property-value",
              );
              valueElement.textContent = value.toString();
              propertyElement.appendChild(nameElement);
              propertyElement.appendChild(valueElement);
              parent.appendChild(propertyElement);
            }
          }
        }
      }
    }
    return true;
  }

  displaySelectionState(
    state: this["selectionState"],
    parent: HTMLElement,
    context: DependentViewContext,
  ): boolean {
    let displayed = this.displaySegmentationSelection(state, parent, context);
    if (super.displaySelectionState(state, parent, context)) displayed = true;
    return displayed;
  }

  moveToSegment(id: bigint) {
    for (const layer of this.renderLayers) {
      if (
        !(layer instanceof MultiscaleMeshLayer || layer instanceof MeshLayer)
      ) {
        continue;
      }
      const transform = layer.displayState.transform.value;
      if (transform.error !== undefined) return undefined;
      const { rank, globalToRenderLayerDimensions } = transform;
      const { globalPosition } = this.manager.root;
      const globalLayerPosition = new Float32Array(rank);
      const renderToGlobalLayerDimensions = [];
      for (let i = 0; i < rank; i++) {
        renderToGlobalLayerDimensions[globalToRenderLayerDimensions[i]] = i;
      }
      gatherUpdate(
        globalLayerPosition,
        globalPosition.value,
        renderToGlobalLayerDimensions,
      );
      const layerPosition =
        layer instanceof MeshLayer
          ? layer.getObjectPosition(id, globalLayerPosition)
          : layer.getObjectPosition(id);
      if (layerPosition === undefined) continue;
      this.setLayerPosition(transform, layerPosition);
      return;
    }
    StatusMessage.showTemporaryMessage(
      `No position information loaded for segment ${id}`,
    );
  }

  observeLayerColor(callback: () => void) {
    const disposer = super.observeLayerColor(callback);
    const defaultColorDisposer = observeWatchable(
      callback,
      this.displayState.segmentDefaultColor,
    );
    const visibleSegmentDisposer =
      this.displayState.segmentationGroupState.value.visibleSegments.changed.add(
        callback,
      );
    const colorHashChangeDisposer =
      this.displayState.segmentationColorGroupState.value.segmentColorHash.changed.add(
        callback,
      );
    const showAllByDefaultDisposer =
      this.displayState.ignoreNullVisibleSet.changed.add(callback);
    const hasVolumeDisposer = this.displayState.hasVolume.changed.add(callback);
    return () => {
      disposer();
      defaultColorDisposer();
      visibleSegmentDisposer();
      colorHashChangeDisposer();
      showAllByDefaultDisposer();
      hasVolumeDisposer();
    };
  }

  get automaticLayerBarColors() {
    const { displayState } = this;
    const visibleSegmentsSet =
      displayState.segmentationGroupState.value.visibleSegments;
    const fixedColor = displayState.segmentDefaultColor.value;

    const noVisibleSegments = visibleSegmentsSet.size === 0;
    const tooManyVisibleSegments =
      visibleSegmentsSet.size > MAX_LAYER_BAR_UI_INDICATOR_COLORS;
    const hasMappedColors =
      displayState.segmentationColorGroupState.value.segmentStatedColors.size >
      0;
    const isFixedColorOnly = fixedColor !== undefined && !hasMappedColors;
    const showAllByDefault = displayState.ignoreNullVisibleSet.value;
    const hasVolume = displayState.hasVolume.value;

    if (noVisibleSegments) {
      if (!showAllByDefault || !hasVolume) return []; // No segments visible
      if (isFixedColorOnly) return [getCssColor(fixedColor)];
      return undefined; // Rainbow colors
    }
    if (isFixedColorOnly) {
      return [getCssColor(fixedColor)]; // All segments show as one color
    }

    // Because manually mapped colors are not guaranteed to be unique,
    // we need to actually check all the visible segments if
    // manually mapped colors are used
    if (!hasMappedColors && tooManyVisibleSegments) {
      return undefined; // Too many segments to show
    }

    const visibleSegments = [...visibleSegmentsSet];
    const colors = visibleSegments.map((id) => {
      const color = getCssColor(getBaseObjectColor(displayState, id));
      return { color, id };
    });

    // Sort the colors by their segment ID
    // Otherwise, the order is random which is a bit confusing in the UI
    colors.sort((a, b) => {
      const aId = a.id;
      const bId = b.id;
      return aId < bId ? -1 : aId > bId ? 1 : 0;
    });

    const uniqueColors = [...new Set(colors.map((color) => color.color))];
    if (uniqueColors.length > MAX_LAYER_BAR_UI_INDICATOR_COLORS) {
      return undefined; // Too many colors to show
    }
    return uniqueColors;
  }

  static type = "segmentation";
  static typeAbbreviation = "seg";
  static supportsPickOption = true;
  static supportsLayerBarColorSyncOption = true;
}

registerLayerControls(SegmentationUserLayer);

registerLayerType(SegmentationUserLayer);
registerVolumeLayerType(VolumeType.SEGMENTATION, SegmentationUserLayer);
registerLayerTypeDetector((subsource) => {
  if (subsource.mesh !== undefined) {
    return { layerConstructor: SegmentationUserLayer, priority: 1 };
  }
  return undefined;
});

registerLayerShaderControlsTool(
  SegmentationUserLayer,
  (layer) => ({
    shaderControlState:
      layer.displayState.skeletonRenderingOptions.shaderControlState,
  }),
  json_keys.SKELETON_RENDERING_SHADER_CONTROL_TOOL_ID,
);

registerSegmentSplitMergeTools(SegmentationUserLayer);
registerSegmentSelectTools(SegmentationUserLayer);
