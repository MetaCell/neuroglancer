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

import { GPUHashTable, HashSetShaderManager } from "#src/gpu_hash/shader.js";
import {
  SegmentColorShaderManager,
  SegmentStatedColorShaderManager,
} from "#src/segment_color.js";
import {
  updateOneDimensionalTextureElement,
  uploadVertexAttributesToGPU,
} from "#src/skeleton/gpu_upload_utils.js";
import {
  resolveSpatiallyIndexedSkeletonSegmentPick,
} from "#src/skeleton/picking.js";
import { spatiallyIndexedSkeletonTextureAttributeSpecs } from "#src/skeleton/spatial_attribute_layout.js";
import {
  dedupeSpatiallyIndexedSkeletonEntries,
  getSpatiallyIndexedSkeletonGridIndex,
  getSpatiallyIndexedSkeletonSourceView,
  selectSpatiallyIndexedSkeletonEntriesByGrid,
  selectSpatiallyIndexedSkeletonEntriesForView,
  type SpatiallyIndexedSkeletonView,
} from "#src/skeleton/source_selection.js";
import {
  appendNodeToSpatialChunk,
  rebuildTargetSpatialChunkConnections,
  removeNodeFromSpatialChunk,
  type SpatiallyIndexedChunkNodeLocator,
  type SpatiallyIndexedEditableChunkData,
  type SpatiallyIndexedEditableChunkEntry,
  updateNodePositionInSpatialChunk,
} from "#src/skeleton/spatial_edit.js";
import {
  getSpatiallyIndexedSkeletonSourceCapabilities,
  type SpatiallyIndexedSkeletonSourceCapabilities,
} from "#src/skeleton/state.js";

import { ChunkState, LayerChunkProgressInfo } from "#src/chunk_manager/base.js";
import type { ChunkManager } from "#src/chunk_manager/frontend.js";
import {
  Chunk,
  ChunkRenderLayerFrontend,
  ChunkSource,
} from "#src/chunk_manager/frontend.js";
import type {
  LayerView,
  MouseSelectionState,
  PickState,
  UserLayer,
  VisibleLayerInfo,
} from "#src/layer/index.js";
import type { PerspectivePanel } from "#src/perspective_view/panel.js";
import type {
  PerspectiveViewReadyRenderContext,
  PerspectiveViewRenderContext,
} from "#src/perspective_view/render_layer.js";
import { PerspectiveViewRenderLayer } from "#src/perspective_view/render_layer.js";
import type {
  RenderLayer,
  ThreeDimensionalRenderLayerAttachmentState,
} from "#src/renderlayer.js";
import { update3dRenderLayerAttachment } from "#src/renderlayer.js";
import { RenderScaleHistogram } from "#src/render_scale_statistics.js";
import {
  forEachVisibleSegment,
  getObjectKey,
} from "#src/segmentation_display_state/base.js";
import type {
  SegmentationDisplayState3D,
  SegmentationDisplayState,
} from "#src/segmentation_display_state/frontend.js";
import {
  forEachVisibleSegmentToDraw,
  registerRedrawWhenSegmentationDisplayState3DChanged,
  SegmentationLayerSharedObject,
} from "#src/segmentation_display_state/frontend.js";
import type { VertexAttributeInfo } from "#src/skeleton/base.js";
import {
  SKELETON_LAYER_RPC_ID,
  SPATIALLY_INDEXED_SKELETON_RENDER_LAYER_RPC_ID,
  SPATIALLY_INDEXED_SKELETON_RENDER_LAYER_UPDATE_SOURCES_RPC_ID,
  SPATIALLY_INDEXED_SKELETON_SLICEVIEW_RENDER_LAYER_RPC_ID,
} from "#src/skeleton/base.js";
import { RENDERED_VIEW_ADD_LAYER_RPC_ID } from "#src/render_layer_common.js";
import type { SliceViewPanel } from "#src/sliceview/panel.js";
import { SharedWatchableValue } from "#src/shared_watchable_value.js";
import type { RPC } from "#src/worker_rpc.js";
import {
  getVolumetricTransformedSources,
  serializeAllTransformedSources,
  SliceViewSingleResolutionSource,
} from "#src/sliceview/frontend.js";
import {
  type SliceViewPanelReadyRenderContext,
  SliceViewPanelRenderLayer,
  SliceViewPanelRenderContext,
  SliceViewRenderLayer,
  SliceViewRenderContext,
} from "#src/sliceview/renderlayer.js";
import {
  SliceViewChunk,
  SliceViewChunkSource,
  MultiscaleSliceViewChunkSource,
} from "#src/sliceview/frontend.js";
import {
  forEachVisibleVolumetricChunk,
  type SliceViewBase,
  type SliceViewChunkSpecification,
  type TransformedSource,
} from "#src/sliceview/base.js";
import { ChunkLayout } from "#src/sliceview/chunk_layout.js";
import {
  TrackableValue,
  WatchableValue,
  WatchableValueInterface,
  registerNested,
} from "#src/trackable_value.js";
import { DATA_TYPE_SIGNED, DataType } from "#src/util/data_type.js";
import { RefCounted } from "#src/util/disposable.js";
import { mat4 } from "#src/util/geom.js";
import { verifyFinitePositiveFloat } from "#src/util/json.js";
import * as matrix from "#src/util/matrix.js";
import { getObjectId } from "#src/util/object_id.js";
import { NullarySignal } from "#src/util/signal.js";
import type { Trackable } from "#src/util/trackable.js";
import { CompoundTrackable } from "#src/util/trackable.js";
import { TrackableEnum } from "#src/util/trackable_enum.js";
import { GLBuffer } from "#src/webgl/buffer.js";
import {
  defineCircleShader,
  drawCircles,
  initializeCircleShader,
} from "#src/webgl/circles.js";
import { glsl_COLORMAPS } from "#src/webgl/colormaps.js";
import type { GL } from "#src/webgl/context.js";
import type { WatchableShaderError } from "#src/webgl/dynamic_shader.js";
import {
  makeTrackableFragmentMain,
  parameterizedEmitterDependentShaderGetter,
  shaderCodeWithLineDirective,
} from "#src/webgl/dynamic_shader.js";
import {
  defineLineShader,
  drawLines,
  initializeLineShader,
} from "#src/webgl/lines.js";
import type {
  ShaderBuilder,
  ShaderProgram,
  ShaderSamplerType,
} from "#src/webgl/shader.js";
import {
  dataTypeShaderDefinition,
  getShaderType,
} from "#src/webgl/shader_lib.js";
import type { ShaderControlsBuilderState } from "#src/webgl/shader_ui_controls.js";
import {
  addControlsToBuilder,
  getFallbackBuilderState,
  parseShaderUiControls,
  setControlsInShader,
  ShaderControlState,
} from "#src/webgl/shader_ui_controls.js";
import {
  computeTextureFormat,
  getSamplerPrefixForDataType,
  OneDimensionalTextureAccessHelper,
  TextureFormat,
} from "#src/webgl/texture_access.js";
import { defineVertexId, VertexIdHelper } from "#src/webgl/vertex_id.js";

const tempMat2 = mat4.create();
const tempPositionUpdate = new Float32Array(3);

const DEFAULT_FRAGMENT_MAIN = `void main() {
  emitDefault();
}
`;

const SELECTED_NODE_OUTLINE_COLOR_RGB = "1.0, 0.95, 0.35";
const SELECTED_NODE_OUTLINE_MIN_WIDTH_2D = "1.75";
const SELECTED_NODE_OUTLINE_MAX_WIDTH_2D = "3.0";
const SELECTED_NODE_OUTLINE_MIN_WIDTH_3D = "1.5";
const SELECTED_NODE_OUTLINE_MAX_WIDTH_3D = "2.5";

interface VertexAttributeRenderInfo extends VertexAttributeInfo {
  name: string;
  webglDataType: number;
  glslDataType: string;
}

const vertexAttributeSamplerSymbols: symbol[] = [];

const vertexPositionTextureFormat = computeTextureFormat(
  new TextureFormat(),
  DataType.FLOAT32,
  3,
);
const segmentTextureFormat = computeTextureFormat(
  new TextureFormat(),
  DataType.UINT32,
  1,
);
const selectedNodeTextureFormat = computeTextureFormat(
  new TextureFormat(),
  DataType.FLOAT32,
  1,
);

interface SkeletonLayerInterface {
  vertexAttributes: VertexAttributeRenderInfo[];
  segmentColorAttributeIndex?: number;
  dynamicSegmentAppearance?: boolean;
  gl: GL;
  fallbackShaderParameters: WatchableValue<ShaderControlsBuilderState>;
  displayState: SkeletonLayerDisplayState;
}

interface SkeletonChunkInterface {
  vertexAttributeTextures: (WebGLTexture | null)[];
  indexBuffer: GLBuffer;
  numIndices: number;
  numVertices: number;
  pickNodeIds?: Int32Array;
  pickNodePositions?: Float32Array;
  pickSegmentIds?: Uint32Array;
  pickEdgeSegmentIds?: Uint32Array;
}

interface SkeletonChunkData {
  vertexAttributes: Uint8Array;
  indices: Uint32Array;
  numVertices: number;
  vertexAttributeOffsets: Uint32Array;
}

type SpatiallyIndexedSkeletonPickData =
  | {
      kind: "node";
      nodeIds: Int32Array;
      nodePositions: Float32Array;
      segmentIds: Uint32Array;
    }
  | {
      kind: "edge";
      segmentIds: Uint32Array;
    }
  | {
      kind: "segment-node";
      chunk: SpatiallyIndexedSkeletonChunk;
    }
  | {
      kind: "segment-edge";
      chunk: SpatiallyIndexedSkeletonChunk;
    };

class RenderHelper extends RefCounted {
  private textureAccessHelper = new OneDimensionalTextureAccessHelper(
    "vertexData",
  );
  private vertexIdHelper;
  private dynamicSegmentAppearance: boolean;
  private segmentAttributeIndex: number | undefined;
  private segmentColorAttributeIndex: number | undefined;
  private selectedNodeAttributeIndex: number | undefined;
  private visibleSegmentsShaderManager = new HashSetShaderManager(
    "visibleSegments",
  );
  private segmentColorShaderManager = new SegmentColorShaderManager(
    "segmentColorHash",
  );
  private segmentStatedColorShaderManager = new SegmentStatedColorShaderManager(
    "segmentStatedColor",
  );
  private gpuSegmentStatedColorHashTable: GPUHashTable<any> | undefined;
  get vertexAttributes(): VertexAttributeRenderInfo[] {
    return this.base.vertexAttributes;
  }

  defineCommonShader(builder: ShaderBuilder) {
    defineVertexId(builder);
    builder.addUniform("highp vec4", "uColor");
    builder.addUniform("highp mat4", "uProjection");
    builder.addUniform("highp uint", "uPickID");
  }

  private getSegmentColorExpression() {
    const index = this.segmentColorAttributeIndex;
    if (index === undefined) {
      return "uColor";
    }
    return `vCustom${index}`;
  }

  edgeShaderGetter;
  nodeShaderGetter;

  get gl(): GL {
    return this.base.gl;
  }

  disposed() {
    this.gpuSegmentStatedColorHashTable?.dispose();
    super.disposed();
  }

  private defineDynamicSegmentAppearance(builder: ShaderBuilder) {
    this.visibleSegmentsShaderManager.defineShader(builder);
    this.segmentColorShaderManager.defineShader(builder);
    this.segmentStatedColorShaderManager.defineShader(builder);
    builder.addUniform("highp float", "uVisibleAlpha");
    builder.addUniform("highp float", "uHiddenAlpha");
    builder.addUniform("highp vec3", "uSegmentDefaultColor");
    builder.addUniform("highp uint", "uSkipVisibleSegments");
    builder.addUniform("highp uint", "uUseSegmentDefaultColor");
    builder.addUniform("highp uint", "uUseSegmentStatedColors");
    builder.addFragmentCode(`
uint64_t getSegmentAppearanceId(highp uint segmentValue) {
  return uint64_t(uvec2(segmentValue, 0u));
}
vec3 getSegmentLookupColor(uint64_t segmentId) {
  vec4 statedColor;
  if (
    uUseSegmentStatedColors != 0u &&
    ${this.segmentStatedColorShaderManager.getFunctionName}(segmentId, statedColor)
  ) {
    return statedColor.rgb;
  }
  if (uUseSegmentDefaultColor != 0u) {
    return uSegmentDefaultColor;
  }
  return ${this.segmentColorShaderManager.prefix}(segmentId);
}
float getSegmentLookupAlpha(uint64_t segmentId) {
  bool isVisible = ${this.visibleSegmentsShaderManager.hasFunctionName}(segmentId);
  if (uSkipVisibleSegments != 0u && isVisible) {
    return 0.0;
  }
  return isVisible ? uVisibleAlpha : uHiddenAlpha;
}
vec4 getSegmentAppearance(highp uint segmentValue) {
  uint64_t segmentId = getSegmentAppearanceId(segmentValue);
  return vec4(getSegmentLookupColor(segmentId), getSegmentLookupAlpha(segmentId));
}
`);
  }

  enableDynamicSegmentAppearance(
    gl: GL,
    shader: ShaderProgram,
    skipVisibleSegments: boolean,
  ) {
    if (!this.dynamicSegmentAppearance) return;
    const segmentationGroupState =
      this.base.displayState.segmentationGroupState.value;
    const visibleSegments = segmentationGroupState.useTemporaryVisibleSegments
      .value
      ? segmentationGroupState.temporaryVisibleSegments
      : segmentationGroupState.visibleSegments;
    this.visibleSegmentsShaderManager.enable(
      gl,
      shader,
      GPUHashTable.get(gl, visibleSegments.hashTable),
    );
    gl.uniform1f(
      shader.uniform("uVisibleAlpha"),
      this.base.displayState.objectAlpha.value,
    );
    gl.uniform1f(
      shader.uniform("uHiddenAlpha"),
      this.base.displayState.hiddenObjectAlpha?.value ?? 0,
    );
    gl.uniform1ui(
      shader.uniform("uSkipVisibleSegments"),
      skipVisibleSegments ? 1 : 0,
    );

    const colorGroupState =
      this.base.displayState.segmentationColorGroupState.value;
    this.segmentColorShaderManager.enable(
      gl,
      shader,
      colorGroupState.segmentColorHash.value,
    );
    const segmentDefaultColor = colorGroupState.segmentDefaultColor.value;
    if (segmentDefaultColor === undefined) {
      gl.uniform1ui(shader.uniform("uUseSegmentDefaultColor"), 0);
    } else {
      gl.uniform1ui(shader.uniform("uUseSegmentDefaultColor"), 1);
      gl.uniform3f(
        shader.uniform("uSegmentDefaultColor"),
        segmentDefaultColor[0],
        segmentDefaultColor[1],
        segmentDefaultColor[2],
      );
    }

    const segmentStatedColors = colorGroupState.segmentStatedColors;
    if (segmentStatedColors.size === 0) {
      gl.uniform1ui(shader.uniform("uUseSegmentStatedColors"), 0);
      this.segmentStatedColorShaderManager.disable(gl, shader);
      return;
    }
    gl.uniform1ui(shader.uniform("uUseSegmentStatedColors"), 1);
    let { gpuSegmentStatedColorHashTable } = this;
    if (
      gpuSegmentStatedColorHashTable === undefined ||
      gpuSegmentStatedColorHashTable.hashTable !== segmentStatedColors.hashTable
    ) {
      gpuSegmentStatedColorHashTable?.dispose();
      this.gpuSegmentStatedColorHashTable = gpuSegmentStatedColorHashTable =
        GPUHashTable.get(gl, segmentStatedColors.hashTable);
    }
    this.segmentStatedColorShaderManager.enable(
      gl,
      shader,
      gpuSegmentStatedColorHashTable,
    );
  }

  disableDynamicSegmentAppearance(gl: GL, shader: ShaderProgram) {
    if (!this.dynamicSegmentAppearance) return;
    this.visibleSegmentsShaderManager.disable(gl, shader);
    this.segmentStatedColorShaderManager.disable(gl, shader);
  }

  constructor(
    public base: SkeletonLayerInterface,
    public targetIsSliceView: boolean,
  ) {
    super();
    this.vertexIdHelper = this.registerDisposer(VertexIdHelper.get(this.gl));
    const segmentAttrIndex = this.vertexAttributes.findIndex(
      (x) => x.name === segmentAttribute.name,
    );
    this.segmentAttributeIndex =
      segmentAttrIndex >= 0 ? segmentAttrIndex : undefined;
    this.dynamicSegmentAppearance =
      base.dynamicSegmentAppearance === true &&
      this.segmentAttributeIndex !== undefined;
    this.segmentColorAttributeIndex = base.segmentColorAttributeIndex;
    const selectedNodeAttrIndex = this.vertexAttributes.findIndex(
      (x) => x.name === selectedNodeAttribute.name,
    );
    this.selectedNodeAttributeIndex =
      selectedNodeAttrIndex >= 0 ? selectedNodeAttrIndex : undefined;
    this.edgeShaderGetter = parameterizedEmitterDependentShaderGetter(
      this,
      this.gl,
      {
        memoizeKey: {
          type: "skeleton/SkeletonShaderManager/edge",
          dynamicSegmentAppearance: this.dynamicSegmentAppearance,
          vertexAttributes: this.vertexAttributes,
        },
        fallbackParameters: this.base.fallbackShaderParameters,
        parameters:
          this.base.displayState.skeletonRenderingOptions.shaderControlState
            .builderState,
        shaderError: this.base.displayState.shaderError,
        defineShader: (
          builder: ShaderBuilder,
          shaderBuilderState: ShaderControlsBuilderState,
        ) => {
          if (shaderBuilderState.parseResult.errors.length !== 0) {
            throw new Error("Invalid UI control specification");
          }
          this.defineCommonShader(builder);
          this.defineAttributeAccess(builder);
          if (this.dynamicSegmentAppearance) {
            this.defineDynamicSegmentAppearance(builder);
          }
          defineLineShader(builder);
          builder.addAttribute("highp uvec2", "aVertexIndex");
          builder.addUniform("highp float", "uLineWidth");
          builder.addUniform("highp uint", "uPickInstanceStride");
          builder.addVarying("highp uint", "vPickID", "flat");
          if (this.dynamicSegmentAppearance) {
            builder.addVarying("highp uint", "vSegmentValue", "flat");
          }
          let vertexMain = `
highp uint pickOffset = uint(gl_InstanceID) * uPickInstanceStride;
vPickID = uPickID + pickOffset;
highp vec3 vertexA = readAttribute0(aVertexIndex.x);
highp vec3 vertexB = readAttribute0(aVertexIndex.y);
emitLine(uProjection, vertexA, vertexB, uLineWidth);
highp uint lineEndpointIndex = getLineEndpointIndex();
highp uint vertexIndex = aVertexIndex.x * (1u - lineEndpointIndex) + aVertexIndex.y * lineEndpointIndex;
`;
          if (
            this.dynamicSegmentAppearance &&
            this.segmentAttributeIndex !== undefined
          ) {
            vertexMain += `vSegmentValue = toRaw(readAttribute${this.segmentAttributeIndex}(aVertexIndex.x));\n`;
          }

          const segmentColorExpression = this.getSegmentColorExpression();
          const segmentAlphaExpression =
            this.segmentColorAttributeIndex === undefined
              ? "uColor.a"
              : `${segmentColorExpression}.a`;
          if (this.dynamicSegmentAppearance) {
            builder.addFragmentCode(`
vec4 segmentColor() {
  return getSegmentAppearance(vSegmentValue);
}
void emitRGB(vec3 color) {
  vec4 baseColor = segmentColor();
  highp float alpha = baseColor.a * getLineAlpha() * ${this.getCrossSectionFadeFactor()};
  if (alpha <= 0.0) discard;
  emit(vec4(color * alpha, alpha), vPickID);
}
void emitDefault() {
  vec4 baseColor = segmentColor();
  highp float alpha = baseColor.a * getLineAlpha() * ${this.getCrossSectionFadeFactor()};
  if (alpha <= 0.0) discard;
  emit(vec4(baseColor.rgb * alpha, alpha), vPickID);
}
`);
          } else if (this.segmentColorAttributeIndex === undefined) {
            // Preserve legacy skeleton behavior where `uColor` is already
            // premultiplied by `objectAlpha` in `getObjectColor`.
            builder.addFragmentCode(`
vec4 segmentColor() {
  return ${segmentColorExpression};
}
void emitRGB(vec3 color) {
  emit(vec4(color * uColor.a, uColor.a * getLineAlpha() * ${this.getCrossSectionFadeFactor()}), vPickID);
}
void emitDefault() {
  emit(vec4(uColor.rgb, uColor.a * getLineAlpha() * ${this.getCrossSectionFadeFactor()}), vPickID);
}
`);
          } else {
            builder.addFragmentCode(`
vec4 segmentColor() {
  return ${segmentColorExpression};
}
void emitRGB(vec3 color) {
  highp float alpha = ${segmentAlphaExpression} * getLineAlpha() * ${this.getCrossSectionFadeFactor()};
  emit(vec4(color * alpha, alpha), vPickID);
}
void emitDefault() {
  vec4 baseColor = segmentColor();
  highp float alpha = baseColor.a * getLineAlpha() * ${this.getCrossSectionFadeFactor()};
  emit(vec4(baseColor.rgb * alpha, alpha), vPickID);
}
`);
          }
          builder.addFragmentCode(glsl_COLORMAPS);
          const { vertexAttributes } = this;
          const numAttributes = vertexAttributes.length;
          for (let i = 1; i < numAttributes; ++i) {
            const info = vertexAttributes[i];
            if (
              this.dynamicSegmentAppearance &&
              i === this.segmentAttributeIndex
            ) {
              builder.addFragmentCode(dataTypeShaderDefinition[info.dataType]);
              builder.addFragmentCode(
                `#define ${info.name} ${info.glslDataType}(vSegmentValue)\n`,
              );
              builder.addFragmentCode(
                `#define prop_${info.name}() ${info.glslDataType}(vSegmentValue)\n`,
              );
              continue;
            }
            builder.addVarying(
              `highp ${getVertexAttributeVaryingType(info)}`,
              `vCustom${i}`,
              getVertexAttributeInterpolationMode(info.dataType),
            );
            vertexMain += `vCustom${i} = ${getVertexAttributeReadExpression(i, "vertexIndex", info)};\n`;
            if (info.dataType !== DataType.FLOAT32) {
              builder.addFragmentCode(dataTypeShaderDefinition[info.dataType]);
            }
            const fragmentExpression = getVertexAttributeFragmentExpression(
              `vCustom${i}`,
              info,
            );
            builder.addFragmentCode(
              `#define ${info.name} ${fragmentExpression}\n`,
            );
            builder.addFragmentCode(
              `#define prop_${info.name}() ${fragmentExpression}\n`,
            );
          }
          builder.setVertexMain(vertexMain);
          addControlsToBuilder(shaderBuilderState, builder);
          const edgeFragmentCode = shaderCodeWithLineDirective(
            shaderBuilderState.parseResult.code,
          );
          builder.setFragmentMainFunction(edgeFragmentCode);
        },
      },
    );

    this.nodeShaderGetter = parameterizedEmitterDependentShaderGetter(
      this,
      this.gl,
      {
        memoizeKey: {
          type: "skeleton/SkeletonShaderManager/node",
          dynamicSegmentAppearance: this.dynamicSegmentAppearance,
          vertexAttributes: this.vertexAttributes,
        },
        fallbackParameters: this.base.fallbackShaderParameters,
        parameters:
          this.base.displayState.skeletonRenderingOptions.shaderControlState
            .builderState,
        shaderError: this.base.displayState.shaderError,
        defineShader: (
          builder: ShaderBuilder,
          shaderBuilderState: ShaderControlsBuilderState,
        ) => {
          if (shaderBuilderState.parseResult.errors.length !== 0) {
            throw new Error("Invalid UI control specification");
          }
          this.defineCommonShader(builder);
          this.defineAttributeAccess(builder);
          if (this.dynamicSegmentAppearance) {
            this.defineDynamicSegmentAppearance(builder);
          }
          defineCircleShader(
            builder,
            /*crossSectionFade=*/ this.targetIsSliceView,
          );
          builder.addUniform("highp float", "uNodeDiameter");
          builder.addUniform("highp uint", "uPickInstanceStride");
          builder.addVarying("highp uint", "vPickID", "flat");
          if (this.dynamicSegmentAppearance) {
            builder.addVarying("highp uint", "vSegmentValue", "flat");
          }
          const selectedOutlineMinWidth = this.targetIsSliceView
            ? SELECTED_NODE_OUTLINE_MIN_WIDTH_2D
            : SELECTED_NODE_OUTLINE_MIN_WIDTH_3D;
          const selectedOutlineMaxWidth = this.targetIsSliceView
            ? SELECTED_NODE_OUTLINE_MAX_WIDTH_2D
            : SELECTED_NODE_OUTLINE_MAX_WIDTH_3D;
          const selectedNodeAttributeReadExpression =
            this.selectedNodeAttributeIndex === undefined
              ? "0.0"
              : `readAttribute${this.selectedNodeAttributeIndex}(vertexIndex)`;
          const selectedOutlineWidthExpression =
            this.selectedNodeAttributeIndex === undefined
              ? "0.0"
              : `((${selectedNodeAttributeReadExpression} > 0.5) ? clamp(0.25 * uNodeDiameter, ${selectedOutlineMinWidth}, ${selectedOutlineMaxWidth}) : 0.0)`;
          let vertexMain = `
highp uint vertexIndex = uint(gl_InstanceID);
highp uint pickOffset = vertexIndex * uPickInstanceStride;
vPickID = uPickID + pickOffset;
highp vec3 vertexPosition = readAttribute0(vertexIndex);
emitCircle(
  uProjection * vec4(vertexPosition, 1.0),
  uNodeDiameter,
  ${selectedOutlineWidthExpression}
);
`;
          if (
            this.dynamicSegmentAppearance &&
            this.segmentAttributeIndex !== undefined
          ) {
            vertexMain += `vSegmentValue = toRaw(readAttribute${this.segmentAttributeIndex}(vertexIndex));\n`;
          }

          const segmentColorExpression = this.getSegmentColorExpression();
          if (
            this.dynamicSegmentAppearance &&
            this.segmentAttributeIndex !== undefined
          ) {
            const segmentExpression = `vSegmentValue`;
            const selectedNodeExpression =
              this.selectedNodeAttributeIndex === undefined
                ? undefined
                : `vCustom${this.selectedNodeAttributeIndex}`;
            const borderColorExpression =
              selectedNodeExpression === undefined
                ? "renderColor"
                : `((${selectedNodeExpression} > 0.5) ? vec4(${SELECTED_NODE_OUTLINE_COLOR_RGB}, renderColor.a) : renderColor)`;
            builder.addFragmentCode(`
vec4 segmentColor() {
  return getSegmentAppearance(${segmentExpression});
}
void emitRGBA(vec4 color) {
  vec4 baseColor = segmentColor();
  highp float alpha = color.a * baseColor.a;
  if (alpha <= 0.0) discard;
  vec4 renderColor = vec4(color.rgb, alpha);
  vec4 borderColor = ${borderColorExpression};
  vec4 circleColor = getCircleColor(renderColor, borderColor);
  emit(vec4(circleColor.rgb * circleColor.a, circleColor.a), vPickID);
}
void emitRGB(vec3 color) {
  emitRGBA(vec4(color, 1.0));
}
void emitDefault() {
  vec4 baseColor = segmentColor();
  highp float alpha = baseColor.a;
  if (alpha <= 0.0) discard;
  vec4 renderColor = vec4(baseColor.rgb, alpha);
  vec4 borderColor = ${borderColorExpression};
  vec4 circleColor = getCircleColor(renderColor, borderColor);
  emit(vec4(circleColor.rgb * circleColor.a, circleColor.a), vPickID);
}
`);
          } else if (this.segmentColorAttributeIndex === undefined) {
            // Preserve legacy skeleton behavior for non-spatial skeletons.
            builder.addFragmentCode(`
vec4 segmentColor() {
  return ${segmentColorExpression};
}
void emitRGBA(vec4 color) {
  vec4 borderColor = color;
  emit(getCircleColor(color, borderColor), vPickID);
}
void emitRGB(vec3 color) {
  emitRGBA(vec4(color, 1.0));
}
void emitDefault() {
  emitRGBA(uColor);
}
`);
          } else {
            const selectedNodeExpression =
              this.selectedNodeAttributeIndex === undefined
                ? undefined
                : `vCustom${this.selectedNodeAttributeIndex}`;
            const borderColorExpression =
              selectedNodeExpression === undefined
                ? "renderColor"
                : `((${selectedNodeExpression} > 0.5) ? vec4(${SELECTED_NODE_OUTLINE_COLOR_RGB}, renderColor.a) : renderColor)`;
            builder.addFragmentCode(`
vec4 segmentColor() {
  return ${segmentColorExpression};
}
void emitRGBA(vec4 color) {
  vec4 renderColor = color;
  vec4 borderColor = ${borderColorExpression};
  vec4 circleColor = getCircleColor(renderColor, borderColor);
  emit(vec4(circleColor.rgb * circleColor.a, circleColor.a), vPickID);
}
void emitRGB(vec3 color) {
  emitRGBA(vec4(color, 1.0));
}
void emitDefault() {
  emitRGBA(segmentColor());
}
`);
          }
          builder.addFragmentCode(glsl_COLORMAPS);
          const { vertexAttributes } = this;
          const numAttributes = vertexAttributes.length;
          for (let i = 1; i < numAttributes; ++i) {
            const info = vertexAttributes[i];
            if (
              this.dynamicSegmentAppearance &&
              i === this.segmentAttributeIndex
            ) {
              builder.addFragmentCode(dataTypeShaderDefinition[info.dataType]);
              builder.addFragmentCode(
                `#define ${info.name} ${info.glslDataType}(vSegmentValue)\n`,
              );
              builder.addFragmentCode(
                `#define prop_${info.name}() ${info.glslDataType}(vSegmentValue)\n`,
              );
              continue;
            }
            builder.addVarying(
              `highp ${getVertexAttributeVaryingType(info)}`,
              `vCustom${i}`,
              getVertexAttributeInterpolationMode(info.dataType),
            );
            vertexMain += `vCustom${i} = ${getVertexAttributeReadExpression(i, "vertexIndex", info)};\n`;
            if (info.dataType !== DataType.FLOAT32) {
              builder.addFragmentCode(dataTypeShaderDefinition[info.dataType]);
            }
            const fragmentExpression = getVertexAttributeFragmentExpression(
              `vCustom${i}`,
              info,
            );
            builder.addFragmentCode(
              `#define ${info.name} ${fragmentExpression}\n`,
            );
            builder.addFragmentCode(
              `#define prop_${info.name}() ${fragmentExpression}\n`,
            );
          }
          builder.setVertexMain(vertexMain);
          addControlsToBuilder(shaderBuilderState, builder);
          builder.setFragmentMainFunction(
            shaderCodeWithLineDirective(shaderBuilderState.parseResult.code),
          );
        },
      },
    );
  }

  defineAttributeAccess(builder: ShaderBuilder) {
    const { textureAccessHelper } = this;
    textureAccessHelper.defineShader(builder);
    const numAttributes = this.vertexAttributes.length;
    for (let j = vertexAttributeSamplerSymbols.length; j < numAttributes; ++j) {
      vertexAttributeSamplerSymbols[j] = Symbol(
        `SkeletonShader.vertexAttributeTextureUnit${j}`,
      );
    }
    this.vertexAttributes.forEach((info, i) => {
      builder.addTextureSampler(
        `${getSamplerPrefixForDataType(
          info.dataType,
        )}sampler2D` as ShaderSamplerType,
        `uVertexAttributeSampler${i}`,
        vertexAttributeSamplerSymbols[i],
      );
      builder.addVertexCode(
        textureAccessHelper.getAccessor(
          `readAttribute${i}`,
          `uVertexAttributeSampler${i}`,
          info.dataType,
          info.numComponents,
        ),
      );
    });
  }

  getCrossSectionFadeFactor() {
    if (this.targetIsSliceView) {
      return "(clamp(1.0 - 2.0 * abs(0.5 - gl_FragCoord.z), 0.0, 1.0))";
    }
    return "(1.0)";
  }

  beginLayer(
    gl: GL,
    shader: ShaderProgram,
    renderContext: SliceViewPanelRenderContext | PerspectiveViewRenderContext,
    modelMatrix: mat4,
  ) {
    const { viewProjectionMat } = renderContext.projectionParameters;
    const mat = mat4.multiply(tempMat2, viewProjectionMat, modelMatrix);
    gl.uniformMatrix4fv(shader.uniform("uProjection"), false, mat);
    this.vertexIdHelper.enable();
  }

  setColor(gl: GL, shader: ShaderProgram, color: Float32Array | number[]) {
    const a =
      (color as Float32Array).length >= 4
        ? (color as Float32Array)[3]
        : this.base.displayState.objectAlpha.value;
    gl.uniform4f(
      shader.uniform("uColor"),
      (color as Float32Array)[0],
      (color as Float32Array)[1],
      (color as Float32Array)[2],
      a,
    );
  }

  setPickID(gl: GL, shader: ShaderProgram, pickID: number) {
    gl.uniform1ui(shader.uniform("uPickID"), pickID);
  }

  setEdgePickInstanceStride(gl: GL, shader: ShaderProgram, stride: number) {
    gl.uniform1ui(shader.uniform("uPickInstanceStride"), stride);
  }

  setNodePickInstanceStride(gl: GL, shader: ShaderProgram, stride: number) {
    gl.uniform1ui(shader.uniform("uPickInstanceStride"), stride);
  }

  private bindVertexAttributeTextures(
    gl: GL,
    shader: ShaderProgram,
    vertexAttributeTextures: readonly (WebGLTexture | null)[],
  ) {
    const { vertexAttributes } = this;
    const numAttributes = vertexAttributes.length;
    for (let i = 0; i < numAttributes; ++i) {
      const textureUnit =
        WebGL2RenderingContext.TEXTURE0 +
        shader.textureUnit(vertexAttributeSamplerSymbols[i]);
      gl.activeTexture(textureUnit);
      gl.bindTexture(
        WebGL2RenderingContext.TEXTURE_2D,
        vertexAttributeTextures[i],
      );
    }
  }

  drawSkeleton(
    gl: GL,
    edgeShader: ShaderProgram,
    nodeShader: ShaderProgram | null,
    skeletonChunk: SkeletonChunkInterface,
    projectionParameters: { width: number; height: number },
  ) {
    const { vertexAttributeTextures } = skeletonChunk;

    // Draw edges
    {
      edgeShader.bind();
      this.bindVertexAttributeTextures(gl, edgeShader, vertexAttributeTextures);
      const aVertexIndex = edgeShader.attribute("aVertexIndex");
      skeletonChunk.indexBuffer.bindToVertexAttribI(
        aVertexIndex,
        2,
        WebGL2RenderingContext.UNSIGNED_INT,
      );
      gl.vertexAttribDivisor(aVertexIndex, 1);
      initializeLineShader(
        edgeShader,
        projectionParameters,
        this.targetIsSliceView ? 1.0 : 0.0,
      );
      drawLines(gl, 1, skeletonChunk.numIndices / 2);
      gl.vertexAttribDivisor(aVertexIndex, 0);
      gl.disableVertexAttribArray(aVertexIndex);
    }

    if (nodeShader !== null) {
      nodeShader.bind();
      // Node and edge programs can allocate sampler units differently.
      this.bindVertexAttributeTextures(gl, nodeShader, vertexAttributeTextures);
      initializeCircleShader(nodeShader, projectionParameters, {
        featherWidthInPixels: this.targetIsSliceView ? 1.0 : 0.0,
      });
      drawCircles(nodeShader.gl, 2, skeletonChunk.numVertices);
    }
  }

  endLayer(gl: GL, ...shaders: Array<ShaderProgram | null>) {
    const { vertexAttributes } = this;
    const numAttributes = vertexAttributes.length;
    const clearedTextureUnits = new Set<number>();
    for (const shader of shaders) {
      if (shader === null) continue;
      for (let i = 0; i < numAttributes; ++i) {
        const curTextureUnit =
          shader.textureUnit(vertexAttributeSamplerSymbols[i]) +
          WebGL2RenderingContext.TEXTURE0;
        if (clearedTextureUnits.has(curTextureUnit)) continue;
        clearedTextureUnits.add(curTextureUnit);
        gl.activeTexture(curTextureUnit);
        gl.bindTexture(gl.TEXTURE_2D, null);
      }
    }
    this.vertexIdHelper.disable();
  }
}

export enum SkeletonRenderMode {
  LINES = 0,
  LINES_AND_POINTS = 1,
}

export class TrackableSkeletonRenderMode extends TrackableEnum<SkeletonRenderMode> {
  constructor(
    value: SkeletonRenderMode,
    defaultValue: SkeletonRenderMode = value,
  ) {
    super(SkeletonRenderMode, value, defaultValue);
  }
}

export class TrackableSkeletonLineWidth extends TrackableValue<number> {
  constructor(value: number, defaultValue: number = value) {
    super(value, verifyFinitePositiveFloat, defaultValue);
  }
}

function getSkeletonNodeDiameter(
  renderMode: SkeletonRenderMode,
  lineWidth: number,
) {
  if (renderMode === SkeletonRenderMode.LINES_AND_POINTS) {
    return Math.max(5, lineWidth * 2);
  }
  return lineWidth;
}

function snapMouseStateToSpatialSkeletonNode(
  mouseState: MouseSelectionState,
  nodePositions: Float32Array,
  pickedOffset: number,
) {
  const sourceOffset = pickedOffset * 3;
  if (sourceOffset + 2 >= nodePositions.length) {
    return;
  }
  const x = nodePositions[sourceOffset];
  const y = nodePositions[sourceOffset + 1];
  const z = nodePositions[sourceOffset + 2];
  if (!Number.isFinite(x) || !Number.isFinite(y) || !Number.isFinite(z)) {
    return;
  }
  const { position } = mouseState;
  if (position.length > 0) {
    position[0] = x;
  }
  if (position.length > 1) {
    position[1] = y;
  }
  if (position.length > 2) {
    position[2] = z;
  }
}

export interface ViewSpecificSkeletonRenderingOptions {
  mode: TrackableSkeletonRenderMode;
  lineWidth: TrackableSkeletonLineWidth;
}

export class SkeletonRenderingOptions implements Trackable {
  private compound = new CompoundTrackable();
  get changed() {
    return this.compound.changed;
  }

  shader = makeTrackableFragmentMain(DEFAULT_FRAGMENT_MAIN);
  shaderControlState = new ShaderControlState(this.shader);
  params2d: ViewSpecificSkeletonRenderingOptions = {
    mode: new TrackableSkeletonRenderMode(SkeletonRenderMode.LINES_AND_POINTS),
    lineWidth: new TrackableSkeletonLineWidth(2),
  };
  params3d: ViewSpecificSkeletonRenderingOptions = {
    mode: new TrackableSkeletonRenderMode(SkeletonRenderMode.LINES),
    lineWidth: new TrackableSkeletonLineWidth(1),
  };

  constructor() {
    const { compound } = this;
    compound.add("shader", this.shader);
    compound.add("shaderControls", this.shaderControlState);
    compound.add("mode2d", this.params2d.mode);
    compound.add("lineWidth2d", this.params2d.lineWidth);
    compound.add("mode3d", this.params3d.mode);
    compound.add("lineWidth3d", this.params3d.lineWidth);
  }

  reset() {
    this.compound.reset();
  }

  restoreState(obj: any) {
    if (obj === undefined) return;
    this.compound.restoreState(obj);
  }

  toJSON(): any {
    const obj = this.compound.toJSON();
    for (const v of Object.values(obj)) {
      if (v !== undefined) return obj;
    }
    return undefined;
  }
}

export interface SkeletonLayerDisplayState extends SegmentationDisplayState3D {
  shaderError: WatchableShaderError;
  skeletonRenderingOptions: SkeletonRenderingOptions;
}

export class SkeletonLayer extends RefCounted {
  layerChunkProgressInfo = new LayerChunkProgressInfo();
  redrawNeeded = new NullarySignal();
  private sharedObject: SegmentationLayerSharedObject;
  vertexAttributes: VertexAttributeRenderInfo[];
  segmentColorAttributeIndex: number | undefined = undefined;
  fallbackShaderParameters = new WatchableValue(
    getFallbackBuilderState(parseShaderUiControls(DEFAULT_FRAGMENT_MAIN)),
  );

  get visibility() {
    return this.sharedObject.visibility;
  }

  constructor(
    public chunkManager: ChunkManager,
    public source: SkeletonSource,
    public displayState: SkeletonLayerDisplayState,
  ) {
    super();

    registerRedrawWhenSegmentationDisplayState3DChanged(displayState, this);
    this.displayState.shaderError.value = undefined;
    const { skeletonRenderingOptions: renderingOptions } = displayState;
    this.registerDisposer(
      renderingOptions.shader.changed.add(() => {
        this.displayState.shaderError.value = undefined;
        this.redrawNeeded.dispatch();
      }),
    );
    const sharedObject = (this.sharedObject = this.registerDisposer(
      new SegmentationLayerSharedObject(
        chunkManager,
        displayState,
        this.layerChunkProgressInfo,
      ),
    ));
    sharedObject.RPC_TYPE_ID = SKELETON_LAYER_RPC_ID;
    sharedObject.initializeCounterpartWithChunkManager({
      source: source.addCounterpartRef(),
    });

    const vertexAttributes = (this.vertexAttributes = [
      vertexPositionAttribute,
    ]);

    for (const [name, info] of source.vertexAttributes) {
      vertexAttributes.push({
        name,
        dataType: info.dataType,
        numComponents: info.numComponents,
        webglDataType: getWebglDataType(info.dataType),
        glslDataType: getShaderType(info.dataType, info.numComponents),
      });
    }
  }

  get gl() {
    return this.chunkManager.chunkQueueManager.gl;
  }

  draw(
    renderContext: SliceViewPanelRenderContext | PerspectiveViewRenderContext,
    layer: RenderLayer,
    renderHelper: RenderHelper,
    renderOptions: ViewSpecificSkeletonRenderingOptions,
    attachment: VisibleLayerInfo<
      LayerView,
      ThreeDimensionalRenderLayerAttachmentState
    >,
  ) {
    const lineWidth = renderOptions.lineWidth.value;
    const { gl, displayState, source } = this;
    if (displayState.objectAlpha.value <= 0.0) {
      // Skip drawing.
      return;
    }
    const modelMatrix = update3dRenderLayerAttachment(
      displayState.transform.value,
      renderContext.projectionParameters.displayDimensionRenderInfo,
      attachment,
    );
    if (modelMatrix === undefined) return;
    const pointDiameter = getSkeletonNodeDiameter(
      renderOptions.mode.value,
      lineWidth,
    );

    const edgeShaderResult = renderHelper.edgeShaderGetter(
      renderContext.emitter,
    );
    const nodeShaderResult = renderHelper.nodeShaderGetter(
      renderContext.emitter,
    );
    const { shader: edgeShader, parameters: edgeShaderParameters } =
      edgeShaderResult;
    const { shader: nodeShader, parameters: nodeShaderParameters } =
      nodeShaderResult;
    if (edgeShader === null || nodeShader === null) {
      // Shader error, skip drawing.
      return;
    }

    const { shaderControlState } = this.displayState.skeletonRenderingOptions;

    edgeShader.bind();
    renderHelper.beginLayer(gl, edgeShader, renderContext, modelMatrix);
    renderHelper.setEdgePickInstanceStride(gl, edgeShader, 0);
    setControlsInShader(
      gl,
      edgeShader,
      shaderControlState,
      edgeShaderParameters.parseResult.controls,
    );
    gl.uniform1f(edgeShader.uniform("uLineWidth"), lineWidth!);

    nodeShader.bind();
    renderHelper.beginLayer(gl, nodeShader, renderContext, modelMatrix);
    gl.uniform1f(nodeShader.uniform("uNodeDiameter"), pointDiameter);
    renderHelper.setNodePickInstanceStride(gl, nodeShader, 0);
    setControlsInShader(
      gl,
      nodeShader,
      shaderControlState,
      nodeShaderParameters.parseResult.controls,
    );

    const skeletons = source.chunks;

    forEachVisibleSegmentToDraw(
      displayState,
      layer,
      renderContext.emitColor,
      renderContext.emitPickID ? renderContext.pickIDs : undefined,
      (objectId, color, pickIndex) => {
        const key = getObjectKey(objectId);
        const skeleton = skeletons.get(key);
        if (
          skeleton === undefined ||
          skeleton.state !== ChunkState.GPU_MEMORY
        ) {
          return;
        }
        if (color !== undefined) {
          edgeShader.bind();
          renderHelper.setColor(gl, edgeShader, color as Float32Array);
          nodeShader.bind();
          renderHelper.setColor(gl, nodeShader, color as Float32Array);
        }
        if (pickIndex !== undefined) {
          edgeShader.bind();
          renderHelper.setPickID(gl, edgeShader, pickIndex);
          nodeShader.bind();
          renderHelper.setPickID(gl, nodeShader, pickIndex);
        }
        renderHelper.drawSkeleton(
          gl,
          edgeShader,
          nodeShader,
          skeleton,
          renderContext.projectionParameters,
        );
      },
    );
    renderHelper.endLayer(gl, edgeShader, nodeShader);
  }

  isReady() {
    const { source, displayState } = this;
    if (displayState.objectAlpha.value <= 0.0) {
      // Skip drawing.
      return true;
    }

    const skeletons = source.chunks;

    let ready = true;

    forEachVisibleSegment(
      displayState.segmentationGroupState.value,
      (objectId) => {
        const key = getObjectKey(objectId);
        const skeleton = skeletons.get(key);
        if (
          skeleton === undefined ||
          skeleton.state !== ChunkState.GPU_MEMORY
        ) {
          ready = false;
          return;
        }
      },
    );
    return ready;
  }
}

export class PerspectiveViewSkeletonLayer extends PerspectiveViewRenderLayer {
  private renderHelper: RenderHelper;
  private renderOptions: ViewSpecificSkeletonRenderingOptions;
  constructor(public base: SkeletonLayer) {
    super();
    this.renderHelper = this.registerDisposer(new RenderHelper(base, false));
    this.renderOptions = base.displayState.skeletonRenderingOptions.params3d;

    this.layerChunkProgressInfo = base.layerChunkProgressInfo;
    this.registerDisposer(base);
    this.registerDisposer(base.redrawNeeded.add(this.redrawNeeded.dispatch));
    const { renderOptions } = this;
    this.registerDisposer(
      renderOptions.mode.changed.add(this.redrawNeeded.dispatch),
    );
    this.registerDisposer(
      renderOptions.lineWidth.changed.add(this.redrawNeeded.dispatch),
    );
    this.registerDisposer(base.visibility.add(this.visibility));
  }
  get gl() {
    return this.base.gl;
  }

  get isTransparent() {
    return this.base.displayState.objectAlpha.value < 1.0;
  }

  draw(
    renderContext: PerspectiveViewRenderContext,
    attachment: VisibleLayerInfo<
      PerspectivePanel,
      ThreeDimensionalRenderLayerAttachmentState
    >,
  ) {
    if (!renderContext.emitColor && renderContext.alreadyEmittedPickID) {
      // No need for a separate pick ID pass.
      return;
    }
    this.base.draw(
      renderContext,
      this,
      this.renderHelper,
      this.renderOptions,
      attachment,
    );
  }

  isReady() {
    return this.base.isReady();
  }
}

export class SliceViewPanelSkeletonLayer extends SliceViewPanelRenderLayer {
  private renderHelper: RenderHelper;
  private renderOptions: ViewSpecificSkeletonRenderingOptions;
  constructor(public base: SkeletonLayer) {
    super();
    this.renderHelper = this.registerDisposer(new RenderHelper(base, true));
    this.renderOptions = base.displayState.skeletonRenderingOptions.params2d;
    this.layerChunkProgressInfo = base.layerChunkProgressInfo;
    this.registerDisposer(base);
    const { renderOptions } = this;
    this.registerDisposer(
      renderOptions.mode.changed.add(this.redrawNeeded.dispatch),
    );
    this.registerDisposer(
      renderOptions.lineWidth.changed.add(this.redrawNeeded.dispatch),
    );
    this.registerDisposer(base.redrawNeeded.add(this.redrawNeeded.dispatch));
    this.registerDisposer(base.visibility.add(this.visibility));
  }
  get gl() {
    return this.base.gl;
  }

  draw(
    renderContext: SliceViewPanelRenderContext,
    attachment: VisibleLayerInfo<
      SliceViewPanel,
      ThreeDimensionalRenderLayerAttachmentState
    >,
  ) {
    this.base.draw(
      renderContext,
      this,
      this.renderHelper,
      this.renderOptions,
      attachment,
    );
  }

  isReady() {
    return this.base.isReady();
  }
}

function getWebglDataType(dataType: DataType) {
  switch (dataType) {
    case DataType.FLOAT32:
      return WebGL2RenderingContext.FLOAT;
    case DataType.INT32:
      return WebGL2RenderingContext.INT;
    case DataType.UINT32:
      return WebGL2RenderingContext.UNSIGNED_INT;
    default:
      throw new Error(
        `Data type not supported by WebGL: ${DataType[dataType]}`,
      );
  }
}

function getVertexAttributeInterpolationMode(dataType: DataType) {
  return dataType === DataType.FLOAT32 ? "" : "flat";
}

// Custom integer wrapper types like `uint32_t` are defined in fragment code,
// which is emitted after varying declarations. Keep varyings on raw GLSL
// scalar/vector types and wrap them back into helper structs in fragment code.
function getVertexAttributeVaryingType(info: VertexAttributeInfo) {
  const { dataType, numComponents } = info;
  if (dataType === DataType.FLOAT32) {
    return getShaderType(dataType, numComponents);
  }
  if (dataType === DataType.UINT64) {
    if (numComponents === 1) return "uvec2";
    if (numComponents === 2) return "uvec4";
  }
  const vectorTypePrefix = DATA_TYPE_SIGNED[dataType] ? "ivec" : "uvec";
  if (numComponents === 1) {
    return DATA_TYPE_SIGNED[dataType] ? "int" : "uint";
  }
  if (numComponents >= 2 && numComponents <= 4) {
    return `${vectorTypePrefix}${numComponents}`;
  }
  throw new Error(
    `No varying type for ${DataType[dataType]}[${numComponents}].`,
  );
}

function getVertexAttributeReadExpression(
  attributeIndex: number,
  indexExpression: string,
  info: VertexAttributeInfo,
) {
  const readExpression = `readAttribute${attributeIndex}(${indexExpression})`;
  if (info.dataType === DataType.FLOAT32) {
    return readExpression;
  }
  if (info.dataType === DataType.UINT64) {
    return `${readExpression}.value`;
  }
  return `toRaw(${readExpression})`;
}

function getVertexAttributeFragmentExpression(
  varyingName: string,
  info: VertexAttributeRenderInfo,
) {
  if (info.dataType === DataType.FLOAT32) {
    return varyingName;
  }
  return `${info.glslDataType}(${varyingName})`;
}

const vertexPositionAttribute: VertexAttributeRenderInfo = {
  dataType: DataType.FLOAT32,
  numComponents: 3,
  name: "",
  webglDataType: WebGL2RenderingContext.FLOAT,
  glslDataType: "vec3",
};

const segmentAttribute: VertexAttributeRenderInfo = {
  dataType: DataType.UINT32,
  numComponents: 1,
  name: "segment",
  webglDataType: WebGL2RenderingContext.UNSIGNED_INT,
  glslDataType: getShaderType(DataType.UINT32, 1),
};

const selectedNodeAttribute: VertexAttributeRenderInfo = {
  dataType: DataType.FLOAT32,
  numComponents: 1,
  name: "selectedNodeAttr",
  webglDataType: WebGL2RenderingContext.FLOAT,
  glslDataType: "float",
};

export class SkeletonChunk extends Chunk implements SkeletonChunkInterface {
  declare source: SkeletonSource;
  vertexAttributes: Uint8Array;
  indices: Uint32Array;
  indexBuffer: GLBuffer;
  numIndices: number;
  numVertices: number;
  vertexAttributeOffsets: Uint32Array;
  vertexAttributeTextures: (WebGLTexture | null)[];

  constructor(source: SkeletonSource, x: any) {
    super(source);
    this.vertexAttributes = x.vertexAttributes;
    const indices = (this.indices = x.indices);
    this.numVertices = x.numVertices;
    this.vertexAttributeOffsets = x.vertexAttributeOffsets;
    this.numIndices = indices.length;
  }

  copyToGPU(gl: GL) {
    super.copyToGPU(gl);
    const { attributeTextureFormats } = this.source;
    const { vertexAttributes, vertexAttributeOffsets } = this;

    this.vertexAttributeTextures = uploadVertexAttributesToGPU(
      gl,
      vertexAttributes,
      vertexAttributeOffsets,
      attributeTextureFormats,
    );

    this.indexBuffer = GLBuffer.fromData(
      gl,
      this.indices,
      WebGL2RenderingContext.ARRAY_BUFFER,
      WebGL2RenderingContext.STATIC_DRAW,
    );
  }

  freeGPUMemory(gl: GL) {
    super.freeGPUMemory(gl);
    const { vertexAttributeTextures } = this;
    for (const texture of vertexAttributeTextures) {
      gl.deleteTexture(texture);
    }
    vertexAttributeTextures.length = 0;
    this.indexBuffer.dispose();
  }
}

export class SpatiallyIndexedSkeletonChunk
  extends SliceViewChunk
  implements SkeletonChunkInterface
{
  declare source: SpatiallyIndexedSkeletonSource;
  vertexAttributes: Uint8Array;
  indices: Uint32Array;
  indexBuffer: GLBuffer;
  numIndices: number;
  numVertices: number;
  vertexAttributeOffsets: Uint32Array;
  vertexAttributeTextures: (WebGLTexture | null)[];
  missingConnections: Array<{
    nodeId: number;
    parentId: number;
    vertexIndex: number;
    skeletonId: number;
  }> = [];
  nodeMap: Map<number, number> = new Map(); // Maps node ID to vertex index
  lod: number | undefined;

  // Filtering support
  filteredIndexBuffer: GLBuffer | undefined;
  filteredGeneration: number = -1;
  filteredMissingConnectionsHash = 0;
  filteredNodeLocatorIndexKey: string | undefined;
  numFilteredIndices: number = 0;
  numFilteredVertices: number = 0;
  filteredEmpty = false;
  filteredVertexAttributeTextures?: (WebGLTexture | null)[];
  filteredPickNodeIds?: Int32Array;
  filteredPickNodePositions?: Float32Array;
  filteredPickSegmentIds?: Uint32Array;
  filteredPickEdgeSegmentIds?: Uint32Array;
  filteredOldToNew?: Int32Array;
  filteredExtraVertexMap?: Map<number, number>;

  constructor(
    source: SpatiallyIndexedSkeletonSource,
    chunkData: SkeletonChunkData,
  ) {
    super(source, chunkData);
    this.vertexAttributes = chunkData.vertexAttributes;
    const indices = (this.indices = chunkData.indices);
    this.numVertices = chunkData.numVertices;
    this.numIndices = indices.length;
    this.vertexAttributeOffsets = chunkData.vertexAttributeOffsets;
    this.missingConnections = (chunkData as any).missingConnections || [];
    this.lod = (chunkData as any).lod;

    // Deserialize nodeMap from array format [nodeId, vertexIndex, ...]
    const nodeMapData = (chunkData as any).nodeMap;
    if (Array.isArray(nodeMapData) && nodeMapData.length > 0) {
      this.nodeMap = new Map(nodeMapData);
    } else {
      this.nodeMap = new Map();
    }

    const gl = source.gl;
    this.indexBuffer = GLBuffer.fromData(
      gl,
      indices,
      WebGL2RenderingContext.ARRAY_BUFFER,
      WebGL2RenderingContext.STATIC_DRAW,
    );

    const { attributeTextureFormats } = source;
    this.vertexAttributeTextures = uploadVertexAttributesToGPU(
      gl,
      this.vertexAttributes,
      this.vertexAttributeOffsets,
      attributeTextureFormats,
    );
  }

  copyToGPU(gl: GL) {
    const wasGpuResident = this.state === ChunkState.GPU_MEMORY;
    super.copyToGPU(gl);
    if (!wasGpuResident) {
      this.source.bumpLookupGeneration();
    }
  }

  freeGPUMemory(gl: GL) {
    const wasGpuResident = this.state === ChunkState.GPU_MEMORY;
    super.freeGPUMemory(gl);
    this.indexBuffer.dispose();
    const { vertexAttributeTextures } = this;
    for (let i = 0, length = vertexAttributeTextures.length; i < length; ++i) {
      gl.deleteTexture(vertexAttributeTextures[i]);
    }

    if (this.filteredIndexBuffer) {
      this.filteredIndexBuffer.dispose();
      this.filteredIndexBuffer = undefined;
    }

    if (this.filteredVertexAttributeTextures) {
      for (const tex of this.filteredVertexAttributeTextures) {
        if (tex) gl.deleteTexture(tex);
      }
      this.filteredVertexAttributeTextures = undefined;
    }
    this.filteredPickNodeIds = undefined;
    this.filteredPickNodePositions = undefined;
    this.filteredPickSegmentIds = undefined;
    this.filteredPickEdgeSegmentIds = undefined;
    this.filteredOldToNew = undefined;
    this.filteredExtraVertexMap = undefined;
    this.filteredEmpty = false;
    if (wasGpuResident) {
      this.source.bumpLookupGeneration();
    }
  }
}

export interface SpatiallyIndexedSkeletonChunkSpecification
  extends SliceViewChunkSpecification {
  chunkLayout: ChunkLayout;
}

type SpatiallyIndexedSkeletonChunkListener = (
  key: string,
  chunk: SpatiallyIndexedSkeletonChunk,
) => void;

export class SpatiallyIndexedSkeletonSource extends SliceViewChunkSource<
  SpatiallyIndexedSkeletonChunkSpecification,
  SpatiallyIndexedSkeletonChunk
> {
  vertexAttributes: VertexAttributeRenderInfo[];
  lookupGeneration = 0;
  private attributeTextureFormats_?: TextureFormat[];
  private chunkListeners = new Set<SpatiallyIndexedSkeletonChunkListener>();

  constructor(chunkManager: ChunkManager, options: any) {
    super(chunkManager, options);
    this.vertexAttributes = [vertexPositionAttribute, segmentAttribute];
  }

  get attributeTextureFormats() {
    let attributeTextureFormats = this.attributeTextureFormats_;
    if (attributeTextureFormats === undefined) {
      attributeTextureFormats = this.attributeTextureFormats_ =
        spatiallyIndexedSkeletonTextureAttributeSpecs.map(
          ({ dataType, numComponents }) =>
            computeTextureFormat(new TextureFormat(), dataType, numComponents),
        );
    }
    return attributeTextureFormats;
  }

  static encodeSpec(spec: SpatiallyIndexedSkeletonChunkSpecification) {
    const base = SliceViewChunkSource.encodeSpec(spec);
    return { ...base, chunkLayout: spec.chunkLayout.toObject() };
  }

  bumpLookupGeneration() {
    ++this.lookupGeneration;
  }

  addChunkListener(listener: SpatiallyIndexedSkeletonChunkListener) {
    this.chunkListeners.add(listener);
    return () => this.chunkListeners.delete(listener);
  }

  addChunk(key: string, chunk: SpatiallyIndexedSkeletonChunk) {
    super.addChunk(key, chunk);
    for (const listener of this.chunkListeners) {
      listener(key, chunk);
    }
  }

  getChunk(chunkData: SkeletonChunkData) {
    return new SpatiallyIndexedSkeletonChunk(this, chunkData);
  }
}

export abstract class MultiscaleSpatiallyIndexedSkeletonSource extends MultiscaleSliceViewChunkSource<SpatiallyIndexedSkeletonSource> {
  getPerspectiveSources(): SliceViewSingleResolutionSource<SpatiallyIndexedSkeletonSource>[] {
    const sources = this.getSources({ view: "3d" } as any);
    const flattened: SliceViewSingleResolutionSource<SpatiallyIndexedSkeletonSource>[] =
      [];
    for (const scale of sources) {
      if (scale.length > 0) {
        flattened.push(scale[0]);
      }
    }
    return flattened;
  }

  getSliceViewPanelSources(): SliceViewSingleResolutionSource<SpatiallyIndexedSkeletonSource>[] {
    return this.getPerspectiveSources();
  }

  getSpatialSkeletonGridSizes():
    | { x: number; y: number; z: number }[]
    | undefined {
    return undefined;
  }
}

export class MultiscaleSliceViewSpatiallyIndexedSkeletonLayer extends SliceViewRenderLayer<SpatiallyIndexedSkeletonSource> {
  private renderOptions: ViewSpecificSkeletonRenderingOptions;
  RPC_TYPE_ID = SPATIALLY_INDEXED_SKELETON_SLICEVIEW_RENDER_LAYER_RPC_ID;
  constructor(
    public chunkManager: ChunkManager,
    public multiscaleSource: MultiscaleSpatiallyIndexedSkeletonSource,
    public displayState: SegmentationDisplayState,
  ) {
    const renderScaleTarget = (displayState as any)
      .renderScaleTarget as WatchableValueInterface<number>;
    const gridLevel2d = (displayState as any).spatialSkeletonGridLevel2d as
      | WatchableValueInterface<number>
      | undefined;
    super(chunkManager, multiscaleSource, {
      transform: (displayState as any).transform,
      localPosition: (displayState as any).localPosition,
      renderScaleTarget,
      visibleSourcesInvalidation:
        gridLevel2d === undefined ? [] : [gridLevel2d],
    });
    this.renderOptions = (
      displayState as any
    ).skeletonRenderingOptions.params2d;
    this.registerDisposer(
      this.renderOptions.mode.changed.add(this.redrawNeeded.dispatch),
    );
    this.registerDisposer(
      this.renderOptions.lineWidth.changed.add(this.redrawNeeded.dispatch),
    );
    const rpc = this.chunkManager.rpc!;
    const lod2d = (displayState as any).spatialSkeletonLod2d;
    if (gridLevel2d !== undefined && lod2d !== undefined) {
      this.rpcTransfer = {
        ...this.rpcTransfer,
        chunkManager: this.chunkManager.rpcId,
        skeletonGridLevel: this.registerDisposer(
          SharedWatchableValue.makeFromExisting(rpc, gridLevel2d),
        ).rpcId,
        skeletonLod: this.registerDisposer(
          SharedWatchableValue.makeFromExisting(rpc, lod2d),
        ).rpcId,
      };
    }
    this.initializeCounterpart();
  }

  filterVisibleSources(
    sliceView: SliceViewBase,
    sources: readonly TransformedSource[],
  ): Iterable<TransformedSource> {
    const gridLevel = (this.displayState as any).spatialSkeletonGridLevel2d
      ?.value as number | undefined;
    if (
      gridLevel === undefined ||
      sources.length === 0 ||
      !sources.every(
        (source) => getSpatiallyIndexedSkeletonGridIndex(source) !== undefined,
      )
    ) {
      return super.filterVisibleSources(sliceView, sources);
    }
    return selectSpatiallyIndexedSkeletonEntriesByGrid(
      sources,
      gridLevel,
      getSpatiallyIndexedSkeletonGridIndex,
    );
  }

  // Draw is no-op as per SliceViewSpatiallyIndexedSkeletonLayer pattern
  draw(renderContext: SliceViewRenderContext) {
    renderContext;
  }
}

type SpatiallyIndexedSkeletonSourceEntry =
  SliceViewSingleResolutionSource<SpatiallyIndexedSkeletonSource>;

interface SpatiallyIndexedNodeLocatorEntry {
  chunk: SpatiallyIndexedSkeletonChunk;
  vertexIndex: number;
  sourceId: string;
}

interface SpatiallyIndexedParentReferenceEntry {
  chunk: SpatiallyIndexedSkeletonChunk;
  sourceId: string;
}

interface SpatiallyIndexedSkeletonLayerOptions {
  gridLevel?: WatchableValueInterface<number>;
  lod?: WatchableValueInterface<number>;
  sources2d?: SpatiallyIndexedSkeletonSourceEntry[];
  selectedNodeId?: WatchableValueInterface<number | undefined>;
  editMode?: WatchableValueInterface<boolean>;
  mergeMode?: WatchableValueInterface<boolean>;
  splitMode?: WatchableValueInterface<boolean>;
  pendingNodePositionVersion?: WatchableValueInterface<number>;
  getPendingNodePosition?: (nodeId: number) => ArrayLike<number> | undefined;
  getCachedNode?: (
    nodeId: number,
  ) => SpatiallyIndexedSkeletonNodeInfo | undefined;
}

export interface SpatiallyIndexedSkeletonNodeInfo {
  nodeId: number;
  segmentId: number;
  position: Float32Array;
  parentNodeId?: number;
  radius?: number;
  confidence?: number;
  labels?: readonly string[];
}

interface SpatiallyIndexedEffectiveNodeState {
  nodeId: number;
  segmentId: number;
  position: ArrayLike<number>;
  parentNodeId?: number;
}

interface SpatiallyIndexedResidentNodeState
  extends SpatiallyIndexedEffectiveNodeState {
  selected: boolean;
}

interface SpatiallyIndexedLocalAddNodeResult {
  node: SpatiallyIndexedSkeletonNodeInfo;
  localUpdateApplied: boolean;
}

interface SpatiallyIndexedLocalMoveNodeResult {
  localUpdateApplied: boolean;
  changed: boolean;
}

function getSpatialSkeletonGridSpacing(
  transformedSource: TransformedSource,
  levels: Array<{ size: { x: number; y: number; z: number } }> | undefined,
  gridIndex: number,
) {
  const levelSize = levels?.[gridIndex]?.size;
  if (levelSize !== undefined) {
    return Math.max(Math.min(levelSize.x, levelSize.y, levelSize.z), 1e-6);
  }
  const chunkSize = transformedSource.chunkLayout.size;
  return Math.max(Math.min(chunkSize[0], chunkSize[1], chunkSize[2]), 1e-6);
}

function updateSpatialSkeletonGridRenderScaleHistogram(
  histogram: RenderScaleHistogram,
  frameNumber: number,
  transformedSources: readonly TransformedSource[][],
  projectionParameters: any,
  localPosition: Float32Array,
  lod: number | undefined,
  levels: Array<{ size: { x: number; y: number; z: number } }> | undefined,
  relative: boolean,
  pixelSize: number,
) {
  histogram.begin(frameNumber);
  if (lod === undefined || transformedSources.length === 0) {
    return;
  }
  const lodSuffix = `:${lod}`;
  const scales = transformedSources[0] ?? [];
  if (scales.length === 0) {
    return;
  }
  const safePixelSize = Math.max(pixelSize, 1e-6);
  for (const tsource of scales) {
    const gridIndex = (tsource.source as any).parameters?.gridIndex as
      | number
      | undefined;
    if (gridIndex === undefined) {
      continue;
    }
    const source = tsource.source as unknown as {
      chunks: Map<string, SpatiallyIndexedSkeletonChunk>;
    };
    let presentCount = 0;
    let missingCount = 0;
    forEachVisibleVolumetricChunk(
      projectionParameters,
      localPosition,
      tsource,
      (positionInChunks) => {
        const key = `${positionInChunks.join()}${lodSuffix}`;
        const chunk = source.chunks.get(key) as
          | SpatiallyIndexedSkeletonChunk
          | undefined;
        if (chunk?.state === ChunkState.GPU_MEMORY) {
          presentCount++;
        } else {
          missingCount++;
        }
      },
    );
    const spacing = getSpatialSkeletonGridSpacing(tsource, levels, gridIndex);
    const renderScale = relative ? spacing / safePixelSize : spacing;
    const total = presentCount + missingCount;
    if (total > 0) {
      histogram.add(spacing, renderScale, presentCount, missingCount);
    } else {
      // Keep all grid rows visible in the histogram even when currently empty.
      histogram.add(spacing, renderScale, 0, 1, true);
    }
  }
}

type VisibleSpatialChunksBySource = Map<
  string,
  readonly SpatiallyIndexedSkeletonChunk[]
>;

type VisibleSpatialChunkSetsBySource = Map<
  string,
  ReadonlySet<SpatiallyIndexedSkeletonChunk>
>;

export class SpatiallyIndexedSkeletonLayer
  extends RefCounted
  implements SkeletonLayerInterface
{
  layerChunkProgressInfo = new LayerChunkProgressInfo();
  redrawNeeded = new NullarySignal();
  dynamicSegmentAppearance = true;
  vertexAttributes: VertexAttributeRenderInfo[];
  segmentColorAttributeIndex: number | undefined;
  selectedNodeAttributeIndex: number | undefined;
  readonly chunkGeometryRenderLayerInterface: SkeletonLayerInterface;
  fallbackShaderParameters = new WatchableValue(
    getFallbackBuilderState(parseShaderUiControls(DEFAULT_FRAGMENT_MAIN)),
  );
  backend: ChunkRenderLayerFrontend;
  localPosition: WatchableValueInterface<Float32Array>;
  rpc: RPC | undefined;

  private generation = 0;
  private filteredAttributeTextureFormats = [
    vertexPositionTextureFormat,
    segmentTextureFormat,
    selectedNodeTextureFormat,
  ];
  private regularSkeletonLayerWatchable = new WatchableValue(false);
  private regularSkeletonLayerUserLayer: UserLayer | undefined;
  private removeRegularSkeletonLayerUserLayerListener:
    | (() => boolean)
    | undefined;
  private nodeLocatorIndexKey: string | undefined;
  private nodeLocatorIndex = new Map<
    number,
    SpatiallyIndexedNodeLocatorEntry[]
  >();
  private baseParentByNodeId = new Map<number, number>();
  private parentReferenceIndex = new Map<
    number,
    SpatiallyIndexedParentReferenceEntry[]
  >();
  private visibleChunksByView = new Map<
    SpatiallyIndexedSkeletonView,
    VisibleSpatialChunksBySource
  >();
  private visibleChunkSetsByView = new Map<
    SpatiallyIndexedSkeletonView,
    VisibleSpatialChunkSetsBySource
  >();
  private visibleSourceIdsByView = new Map<
    SpatiallyIndexedSkeletonView,
    ReadonlySet<string>
  >();
  private visibleChunkLodByView = new Map<
    SpatiallyIndexedSkeletonView,
    number | undefined
  >();
  gridLevel: WatchableValueInterface<number>;
  lod: WatchableValueInterface<number>;
  private selectedNodeId:
    | WatchableValueInterface<number | undefined>
    | undefined;
  private editMode: WatchableValueInterface<boolean> | undefined;
  private mergeMode: WatchableValueInterface<boolean> | undefined;
  private splitMode: WatchableValueInterface<boolean> | undefined;
  private getPendingNodePositionOverride:
    | ((nodeId: number) => ArrayLike<number> | undefined)
    | undefined;
  private getCachedNodeInfo:
    | ((nodeId: number) => SpatiallyIndexedSkeletonNodeInfo | undefined)
    | undefined;

  private markFilteredDataDirty() {
    this.generation++;
    this.redrawNeeded.dispatch();
  }

  private requestFilteredDataRedraw() {
    this.redrawNeeded.dispatch();
  }

  private isDetailedInteractionActive() {
    return (
      this.selectedNodeId?.value !== undefined ||
      this.editMode?.value === true ||
      this.mergeMode?.value === true ||
      this.splitMode?.value === true
    );
  }

  private *iterateUniqueChunkSources() {
    const seenSourceIds = new Set<string>();
    for (const sourceEntry of [...this.sources, ...this.sources2d]) {
      const sourceId = getObjectId(sourceEntry.chunkSource);
      if (seenSourceIds.has(sourceId)) continue;
      seenSourceIds.add(sourceId);
      yield sourceEntry.chunkSource;
    }
  }

  private clearDetailedInteractionCaches() {
    this.invalidateNodeLocatorIndex();
    for (const chunkSource of this.iterateUniqueChunkSources()) {
      for (const chunk of chunkSource.chunks.values()) {
        this.clearFilteredChunkData(chunk as SpatiallyIndexedSkeletonChunk);
      }
    }
  }

  private handleDetailedInteractionStateChanged() {
    if (this.isDetailedInteractionActive()) {
      this.markFilteredDataDirty();
      return;
    }
    this.clearDetailedInteractionCaches();
    this.redrawNeeded.dispatch();
  }

  private computeHasRegularSkeletonLayer(userLayer: UserLayer) {
    for (const renderLayer of userLayer.renderLayers) {
      if (
        renderLayer instanceof PerspectiveViewSkeletonLayer ||
        renderLayer instanceof SliceViewPanelSkeletonLayer
      ) {
        return true;
      }
    }
    return false;
  }

  private updateHasRegularSkeletonLayerWatchable(
    userLayer: UserLayer | undefined,
  ) {
    if (this.regularSkeletonLayerUserLayer !== userLayer) {
      this.removeRegularSkeletonLayerUserLayerListener?.();
      this.removeRegularSkeletonLayerUserLayerListener = undefined;
      this.regularSkeletonLayerUserLayer = userLayer;
      if (userLayer !== undefined) {
        const update = () => {
          const nextValue = this.computeHasRegularSkeletonLayer(userLayer);
          if (this.regularSkeletonLayerWatchable.value !== nextValue) {
            this.regularSkeletonLayerWatchable.value = nextValue;
            this.redrawNeeded.dispatch();
          }
        };
        update();
        this.removeRegularSkeletonLayerUserLayerListener =
          userLayer.layersChanged.add(update);
      } else if (this.regularSkeletonLayerWatchable.value) {
        this.regularSkeletonLayerWatchable.value = false;
        this.redrawNeeded.dispatch();
      }
    }
    return this.regularSkeletonLayerWatchable.value;
  }

  private lodMatches(
    chunk: SpatiallyIndexedSkeletonChunk,
    targetLod: number | undefined,
  ) {
    if (targetLod === undefined || chunk.lod === undefined) {
      return true;
    }
    return Math.abs(chunk.lod - targetLod) < 1e-6;
  }

  private getIndexedSources() {
    return dedupeSpatiallyIndexedSkeletonEntries(
      [...this.sources, ...this.sources2d],
      (entry) => getObjectId(entry.chunkSource),
    );
  }

  private makeNodeLocatorIndexKey(
    sourceEntries: readonly SpatiallyIndexedSkeletonSourceEntry[],
  ) {
    return sourceEntries
      .map(
        (entry) =>
          `${getObjectId(entry.chunkSource)}:${entry.chunkSource.lookupGeneration}`,
      )
      .join(",");
  }

  private invalidateNodeLocatorIndex() {
    this.nodeLocatorIndex.clear();
    this.baseParentByNodeId.clear();
    this.parentReferenceIndex.clear();
    this.nodeLocatorIndexKey = undefined;
  }

  private ensureNodeLocatorIndex() {
    const sourceEntries = this.getIndexedSources();
    const key = this.makeNodeLocatorIndexKey(sourceEntries);
    if (this.nodeLocatorIndexKey === key) return;
    this.nodeLocatorIndexKey = key;
    this.nodeLocatorIndex.clear();
    this.baseParentByNodeId.clear();
    this.parentReferenceIndex.clear();
    for (const sourceEntry of sourceEntries) {
      const sourceId = getObjectId(sourceEntry.chunkSource);
      for (const chunk of sourceEntry.chunkSource.chunks.values()) {
        const typedChunk = chunk as SpatiallyIndexedSkeletonChunk;
        if (typedChunk.state !== ChunkState.GPU_MEMORY) continue;
        for (const [nodeId, vertexIndex] of typedChunk.nodeMap.entries()) {
          let entries = this.nodeLocatorIndex.get(nodeId);
          if (entries === undefined) {
            entries = [];
            this.nodeLocatorIndex.set(nodeId, entries);
          }
          entries.push({
            chunk: typedChunk,
            vertexIndex,
            sourceId,
          });
        }
        const vertexToNodeId = new Map<number, number>();
        for (const [nodeId, vertexIndex] of typedChunk.nodeMap.entries()) {
          if (vertexIndex < 0 || vertexIndex >= typedChunk.numVertices)
            continue;
          vertexToNodeId.set(vertexIndex, nodeId);
        }
        const indices = typedChunk.indices;
        for (let i = 0; i < typedChunk.numIndices; i += 2) {
          const childNodeId = vertexToNodeId.get(indices[i]);
          const parentNodeId = vertexToNodeId.get(indices[i + 1]);
          if (
            childNodeId === undefined ||
            parentNodeId === undefined ||
            childNodeId === parentNodeId ||
            this.baseParentByNodeId.has(childNodeId)
          ) {
            continue;
          }
          this.baseParentByNodeId.set(childNodeId, parentNodeId);
        }
        if (typedChunk.missingConnections.length === 0) continue;
        const seenParentIds = new Set<number>();
        for (const connection of typedChunk.missingConnections) {
          if (
            !Number.isFinite(connection.nodeId) ||
            !Number.isFinite(connection.parentId) ||
            connection.nodeId === connection.parentId
          ) {
            continue;
          }
          if (!this.baseParentByNodeId.has(connection.nodeId)) {
            this.baseParentByNodeId.set(connection.nodeId, connection.parentId);
          }
          if (seenParentIds.has(connection.parentId)) {
            continue;
          }
          seenParentIds.add(connection.parentId);
          let entries = this.parentReferenceIndex.get(connection.parentId);
          if (entries === undefined) {
            entries = [];
            this.parentReferenceIndex.set(connection.parentId, entries);
          }
          entries.push({
            chunk: typedChunk,
            sourceId,
          });
        }
      }
    }
  }

  private makeSourceIdSet(
    sourceEntries: readonly SpatiallyIndexedSkeletonSourceEntry[],
  ) {
    const sourceIds = new Set<string>();
    for (const sourceEntry of sourceEntries) {
      sourceIds.add(getObjectId(sourceEntry.chunkSource));
    }
    return sourceIds;
  }

  private getNodeLocatorEntries(
    nodeId: number,
    selectedSourceIds: ReadonlySet<string>,
    targetLod: number | undefined,
    options: {
      requireVisible?: boolean;
    } = {},
  ) {
    this.ensureNodeLocatorIndex();
    const entries = this.nodeLocatorIndex.get(nodeId);
    if (entries === undefined) return [];
    return entries.filter(
      (entry) =>
        selectedSourceIds.has(entry.sourceId) &&
        entry.chunk.state === ChunkState.GPU_MEMORY &&
        this.lodMatches(entry.chunk, targetLod) &&
        (!(options.requireVisible ?? false) ||
          this.isChunkVisibleForEditing(entry.chunk, targetLod)),
    );
  }

  private getPrimaryNodeLocator(
    nodeId: number,
    selectedSourceIds: ReadonlySet<string>,
    targetLod: number | undefined,
    options: {
      requireVisible?: boolean;
    } = {},
  ) {
    this.ensureNodeLocatorIndex();
    const entries = this.nodeLocatorIndex.get(nodeId);
    if (entries === undefined) return undefined;
    for (const entry of entries) {
      if (!selectedSourceIds.has(entry.sourceId)) continue;
      if (entry.chunk.state !== ChunkState.GPU_MEMORY) continue;
      if (!this.lodMatches(entry.chunk, targetLod)) continue;
      if (
        (options.requireVisible ?? false) &&
        !this.isChunkVisibleForEditing(entry.chunk, targetLod)
      ) {
        continue;
      }
      return entry;
    }
    return undefined;
  }

  private getMaxIndexedNodeId() {
    this.ensureNodeLocatorIndex();
    let maxNodeId = 0;
    for (const nodeId of this.nodeLocatorIndex.keys()) {
      if (nodeId > maxNodeId) maxNodeId = nodeId;
    }
    return maxNodeId;
  }

  private hasIndexedNodeId(nodeId: number) {
    this.ensureNodeLocatorIndex();
    return this.nodeLocatorIndex.has(nodeId);
  }

  get visibility() {
    return this.displayState.objectAlpha;
  }

  sources: SpatiallyIndexedSkeletonSourceEntry[];
  sources2d: SpatiallyIndexedSkeletonSourceEntry[];
  source: SpatiallyIndexedSkeletonSource;

  constructor(
    public chunkManager: ChunkManager,
    sources:
      | SpatiallyIndexedSkeletonSourceEntry[]
      | SpatiallyIndexedSkeletonSource,
    public displayState: SkeletonLayerDisplayState & {
      localPosition: WatchableValueInterface<Float32Array>;
    },
    options: SpatiallyIndexedSkeletonLayerOptions = {},
  ) {
    super();
    this.registerDisposer(() => {
      this.removeRegularSkeletonLayerUserLayerListener?.();
      this.removeRegularSkeletonLayerUserLayerListener = undefined;
      this.regularSkeletonLayerUserLayer = undefined;
      this.clearDetailedInteractionCaches();
    });
    let sources3d: SpatiallyIndexedSkeletonSourceEntry[];
    let sources2d = options.sources2d ?? [];
    if (Array.isArray(sources)) {
      sources3d = sources;
    } else {
      sources3d = [
        {
          chunkSource: sources,
          chunkToMultiscaleTransform: mat4.create(),
        },
      ];
    }
    if (sources3d.length === 0 && sources2d.length > 0) {
      sources3d = sources2d;
    }
    if (sources2d.length === 0) {
      sources2d = sources3d;
    }
    if (sources3d.length === 0) {
      throw new Error(
        "SpatiallyIndexedSkeletonLayer requires at least one source.",
      );
    }
    this.sources = sources3d;
    this.sources2d = sources2d;
    this.source = sources3d[0].chunkSource;
    this.localPosition = displayState.localPosition;
    this.gridLevel =
      options.gridLevel ??
      (displayState as any).spatialSkeletonGridLevel3d ??
      new WatchableValue(0);
    this.lod =
      options.lod ?? (displayState as any).skeletonLod ?? new WatchableValue(0);
    this.selectedNodeId = options.selectedNodeId;
    this.editMode = options.editMode;
    this.mergeMode = options.mergeMode;
    this.splitMode = options.splitMode;
    this.getPendingNodePositionOverride = options.getPendingNodePosition;
    this.getCachedNodeInfo = options.getCachedNode;
    registerRedrawWhenSegmentationDisplayState3DChanged(displayState, this);
    this.displayState.shaderError.value = undefined;
    const { skeletonRenderingOptions: renderingOptions } = displayState;
    this.registerDisposer(
      renderingOptions.shader.changed.add(() => {
        this.displayState.shaderError.value = undefined;
        this.redrawNeeded.dispatch();
      }),
    );

    this.vertexAttributes = [
      ...this.source.vertexAttributes,
      selectedNodeAttribute,
    ];
    this.chunkGeometryRenderLayerInterface = {
      vertexAttributes: this.source.vertexAttributes,
      segmentColorAttributeIndex: this.segmentColorAttributeIndex,
      dynamicSegmentAppearance: this.dynamicSegmentAppearance,
      gl: this.gl,
      fallbackShaderParameters: this.fallbackShaderParameters,
      displayState: this.displayState,
    };
    this.segmentColorAttributeIndex = undefined;
    const selectedNodeIndex = this.vertexAttributes.findIndex(
      (x) => x.name === selectedNodeAttribute.name,
    );
    this.selectedNodeAttributeIndex =
      selectedNodeIndex >= 0 ? selectedNodeIndex : undefined;
    const markDirty = () => this.markFilteredDataDirty();
    const requestRedraw = () => this.redrawNeeded.dispatch();
    const dirtyWatchables = new Set<object>();
    const registerNumericDirtyWatchable = (
      watchable: WatchableValueInterface<number> | undefined,
    ) => {
      if (watchable === undefined) return;
      const key = watchable as object;
      if (dirtyWatchables.has(key)) return;
      dirtyWatchables.add(key);
      this.registerDisposer(watchable.changed.add(() => markDirty()));
    };
    // Monitor visible segment changes to update filtered buffers.
    this.registerDisposer(
      registerNested((context, segmentationGroup) => {
        context.registerDisposer(
          segmentationGroup.visibleSegments.changed.add(() => requestRedraw()),
        );
        context.registerDisposer(
          segmentationGroup.temporaryVisibleSegments.changed.add(() =>
            requestRedraw(),
          ),
        );
        context.registerDisposer(
          segmentationGroup.useTemporaryVisibleSegments.changed.add(() =>
            requestRedraw(),
          ),
        );
      }, this.displayState.segmentationGroupState),
    );
    // Monitor segment color changes to update filtered buffers.
    this.registerDisposer(
      registerNested((context, colorGroupState) => {
        context.registerDisposer(
          colorGroupState.segmentColorHash.changed.add(() => requestRedraw()),
        );
        context.registerDisposer(
          colorGroupState.segmentDefaultColor.changed.add(() =>
            requestRedraw(),
          ),
        );
        context.registerDisposer(
          colorGroupState.segmentStatedColors.changed.add(() =>
            requestRedraw(),
          ),
        );
      }, this.displayState.segmentationColorGroupState),
    );
    this.registerDisposer(
      displayState.objectAlpha.changed.add(() => requestRedraw()),
    );
    const selectedNodeWatchable = this.selectedNodeId;
    if (selectedNodeWatchable?.changed) {
      this.registerDisposer(
        selectedNodeWatchable.changed.add(() =>
          this.handleDetailedInteractionStateChanged(),
        ),
      );
    }
    const editModeWatchable = this.editMode;
    if (editModeWatchable?.changed) {
      this.registerDisposer(
        editModeWatchable.changed.add(() =>
          this.handleDetailedInteractionStateChanged(),
        ),
      );
    }
    const mergeModeWatchable = this.mergeMode;
    if (mergeModeWatchable?.changed) {
      this.registerDisposer(
        mergeModeWatchable.changed.add(() =>
          this.handleDetailedInteractionStateChanged(),
        ),
      );
    }
    const splitModeWatchable = this.splitMode;
    if (splitModeWatchable?.changed) {
      this.registerDisposer(
        splitModeWatchable.changed.add(() =>
          this.handleDetailedInteractionStateChanged(),
        ),
      );
    }
    const pendingNodePositionVersion = options.pendingNodePositionVersion;
    if (pendingNodePositionVersion?.changed) {
      this.registerDisposer(
        pendingNodePositionVersion.changed.add(() => {
          this.redrawNeeded.dispatch();
        }),
      );
    }
    registerNumericDirtyWatchable(this.gridLevel);
    registerNumericDirtyWatchable(
      (displayState as any).spatialSkeletonGridLevel2d,
    );
    registerNumericDirtyWatchable(
      (displayState as any).spatialSkeletonGridLevel3d,
    );
    registerNumericDirtyWatchable(this.lod);
    registerNumericDirtyWatchable((displayState as any).spatialSkeletonLod2d);
    registerNumericDirtyWatchable((displayState as any).skeletonLod);
    if (displayState.hiddenObjectAlpha) {
      this.registerDisposer(
        displayState.hiddenObjectAlpha.changed.add(() => requestRedraw()),
      );
    }

    // Create backend for perspective view chunk management
    const sharedObject = this.registerDisposer(
      new ChunkRenderLayerFrontend(this.layerChunkProgressInfo),
    );
    const rpc = chunkManager.rpc!;
    this.rpc = rpc;
    sharedObject.RPC_TYPE_ID = SPATIALLY_INDEXED_SKELETON_RENDER_LAYER_RPC_ID;

    const renderScaleTargetWatchable = this.registerDisposer(
      SharedWatchableValue.makeFromExisting(
        rpc,
        displayState.renderScaleTarget,
      ),
    );

    const skeletonLodWatchable = this.registerDisposer(
      SharedWatchableValue.makeFromExisting(rpc, this.lod),
    );

    const skeletonGridLevelWatchable = this.registerDisposer(
      SharedWatchableValue.makeFromExisting(rpc, this.gridLevel),
    );

    sharedObject.initializeCounterpart(rpc, {
      chunkManager: chunkManager.rpcId,
      localPosition: this.registerDisposer(
        SharedWatchableValue.makeFromExisting(rpc, this.localPosition),
      ).rpcId,
      renderScaleTarget: renderScaleTargetWatchable.rpcId,
      skeletonLod: skeletonLodWatchable.rpcId,
      skeletonGridLevel: skeletonGridLevelWatchable.rpcId,
    });
    this.backend = sharedObject;
  }

  get gl() {
    return this.chunkManager.chunkQueueManager.gl;
  }

  getSources(view: SpatiallyIndexedSkeletonView) {
    return view === "2d" ? this.sources2d : this.sources;
  }

  getSourceCapabilities(): SpatiallyIndexedSkeletonSourceCapabilities {
    return getSpatiallyIndexedSkeletonSourceCapabilities(this.source);
  }

  private selectSourcesForViewAndGrid(
    view: SpatiallyIndexedSkeletonView,
    gridLevel: number | undefined,
  ) {
    return selectSpatiallyIndexedSkeletonEntriesForView(
      this.getSources(view),
      view,
      gridLevel,
      getSpatiallyIndexedSkeletonSourceView,
      getSpatiallyIndexedSkeletonGridIndex,
    );
  }

  private getCurrentSourcesForEditing() {
    const gridLevel3d = (this.displayState as any).spatialSkeletonGridLevel3d
      ?.value as number | undefined;
    const gridLevel2d = (this.displayState as any).spatialSkeletonGridLevel2d
      ?.value as number | undefined;
    const selected3d = this.selectSourcesForViewAndGrid(
      "3d",
      gridLevel3d ?? this.gridLevel.value,
    );
    const selected2d = this.selectSourcesForViewAndGrid(
      "2d",
      gridLevel2d ?? this.gridLevel.value,
    );
    return dedupeSpatiallyIndexedSkeletonEntries(
      [...selected3d, ...selected2d],
      (sourceEntry) => getObjectId(sourceEntry.chunkSource),
    );
  }

  private getCurrentVisibleSourcesForEditing(targetLod: number | undefined) {
    const selectedSources = this.getCurrentSourcesForEditing();
    const visibleSourceIds = new Set<string>();
    for (const view of ["3d", "2d"] as const) {
      if (!this.visibleChunksMatchTargetLod(view, targetLod)) {
        continue;
      }
      const viewVisibleSourceIds = this.visibleSourceIdsByView.get(view);
      if (viewVisibleSourceIds === undefined) continue;
      for (const sourceId of viewVisibleSourceIds) {
        visibleSourceIds.add(sourceId);
      }
    }
    return selectedSources.filter((sourceEntry) =>
      visibleSourceIds.has(getObjectId(sourceEntry.chunkSource)),
    );
  }

  private visibleChunksMatchTargetLod(
    view: SpatiallyIndexedSkeletonView,
    targetLod: number | undefined,
  ) {
    const visibleLod = this.visibleChunkLodByView.get(view);
    if (visibleLod === undefined || targetLod === undefined) {
      return visibleLod === targetLod;
    }
    return Math.abs(visibleLod - targetLod) < 1e-6;
  }

  private isChunkVisibleForEditing(
    chunk: SpatiallyIndexedSkeletonChunk,
    targetLod: number | undefined,
  ) {
    if (chunk.state !== ChunkState.GPU_MEMORY) {
      return false;
    }
    if (!this.lodMatches(chunk, targetLod)) {
      return false;
    }
    const sourceId = getObjectId(chunk.source);
    for (const view of ["3d", "2d"] as const) {
      if (!this.visibleChunksMatchTargetLod(view, targetLod)) {
        continue;
      }
      const visibleChunkSet = this.visibleChunkSetsByView
        .get(view)
        ?.get(sourceId);
      if (visibleChunkSet?.has(chunk)) {
        return true;
      }
    }
    return false;
  }

  private resolveTargetLod(targetLod: number | undefined) {
    return targetLod ?? this.lod.value;
  }

  private getChunkKey(
    chunk: SpatiallyIndexedSkeletonChunk,
    targetLod: number | undefined,
  ) {
    const lodValue = chunk.lod ?? targetLod;
    const baseKey = chunk.chunkGridPosition.join();
    return lodValue === undefined ? baseKey : `${baseKey}:${lodValue}`;
  }

  private findLoadedChunkForPosition(
    sourceEntry: SpatiallyIndexedSkeletonSourceEntry,
    position: ArrayLike<number>,
    lod: number | undefined,
  ) {
    if (lod === undefined) {
      return undefined;
    }
    const { chunkDataSize } = sourceEntry.chunkSource.spec;
    const rank = chunkDataSize.length;
    const chunkTransform = new Float32Array((rank + 1) ** 2);
    matrix.inverse(
      chunkTransform,
      rank + 1,
      sourceEntry.chunkToMultiscaleTransform,
      rank + 1,
      rank + 1,
    );
    for (let i = 0; i < rank; ++i) {
      for (let j = 0; j < rank + 1; ++j) {
        chunkTransform[(rank + 1) * j + i] /= chunkDataSize[i];
      }
    }
    const input = new Float32Array(rank);
    for (let i = 0; i < rank; ++i) {
      input[i] = Number(position[i] ?? 0);
    }
    const chunkPoint = new Float32Array(rank);
    matrix.transformPoint(chunkPoint, chunkTransform, rank + 1, input, rank);
    for (let i = 0; i < rank; ++i) {
      chunkPoint[i] = Math.floor(chunkPoint[i]);
    }
    const chunkKey = `${chunkPoint.join()}:${lod}`;
    const chunk = sourceEntry.chunkSource.chunks.get(chunkKey) as
      | SpatiallyIndexedSkeletonChunk
      | undefined;
    if (chunk?.state !== ChunkState.GPU_MEMORY) {
      return undefined;
    }
    return chunk;
  }

  private positionsEqual(
    a: ArrayLike<number>,
    b: ArrayLike<number>,
    epsilon = 1e-6,
  ) {
    return (
      Math.abs(Number(a[0]) - Number(b[0])) <= epsilon &&
      Math.abs(Number(a[1]) - Number(b[1])) <= epsilon &&
      Math.abs(Number(a[2]) - Number(b[2])) <= epsilon
    );
  }

  private getEffectiveParentNodeId(nodeId: number) {
    return this.baseParentByNodeId.get(nodeId);
  }

  private getBaseNodeState(
    nodeId: number,
    selectedSourceIds: ReadonlySet<string>,
    targetLod: number | undefined,
    options: {
      includePendingPosition?: boolean;
      requireVisible?: boolean;
    } = {},
  ): SpatiallyIndexedEffectiveNodeState | undefined {
    const entry = this.getPrimaryNodeLocator(
      nodeId,
      selectedSourceIds,
      targetLod,
      { requireVisible: options.requireVisible },
    );
    if (entry === undefined) {
      return undefined;
    }
    const data = this.getChunkPositionAndSegmentArrays(entry.chunk);
    if (data === undefined) {
      return undefined;
    }
    const { positions, segmentIds } = data;
    return {
      nodeId,
      segmentId: segmentIds[entry.vertexIndex],
      position:
        (options.includePendingPosition ?? true)
          ? this.getEffectiveNodePosition(positions, entry.vertexIndex, nodeId)
          : positions.subarray(
              entry.vertexIndex * 3,
              entry.vertexIndex * 3 + 3,
            ),
      parentNodeId: this.baseParentByNodeId.get(nodeId),
    };
  }

  private getEffectiveNodeState(
    nodeId: number,
    selectedSourceIds: ReadonlySet<string>,
    targetLod: number | undefined,
  ): SpatiallyIndexedEffectiveNodeState | undefined {
    return this.getBaseNodeState(nodeId, selectedSourceIds, targetLod);
  }

  private getCachedNodeSnapshot(nodeId: number) {
    const cachedNode = this.getCachedNodeInfo?.(nodeId);
    if (cachedNode === undefined) {
      return undefined;
    }
    const pendingPosition =
      this.getPendingNodePositionOverride?.(cachedNode.nodeId) ??
      cachedNode.position;
    return {
      ...cachedNode,
      position: new Float32Array([
        Number(pendingPosition[0]),
        Number(pendingPosition[1]),
        Number(pendingPosition[2]),
      ]),
    };
  }

  private getEditableChunkData(
    chunk: SpatiallyIndexedSkeletonChunk,
  ): SpatiallyIndexedEditableChunkData | undefined {
    const data = this.getChunkPositionAndSegmentArrays(chunk);
    if (data === undefined) {
      return undefined;
    }
    return {
      nodeMap: new Map(chunk.nodeMap),
      positions: new Float32Array(data.positions),
      segmentIds: new Uint32Array(data.segmentIds),
      indices: new Uint32Array(chunk.indices),
      missingConnections: chunk.missingConnections.map((connection) => ({
        ...connection,
      })),
    };
  }

  private getChunkReadonlyData(
    chunk: SpatiallyIndexedSkeletonChunk,
  ): SpatiallyIndexedEditableChunkData | undefined {
    const data = this.getChunkPositionAndSegmentArrays(chunk);
    if (data === undefined) {
      return undefined;
    }
    return {
      nodeMap: chunk.nodeMap,
      positions: data.positions,
      segmentIds: data.segmentIds,
      indices: chunk.indices,
      missingConnections: chunk.missingConnections,
    };
  }

  private setChunkEditableData(
    chunk: SpatiallyIndexedSkeletonChunk,
    data: SpatiallyIndexedEditableChunkData,
  ) {
    const positionBytes = new Uint8Array(
      data.positions.buffer,
      data.positions.byteOffset,
      data.positions.byteLength,
    );
    const segmentBytes = new Uint8Array(
      data.segmentIds.buffer,
      data.segmentIds.byteOffset,
      data.segmentIds.byteLength,
    );
    const vertexBytes = new Uint8Array(
      positionBytes.byteLength + segmentBytes.byteLength,
    );
    vertexBytes.set(positionBytes, 0);
    vertexBytes.set(segmentBytes, positionBytes.byteLength);
    const vertexOffsets = new Uint32Array([0, positionBytes.byteLength]);
    for (const texture of chunk.vertexAttributeTextures) {
      if (texture) {
        this.gl.deleteTexture(texture);
      }
    }
    chunk.vertexAttributes = vertexBytes;
    chunk.vertexAttributeOffsets = vertexOffsets;
    chunk.numVertices = data.segmentIds.length;
    chunk.vertexAttributeTextures = uploadVertexAttributesToGPU(
      this.gl,
      chunk.vertexAttributes,
      chunk.vertexAttributeOffsets,
      chunk.source.attributeTextureFormats,
    );
    chunk.nodeMap = data.nodeMap;
    chunk.indices = data.indices;
    chunk.numIndices = data.indices.length;
    chunk.indexBuffer.setData(chunk.indices);
    chunk.missingConnections = data.missingConnections;
    chunk.source.bumpLookupGeneration();
  }

  private makeEditableChunkEntry(
    chunk: SpatiallyIndexedSkeletonChunk,
    targetLod: number | undefined,
    editableData?: SpatiallyIndexedEditableChunkData,
  ): SpatiallyIndexedEditableChunkEntry | undefined {
    const data = editableData ?? this.getChunkReadonlyData(chunk);
    if (data === undefined) {
      return undefined;
    }
    const chunkKey = this.getChunkKey(chunk, targetLod);
    return {
      chunkId: `${getObjectId(chunk.source)}:${chunkKey}`,
      chunkKey,
      sourceId: getObjectId(chunk.source),
      data,
    };
  }

  private rebuildConnectionsForChunks(
    chunksToRewrite: Iterable<SpatiallyIndexedSkeletonChunk>,
    targetLod: number | undefined,
    parentByNodeId: ReadonlyMap<number, number | undefined>,
    editableDataByChunk?: ReadonlyMap<
      SpatiallyIndexedSkeletonChunk,
      SpatiallyIndexedEditableChunkData
    >,
  ) {
    const targetPairs: Array<{
      chunk: SpatiallyIndexedSkeletonChunk;
      entry: SpatiallyIndexedEditableChunkEntry;
    }> = [];
    for (const chunk of chunksToRewrite) {
      const entry = this.makeEditableChunkEntry(
        chunk,
        targetLod,
        editableDataByChunk?.get(chunk),
      );
      if (entry !== undefined) {
        targetPairs.push({ chunk, entry });
      }
    }
    if (targetPairs.length === 0) {
      return new Set<SpatiallyIndexedSkeletonChunk>();
    }
    this.ensureNodeLocatorIndex();
    const overriddenChunks = new Set<SpatiallyIndexedSkeletonChunk>();
    const locatorsBySource = new Map<
      string,
      Map<number, SpatiallyIndexedChunkNodeLocator>
    >();
    for (const [chunk, editableData] of editableDataByChunk ?? []) {
      overriddenChunks.add(chunk);
      const sourceId = getObjectId(chunk.source);
      let sourceLocators = locatorsBySource.get(sourceId);
      if (sourceLocators === undefined) {
        sourceLocators = new Map<number, SpatiallyIndexedChunkNodeLocator>();
        locatorsBySource.set(sourceId, sourceLocators);
      }
      const chunkKey = this.getChunkKey(chunk, targetLod);
      for (const [nodeId, vertexIndex] of editableData.nodeMap.entries()) {
        if (vertexIndex < 0 || vertexIndex >= editableData.segmentIds.length) {
          continue;
        }
        sourceLocators.set(nodeId, { chunkKey, vertexIndex });
      }
    }
    const resolvedLocators = new Map<
      string,
      SpatiallyIndexedChunkNodeLocator | null
    >();
    const rebuiltByChunkId = new Map(
      rebuildTargetSpatialChunkConnections(
        targetPairs.map(({ entry }) => entry),
        parentByNodeId,
        (sourceId, nodeId) => {
          const overriddenLocator = locatorsBySource.get(sourceId)?.get(nodeId);
          if (overriddenLocator !== undefined) {
            return overriddenLocator;
          }
          const cacheKey = `${sourceId}:${nodeId}`;
          if (resolvedLocators.has(cacheKey)) {
            return resolvedLocators.get(cacheKey) ?? undefined;
          }
          let resolved: SpatiallyIndexedChunkNodeLocator | null = null;
          const entries = this.nodeLocatorIndex.get(nodeId);
          if (entries !== undefined) {
            for (const entry of entries) {
              if (entry.sourceId !== sourceId) continue;
              if (overriddenChunks.has(entry.chunk)) continue;
              if (entry.chunk.state !== ChunkState.GPU_MEMORY) continue;
              if (!this.lodMatches(entry.chunk, targetLod)) continue;
              if (!this.isChunkVisibleForEditing(entry.chunk, targetLod)) {
                continue;
              }
              resolved = {
                chunkKey: this.getChunkKey(entry.chunk, targetLod),
                vertexIndex: entry.vertexIndex,
              };
              break;
            }
          }
          resolvedLocators.set(cacheKey, resolved);
          return resolved ?? undefined;
        },
      ).map((rebuilt) => [rebuilt.chunkId, rebuilt]),
    );
    const rewrittenChunks = new Set<SpatiallyIndexedSkeletonChunk>();
    for (const { chunk, entry } of targetPairs) {
      const rebuilt = rebuiltByChunkId.get(entry.chunkId);
      this.setChunkEditableData(chunk, {
        ...entry.data,
        indices: rebuilt?.indices ?? entry.data.indices,
        missingConnections:
          rebuilt?.missingConnections ?? entry.data.missingConnections,
      });
      rewrittenChunks.add(chunk);
    }
    return rewrittenChunks;
  }

  private *iterateVisibleCandidateChunks(
    selectedSources: readonly SpatiallyIndexedSkeletonSourceEntry[],
    targetLod: number | undefined,
  ): Iterable<SpatiallyIndexedSkeletonChunk> {
    const selectedSourceIds = this.makeSourceIdSet(selectedSources);
    const yieldedChunkIds = new Set<string>();
    for (const visibleChunksBySource of this.visibleChunksByView.values()) {
      for (const [sourceId, visibleChunks] of visibleChunksBySource) {
        if (!selectedSourceIds.has(sourceId)) continue;
        for (const chunk of visibleChunks) {
          if (!this.lodMatches(chunk, targetLod)) continue;
          if (chunk.state !== ChunkState.GPU_MEMORY) continue;
          const chunkId = getObjectId(chunk);
          if (yieldedChunkIds.has(chunkId)) continue;
          yieldedChunkIds.add(chunkId);
          yield chunk;
        }
      }
    }
  }

  private finalizeEditedChunks(
    changedChunks: Iterable<SpatiallyIndexedSkeletonChunk>,
  ) {
    const seenChunks = new Set<SpatiallyIndexedSkeletonChunk>();
    for (const chunk of changedChunks) {
      if (seenChunks.has(chunk)) {
        continue;
      }
      seenChunks.add(chunk);
      this.clearFilteredChunkData(chunk);
    }
    this.invalidateNodeLocatorIndex();
    this.requestFilteredDataRedraw();
  }

  private getVisibleChunksForView(view: SpatiallyIndexedSkeletonView) {
    return this.visibleChunksByView.get(view);
  }

  private getPendingNodePosition(nodeId: number) {
    return this.getPendingNodePositionOverride?.(nodeId);
  }

  private getEffectiveNodePosition(
    positions: Float32Array,
    vertexIndex: number,
    nodeId: number | undefined,
  ): ArrayLike<number> {
    if (nodeId !== undefined) {
      const pendingPosition = this.getPendingNodePosition(nodeId);
      if (pendingPosition !== undefined) {
        return pendingPosition;
      }
    }
    return positions.subarray(vertexIndex * 3, vertexIndex * 3 + 3);
  }

  private getCommittedNodePosition(
    nodeId: number,
    selectedSources: SpatiallyIndexedSkeletonSourceEntry[],
    targetLod: number | undefined,
  ) {
    const selectedSourceIds = this.makeSourceIdSet(selectedSources);
    const baseNodeState = this.getBaseNodeState(
      nodeId,
      selectedSourceIds,
      targetLod,
      { includePendingPosition: false, requireVisible: true },
    );
    if (baseNodeState !== undefined) {
      return new Float32Array([
        Number(baseNodeState.position[0]),
        Number(baseNodeState.position[1]),
        Number(baseNodeState.position[2]),
      ]);
    }
    const entry = this.getPrimaryNodeLocator(
      nodeId,
      selectedSourceIds,
      targetLod,
      { requireVisible: true },
    );
    if (entry === undefined) {
      return undefined;
    }
    const data = this.getChunkPositionAndSegmentArrays(entry.chunk);
    if (data === undefined) {
      return undefined;
    }
    const { positions } = data;
    const index = entry.vertexIndex * 3;
    return new Float32Array([
      positions[index],
      positions[index + 1],
      positions[index + 2],
    ]);
  }

  private *iterateCandidateChunks(
    selectedSources: readonly SpatiallyIndexedSkeletonSourceEntry[],
    targetLod: number | undefined,
    options: {
      view?: SpatiallyIndexedSkeletonView;
    } = {},
  ): Iterable<SpatiallyIndexedSkeletonChunk> {
    const visibleChunksBySource =
      options.view === undefined
        ? undefined
        : this.getVisibleChunksForView(options.view);
    const useVisibleChunks = visibleChunksBySource !== undefined;
    for (const sourceEntry of selectedSources) {
      if (useVisibleChunks) {
        const visibleChunks = visibleChunksBySource.get(
          getObjectId(sourceEntry.chunkSource),
        );
        if (visibleChunks === undefined) {
          continue;
        }
        for (const chunk of visibleChunks) {
          if (!this.lodMatches(chunk, targetLod)) continue;
          if (chunk.state !== ChunkState.GPU_MEMORY) continue;
          yield chunk;
        }
        continue;
      }
      for (const chunk of sourceEntry.chunkSource.chunks.values()) {
        const typedChunk = chunk as SpatiallyIndexedSkeletonChunk;
        if (!this.lodMatches(typedChunk, targetLod)) continue;
        if (typedChunk.state !== ChunkState.GPU_MEMORY) continue;
        yield typedChunk;
      }
    }
  }

  private collectAffectedChunksForNodePosition(
    nodeId: number,
    selectedSources: SpatiallyIndexedSkeletonSourceEntry[],
    targetLod: number | undefined,
  ) {
    const affectedChunks = new Set<SpatiallyIndexedSkeletonChunk>();
    const selectedSourceIds = this.makeSourceIdSet(selectedSources);
    for (const entry of this.getNodeLocatorEntries(
      nodeId,
      selectedSourceIds,
      targetLod,
      { requireVisible: true },
    )) {
      affectedChunks.add(entry.chunk);
    }
    this.collectChunksReferencingParentNode(
      nodeId,
      selectedSourceIds,
      targetLod,
      affectedChunks,
    );
    return affectedChunks;
  }

  invalidateSourceCaches() {
    let invalidated = false;
    for (const chunkSource of this.iterateUniqueChunkSources()) {
      chunkSource.invalidateCache();
      invalidated = true;
    }
    if (!invalidated) {
      return false;
    }
    this.clearDetailedInteractionCaches();
    this.redrawNeeded.dispatch();
    return true;
  }

  private getChunkPositionAndSegmentArrays(
    chunk: SpatiallyIndexedSkeletonChunk,
  ) {
    const offsets = chunk.vertexAttributeOffsets;
    if (!offsets || offsets.length < 2) return undefined;
    const positions = new Float32Array(
      chunk.vertexAttributes.buffer,
      chunk.vertexAttributes.byteOffset + offsets[0],
      chunk.numVertices * 3,
    );
    const segmentIds = new Uint32Array(
      chunk.vertexAttributes.buffer,
      chunk.vertexAttributes.byteOffset + offsets[1],
      chunk.numVertices,
    );
    return { positions, segmentIds };
  }

  resolveSegmentPickFromChunk(
    chunk: SpatiallyIndexedSkeletonChunk,
    pickedOffset: number,
    kind: "node" | "edge",
  ) {
    const data = this.getChunkPositionAndSegmentArrays(chunk);
    if (data === undefined) {
      return undefined;
    }
    return resolveSpatiallyIndexedSkeletonSegmentPick(
      chunk,
      data.segmentIds,
      pickedOffset,
      kind,
    );
  }

  updateVisibleChunksForView(
    view: SpatiallyIndexedSkeletonView,
    transformedSources: readonly TransformedSource[][],
    projectionParameters: any,
    lod: number | undefined,
  ) {
    if (lod === undefined) {
      this.visibleChunksByView.delete(view);
      this.visibleChunkSetsByView.delete(view);
      this.visibleSourceIdsByView.delete(view);
      this.visibleChunkLodByView.delete(view);
      return;
    }
    const lodSuffix = `:${lod}`;
    const chunksBySource: VisibleSpatialChunksBySource = new Map();
    const chunkSetsBySource: VisibleSpatialChunkSetsBySource = new Map();
    const seenChunkKeysBySource = new Map<string, Set<string>>();
    for (const scales of transformedSources) {
      for (const tsource of scales) {
        const sourceId = getObjectId(tsource.source);
        let visibleChunks = chunksBySource.get(sourceId);
        if (visibleChunks === undefined) {
          visibleChunks = [];
          chunksBySource.set(sourceId, visibleChunks);
        }
        let visibleChunkSet = chunkSetsBySource.get(sourceId);
        if (visibleChunkSet === undefined) {
          visibleChunkSet = new Set<SpatiallyIndexedSkeletonChunk>();
          chunkSetsBySource.set(sourceId, visibleChunkSet);
        }
        let seenChunkKeys = seenChunkKeysBySource.get(sourceId);
        if (seenChunkKeys === undefined) {
          seenChunkKeys = new Set<string>();
          seenChunkKeysBySource.set(sourceId, seenChunkKeys);
        }
        forEachVisibleVolumetricChunk(
          projectionParameters,
          this.localPosition.value,
          tsource,
          (positionInChunks) => {
            const chunkKey = `${positionInChunks.join()}${lodSuffix}`;
            if (seenChunkKeys!.has(chunkKey)) {
              return;
            }
            seenChunkKeys!.add(chunkKey);
            const chunkSource =
              tsource.source as SpatiallyIndexedSkeletonSource;
            const chunk = chunkSource.chunks.get(chunkKey) as
              | SpatiallyIndexedSkeletonChunk
              | undefined;
            if (chunk?.state !== ChunkState.GPU_MEMORY) {
              return;
            }
            (visibleChunks as SpatiallyIndexedSkeletonChunk[]).push(chunk);
            (visibleChunkSet as Set<SpatiallyIndexedSkeletonChunk>).add(chunk);
          },
        );
      }
    }
    this.visibleChunksByView.set(view, chunksBySource);
    this.visibleChunkSetsByView.set(view, chunkSetsBySource);
    this.visibleSourceIdsByView.set(view, new Set(chunkSetsBySource.keys()));
    this.visibleChunkLodByView.set(view, lod);
  }

  private areVisibleChunksReady(
    transformedSources: readonly TransformedSource[][],
    projectionParameters: any,
    lod: number | undefined,
  ) {
    if (this.displayState.objectAlpha.value <= 0.0) {
      return true;
    }
    if (lod === undefined || transformedSources.length === 0) {
      return false;
    }
    const lodSuffix = `:${lod}`;
    const seenChunkKeysBySource = new Map<string, Set<string>>();
    let ready = true;
    for (const scales of transformedSources) {
      for (const tsource of scales) {
        const sourceId = getObjectId(tsource.source);
        let seenChunkKeys = seenChunkKeysBySource.get(sourceId);
        if (seenChunkKeys === undefined) {
          seenChunkKeys = new Set<string>();
          seenChunkKeysBySource.set(sourceId, seenChunkKeys);
        }
        forEachVisibleVolumetricChunk(
          projectionParameters,
          this.localPosition.value,
          tsource,
          (positionInChunks) => {
            if (!ready) {
              return;
            }
            const chunkKey = `${positionInChunks.join()}${lodSuffix}`;
            if (seenChunkKeys!.has(chunkKey)) {
              return;
            }
            seenChunkKeys!.add(chunkKey);
            const chunkSource =
              tsource.source as SpatiallyIndexedSkeletonSource;
            const chunk = chunkSource.chunks.get(chunkKey) as
              | SpatiallyIndexedSkeletonChunk
              | undefined;
            if (chunk?.state !== ChunkState.GPU_MEMORY) {
              ready = false;
            }
          },
        );
        if (!ready) {
          return false;
        }
      }
    }
    return true;
  }

  getNode(
    nodeId: number,
    options: {
      lod?: number;
    } = {},
  ): SpatiallyIndexedSkeletonNodeInfo | undefined {
    if (!Number.isSafeInteger(nodeId) || nodeId <= 0) return undefined;
    const cachedNode = this.getCachedNodeSnapshot(nodeId);
    if (cachedNode !== undefined) {
      return cachedNode;
    }
    if (!this.isDetailedInteractionActive()) {
      return undefined;
    }
    const targetLod = this.resolveTargetLod(options.lod);
    const selectedSources = this.getCurrentSourcesForEditing();
    const effectiveNode = this.getEffectiveNodeState(
      nodeId,
      this.makeSourceIdSet(selectedSources),
      targetLod,
    );
    if (effectiveNode === undefined) {
      return undefined;
    }
    return {
      nodeId,
      segmentId: effectiveNode.segmentId,
      position: new Float32Array([
        Number(effectiveNode.position[0]),
        Number(effectiveNode.position[1]),
        Number(effectiveNode.position[2]),
      ]),
      parentNodeId: effectiveNode.parentNodeId,
    };
  }

  getNodes(
    options: {
      segmentId?: bigint;
      lod?: number;
    } = {},
  ): SpatiallyIndexedSkeletonNodeInfo[] {
    const segmentFilter =
      options.segmentId === undefined ? undefined : Number(options.segmentId);
    const useSegmentFilter = Number.isFinite(segmentFilter);
    const targetLod = this.resolveTargetLod(options.lod);
    const selectedSources = this.getCurrentSourcesForEditing();
    const nodes = new Map<number, SpatiallyIndexedSkeletonNodeInfo>();
    for (const chunk of this.iterateCandidateChunks(
      selectedSources,
      targetLod,
    )) {
      const data = this.getChunkPositionAndSegmentArrays(chunk);
      if (data === undefined) continue;
      const { positions, segmentIds } = data;
      for (const [nodeId, vertexIndex] of chunk.nodeMap.entries()) {
        if (vertexIndex < 0 || vertexIndex >= chunk.numVertices) {
          continue;
        }
        if (nodes.has(nodeId)) continue;
        const segmentId = segmentIds[vertexIndex];
        if (useSegmentFilter && segmentId !== segmentFilter) {
          continue;
        }
        const effectivePosition = this.getEffectiveNodePosition(
          positions,
          vertexIndex,
          nodeId,
        );
        nodes.set(nodeId, {
          nodeId,
          segmentId,
          position: new Float32Array([
            Number(effectivePosition[0]),
            Number(effectivePosition[1]),
            Number(effectivePosition[2]),
          ]),
        });
      }
    }
    for (const nodeInfo of nodes.values()) {
      nodeInfo.parentNodeId = this.getEffectiveParentNodeId(nodeInfo.nodeId);
    }
    return [...nodes.values()].sort((a, b) => a.nodeId - b.nodeId);
  }

  addNode(
    position: ArrayLike<number>,
    options: {
      segmentId: number | bigint;
      parentNodeId?: number;
      nodeId?: number;
      lod?: number;
    },
  ): SpatiallyIndexedLocalAddNodeResult | undefined {
    const x = Number(position[0]);
    const y = Number(position[1]);
    const z = Number(position[2]);
    if (!Number.isFinite(x) || !Number.isFinite(y) || !Number.isFinite(z)) {
      return undefined;
    }
    const segmentId = Number(options.segmentId);
    if (!Number.isFinite(segmentId)) {
      return undefined;
    }
    const targetLod = this.resolveTargetLod(options.lod);
    const selectedSources = this.getCurrentVisibleSourcesForEditing(targetLod);
    if (selectedSources.length === 0) return undefined;
    const parentNodeId = options.parentNodeId;

    const maxNodeId = this.getMaxIndexedNodeId();
    const requestedNodeId = options.nodeId;
    const newNodeId =
      requestedNodeId === undefined
        ? maxNodeId + 1
        : Math.trunc(requestedNodeId);
    if (!Number.isFinite(newNodeId) || newNodeId <= 0) {
      return undefined;
    }
    if (requestedNodeId !== undefined && this.hasIndexedNodeId(newNodeId)) {
      const existingNode = this.getNode(newNodeId, { lod: targetLod });
      if (existingNode === undefined) {
        return undefined;
      }
      return {
        node: existingNode,
        localUpdateApplied: true,
      };
    }
    const localNode: SpatiallyIndexedSkeletonNodeInfo = {
      nodeId: newNodeId,
      segmentId: Math.round(segmentId),
      position: new Float32Array([x, y, z]),
      parentNodeId,
    };
    this.ensureNodeLocatorIndex();
    const parentByNodeId = new Map<number, number | undefined>(
      this.baseParentByNodeId,
    );
    const selectedSourcesById = this.makeSourceIdSet(selectedSources);
    if (
      parentNodeId !== undefined &&
      this.getBaseNodeState(parentNodeId, selectedSourcesById, targetLod, {
        requireVisible: true,
      }) === undefined
    ) {
      return {
        node: localNode,
        localUpdateApplied: false,
      };
    }
    const changedChunks = new Map<
      SpatiallyIndexedSkeletonChunk,
      SpatiallyIndexedEditableChunkData
    >();
    const affectedChunks = new Set<SpatiallyIndexedSkeletonChunk>();
    for (const sourceEntry of selectedSources) {
      const targetChunk = this.findLoadedChunkForPosition(
        sourceEntry,
        position,
        targetLod,
      );
      const editableData =
        targetChunk === undefined
          ? undefined
          : this.getEditableChunkData(targetChunk);
      if (
        targetChunk === undefined ||
        !this.isChunkVisibleForEditing(targetChunk, targetLod) ||
        editableData === undefined
      ) {
        return {
          node: localNode,
          localUpdateApplied: false,
        };
      }
      const appended = appendNodeToSpatialChunk(editableData, {
        nodeId: newNodeId,
        segmentId: Math.round(segmentId),
        position,
      });
      changedChunks.set(targetChunk, appended.data);
      affectedChunks.add(targetChunk);
    }
    parentByNodeId.set(newNodeId, parentNodeId);
    const rewrittenChunks = this.rebuildConnectionsForChunks(
      affectedChunks,
      targetLod,
      parentByNodeId,
      changedChunks,
    );
    this.finalizeEditedChunks(
      rewrittenChunks.size === 0 ? changedChunks.keys() : rewrittenChunks,
    );

    return {
      node: localNode,
      localUpdateApplied: true,
    };
  }

  deleteNode(
    nodeId: number,
    options: {
      lod?: number;
    } = {},
  ) {
    const targetLod = this.resolveTargetLod(options.lod);
    const selectedSources = this.getCurrentVisibleSourcesForEditing(targetLod);
    const deletedVertexByChunk = new Map<
      SpatiallyIndexedSkeletonChunk,
      number
    >();
    for (const chunk of this.iterateVisibleCandidateChunks(
      selectedSources,
      targetLod,
    )) {
      const vertexIndex = chunk.nodeMap.get(nodeId);
      if (vertexIndex === undefined) continue;
      if (vertexIndex < 0 || vertexIndex >= chunk.numVertices) continue;
      deletedVertexByChunk.set(chunk, vertexIndex);
    }
    if (deletedVertexByChunk.size === 0) {
      return false;
    }
    const changedChunks = new Set<SpatiallyIndexedSkeletonChunk>();

    for (const [targetChunk, targetVertexIndex] of deletedVertexByChunk) {
      const data = this.getChunkPositionAndSegmentArrays(targetChunk);
      if (data === undefined) {
        continue;
      }
      const oldNumVertices = targetChunk.numVertices;
      const { positions: oldPositions, segmentIds: oldSegmentIds } = data;
      const newNumVertices = Math.max(0, oldNumVertices - 1);
      const newPositions = new Float32Array(newNumVertices * 3);
      const newSegmentIds = new Uint32Array(newNumVertices);
      let dstVertex = 0;
      for (let srcVertex = 0; srcVertex < oldNumVertices; ++srcVertex) {
        if (srcVertex === targetVertexIndex) continue;
        const srcOffset = srcVertex * 3;
        const dstOffset = dstVertex * 3;
        newPositions[dstOffset] = oldPositions[srcOffset];
        newPositions[dstOffset + 1] = oldPositions[srcOffset + 1];
        newPositions[dstOffset + 2] = oldPositions[srcOffset + 2];
        newSegmentIds[dstVertex] = oldSegmentIds[srcVertex];
        dstVertex++;
      }

      const remapVertexIndex = (vertexIndex: number) =>
        vertexIndex > targetVertexIndex ? vertexIndex - 1 : vertexIndex;
      const newNodeMap = new Map<number, number>();
      for (const [
        candidateNodeId,
        vertexIndex,
      ] of targetChunk.nodeMap.entries()) {
        if (candidateNodeId === nodeId) continue;
        newNodeMap.set(candidateNodeId, remapVertexIndex(vertexIndex));
      }

      const newIndices: number[] = [];
      const oldIndices = targetChunk.indices;
      for (let i = 0; i < targetChunk.numIndices; i += 2) {
        const a = oldIndices[i];
        const b = oldIndices[i + 1];
        if (a === targetVertexIndex || b === targetVertexIndex) {
          continue;
        }
        newIndices.push(remapVertexIndex(a), remapVertexIndex(b));
      }

      const positionBytes = new Uint8Array(newPositions.buffer);
      const segmentBytes = new Uint8Array(newSegmentIds.buffer);
      const vertexBytes = new Uint8Array(
        positionBytes.byteLength + segmentBytes.byteLength,
      );
      vertexBytes.set(positionBytes, 0);
      vertexBytes.set(segmentBytes, positionBytes.byteLength);
      const vertexOffsets = new Uint32Array([0, positionBytes.byteLength]);

      for (const texture of targetChunk.vertexAttributeTextures) {
        this.gl.deleteTexture(texture);
      }
      targetChunk.vertexAttributes = vertexBytes;
      targetChunk.vertexAttributeOffsets = vertexOffsets;
      targetChunk.numVertices = newNumVertices;
      targetChunk.vertexAttributeTextures = uploadVertexAttributesToGPU(
        this.gl,
        targetChunk.vertexAttributes,
        targetChunk.vertexAttributeOffsets,
        targetChunk.source.attributeTextureFormats,
      );
      targetChunk.nodeMap = newNodeMap;
      targetChunk.source.bumpLookupGeneration();
      targetChunk.indices = new Uint32Array(newIndices);
      targetChunk.numIndices = targetChunk.indices.length;
      targetChunk.indexBuffer.setData(targetChunk.indices);
      changedChunks.add(targetChunk);
    }

    for (const chunk of this.iterateVisibleCandidateChunks(
      selectedSources,
      targetLod,
    )) {
      if (chunk.missingConnections.length === 0) continue;
      const deletedVertexIndex = deletedVertexByChunk.get(chunk);
      let chunkChanged = false;
      const nextConnections: typeof chunk.missingConnections = [];
      for (const connection of chunk.missingConnections) {
        if (connection.nodeId === nodeId || connection.parentId === nodeId) {
          chunkChanged = true;
          continue;
        }
        if (deletedVertexIndex !== undefined) {
          if (connection.vertexIndex === deletedVertexIndex) {
            chunkChanged = true;
            continue;
          }
          if (connection.vertexIndex > deletedVertexIndex) {
            chunkChanged = true;
            nextConnections.push({
              ...connection,
              vertexIndex: connection.vertexIndex - 1,
            });
            continue;
          }
        }
        nextConnections.push(connection);
      }
      if (chunkChanged) {
        chunk.missingConnections = nextConnections;
        changedChunks.add(chunk);
      }
    }

    if (changedChunks.size === 0) {
      return false;
    }

    for (const chunk of changedChunks) {
      this.clearFilteredChunkData(chunk);
    }
    this.invalidateNodeLocatorIndex();
    this.requestFilteredDataRedraw();
    return true;
  }

  private clearFilteredChunkData(chunk: SpatiallyIndexedSkeletonChunk) {
    if (chunk.filteredVertexAttributeTextures) {
      for (const texture of chunk.filteredVertexAttributeTextures) {
        if (texture) this.gl.deleteTexture(texture);
      }
      chunk.filteredVertexAttributeTextures = undefined;
    }
    if (chunk.filteredIndexBuffer) {
      chunk.filteredIndexBuffer.dispose();
      chunk.filteredIndexBuffer = undefined;
    }
    chunk.filteredGeneration = -1;
    chunk.filteredNodeLocatorIndexKey = undefined;
    chunk.numFilteredIndices = 0;
    chunk.numFilteredVertices = 0;
    chunk.filteredPickNodeIds = undefined;
    chunk.filteredPickNodePositions = undefined;
    chunk.filteredPickSegmentIds = undefined;
    chunk.filteredPickEdgeSegmentIds = undefined;
    chunk.filteredOldToNew = undefined;
    chunk.filteredExtraVertexMap = undefined;
    chunk.filteredEmpty = false;
  }

  private invalidateEditedChunk(chunk: SpatiallyIndexedSkeletonChunk) {
    this.clearFilteredChunkData(chunk);
  }

  private writePositionUpdate(position: ArrayLike<number>) {
    tempPositionUpdate[0] = Number(position[0]);
    tempPositionUpdate[1] = Number(position[1]);
    tempPositionUpdate[2] = Number(position[2]);
    return tempPositionUpdate;
  }

  private updateChunkPositionTexture(
    texture: WebGLTexture | null | undefined,
    numVertices: number,
    vertexIndex: number,
    position: ArrayLike<number>,
  ) {
    if (texture === undefined || texture === null) return;
    updateOneDimensionalTextureElement(
      this.gl,
      texture,
      vertexPositionTextureFormat,
      numVertices,
      vertexIndex,
      this.writePositionUpdate(position),
    );
  }

  private patchFilteredChunkPosition(
    chunk: SpatiallyIndexedSkeletonChunk,
    nodeId: number,
    position: ArrayLike<number>,
  ) {
    const filteredTexture = chunk.filteredVertexAttributeTextures?.[0];
    if (filteredTexture === undefined || filteredTexture === null) {
      return true;
    }
    let needsInvalidate = false;
    const localVertexIndex = chunk.nodeMap.get(nodeId);
    if (localVertexIndex !== undefined) {
      const filteredOldToNew = chunk.filteredOldToNew;
      if (filteredOldToNew === undefined) {
        needsInvalidate = true;
      } else {
        const filteredIndex = filteredOldToNew[localVertexIndex];
        if (filteredIndex >= 0) {
          this.updateChunkPositionTexture(
            filteredTexture,
            chunk.numFilteredVertices,
            filteredIndex,
            position,
          );
        }
      }
    }
    if (
      chunk.missingConnections.length > 0 &&
      chunk.missingConnections.some(
        (connection) => connection.parentId === nodeId,
      )
    ) {
      const filteredExtraVertexMap = chunk.filteredExtraVertexMap;
      if (filteredExtraVertexMap === undefined) {
        needsInvalidate = true;
      } else {
        const filteredIndex = filteredExtraVertexMap.get(nodeId);
        if (filteredIndex !== undefined) {
          this.updateChunkPositionTexture(
            filteredTexture,
            chunk.numFilteredVertices,
            filteredIndex,
            position,
          );
        }
      }
    }
    return !needsInvalidate;
  }

  private applyNodePositionChangeToFilteredChunks(
    affectedChunks: Set<SpatiallyIndexedSkeletonChunk>,
    nodeId: number,
    position: ArrayLike<number>,
  ) {
    if (affectedChunks.size === 0) {
      return;
    }
    const chunksToInvalidate: SpatiallyIndexedSkeletonChunk[] = [];
    for (const chunk of affectedChunks) {
      const localVertexIndex = chunk.nodeMap.get(nodeId);
      if (localVertexIndex !== undefined) {
        this.updateChunkPositionTexture(
          chunk.vertexAttributeTextures[0],
          chunk.numVertices,
          localVertexIndex,
          position,
        );
      }
      if (!this.patchFilteredChunkPosition(chunk, nodeId, position)) {
        chunksToInvalidate.push(chunk);
      }
    }
    for (const chunk of chunksToInvalidate) {
      this.invalidateEditedChunk(chunk);
    }
  }

  private collectChunksReferencingParentNode(
    nodeId: number,
    selectedSourceIds: ReadonlySet<string>,
    targetLod: number | undefined,
    affectedChunks: Set<SpatiallyIndexedSkeletonChunk>,
  ) {
    this.ensureNodeLocatorIndex();
    for (const entry of this.parentReferenceIndex.get(nodeId) ?? []) {
      if (!selectedSourceIds.has(entry.sourceId)) continue;
      if (entry.chunk.state !== ChunkState.GPU_MEMORY) continue;
      if (!this.lodMatches(entry.chunk, targetLod)) continue;
      if (!this.isChunkVisibleForEditing(entry.chunk, targetLod)) continue;
      affectedChunks.add(entry.chunk);
    }
  }

  private getLoadedChildNodeIds(
    rootNodeId: number,
    options: {
      lod?: number;
    } = {},
  ) {
    const targetLod = this.resolveTargetLod(options.lod);
    const selectedSources = this.getCurrentVisibleSourcesForEditing(targetLod);
    const childrenByParent = new Map<number, number[]>();
    for (const chunk of this.iterateVisibleCandidateChunks(
      selectedSources,
      targetLod,
    )) {
      const vertexToNodeId = new Map<number, number>();
      for (const [nodeId, vertexIndex] of chunk.nodeMap.entries()) {
        vertexToNodeId.set(vertexIndex, nodeId);
      }
      const indices = chunk.indices;
      for (let i = 0; i < chunk.numIndices; i += 2) {
        const childNodeId = vertexToNodeId.get(indices[i]);
        const parentNodeId = vertexToNodeId.get(indices[i + 1]);
        if (
          childNodeId === undefined ||
          parentNodeId === undefined ||
          childNodeId === parentNodeId
        ) {
          continue;
        }
        let children = childrenByParent.get(parentNodeId);
        if (children === undefined) {
          children = [];
          childrenByParent.set(parentNodeId, children);
        }
        children.push(childNodeId);
      }
      for (const connection of chunk.missingConnections) {
        if (
          !Number.isFinite(connection.nodeId) ||
          !Number.isFinite(connection.parentId) ||
          connection.nodeId === connection.parentId
        ) {
          continue;
        }
        let children = childrenByParent.get(connection.parentId);
        if (children === undefined) {
          children = [];
          childrenByParent.set(connection.parentId, children);
        }
        children.push(connection.nodeId);
      }
    }
    const descendants: number[] = [];
    const queue = [rootNodeId];
    const visited = new Set<number>();
    for (let queueIndex = 0; queueIndex < queue.length; ++queueIndex) {
      const nodeId = queue[queueIndex];
      if (visited.has(nodeId)) continue;
      visited.add(nodeId);
      descendants.push(nodeId);
      for (const childNodeId of childrenByParent.get(nodeId) ?? []) {
        queue.push(childNodeId);
      }
    }
    return descendants;
  }

  setNodeParent(
    nodeId: number,
    parentNodeId: number | undefined,
    options: {
      lod?: number;
    } = {},
  ) {
    const normalizedNodeId = Math.round(Number(nodeId));
    if (!Number.isSafeInteger(normalizedNodeId) || normalizedNodeId <= 0) {
      return false;
    }
    const normalizedParentNodeId =
      parentNodeId === undefined ? undefined : Math.round(Number(parentNodeId));
    if (
      normalizedParentNodeId !== undefined &&
      (!Number.isSafeInteger(normalizedParentNodeId) ||
        normalizedParentNodeId <= 0)
    ) {
      return false;
    }
    if (normalizedParentNodeId === normalizedNodeId) {
      return false;
    }

    const targetLod = this.resolveTargetLod(options.lod);
    const selectedSources = this.getCurrentVisibleSourcesForEditing(targetLod);
    const selectedSourceIds = this.makeSourceIdSet(selectedSources);
    const childEntry = this.getPrimaryNodeLocator(
      normalizedNodeId,
      selectedSourceIds,
      targetLod,
      { requireVisible: true },
    );
    if (childEntry === undefined) {
      return false;
    }
    const childChunk = childEntry.chunk;
    if (childChunk.state !== ChunkState.GPU_MEMORY) {
      return false;
    }
    const childData = this.getChunkPositionAndSegmentArrays(childChunk);
    if (childData === undefined) {
      return false;
    }
    const childSegmentId = childData.segmentIds[childEntry.vertexIndex];
    let changed = false;
    const changedChunks = new Set<SpatiallyIndexedSkeletonChunk>();

    const removeExistingParentEdge = (
      chunk: SpatiallyIndexedSkeletonChunk,
      vertexIndex: number | undefined,
    ) => {
      if (vertexIndex !== undefined && chunk.numIndices > 0) {
        const nextIndices: number[] = [];
        let chunkChanged = false;
        for (let i = 0; i < chunk.numIndices; i += 2) {
          if (chunk.indices[i] === vertexIndex) {
            chunkChanged = true;
            continue;
          }
          nextIndices.push(chunk.indices[i], chunk.indices[i + 1]);
        }
        if (chunkChanged) {
          chunk.indices = new Uint32Array(nextIndices);
          chunk.numIndices = chunk.indices.length;
          chunk.indexBuffer.setData(chunk.indices);
          changed = true;
          changedChunks.add(chunk);
        }
      }
      if (chunk.missingConnections.length === 0) return;
      const nextConnections: typeof chunk.missingConnections = [];
      let chunkChanged = false;
      for (const connection of chunk.missingConnections) {
        if (connection.nodeId === normalizedNodeId) {
          chunkChanged = true;
          continue;
        }
        nextConnections.push(connection);
      }
      if (chunkChanged) {
        chunk.missingConnections = nextConnections;
        changed = true;
        changedChunks.add(chunk);
      }
    };

    for (const chunk of this.iterateVisibleCandidateChunks(
      selectedSources,
      targetLod,
    )) {
      const vertexIndex =
        chunk === childChunk ? childEntry.vertexIndex : undefined;
      removeExistingParentEdge(chunk, vertexIndex);
    }

    if (normalizedParentNodeId !== undefined) {
      const parentEntry = this.getPrimaryNodeLocator(
        normalizedParentNodeId,
        selectedSourceIds,
        targetLod,
        { requireVisible: true },
      );
      if (
        parentEntry !== undefined &&
        parentEntry.chunk.state === ChunkState.GPU_MEMORY
      ) {
        if (parentEntry.chunk === childChunk) {
          const indices = childChunk.indices;
          let hasEdge = false;
          for (let i = 0; i < childChunk.numIndices; i += 2) {
            if (
              indices[i] === childEntry.vertexIndex &&
              indices[i + 1] === parentEntry.vertexIndex
            ) {
              hasEdge = true;
              break;
            }
          }
          if (!hasEdge) {
            const nextIndices = new Uint32Array(indices.length + 2);
            nextIndices.set(indices);
            nextIndices[indices.length] = childEntry.vertexIndex;
            nextIndices[indices.length + 1] = parentEntry.vertexIndex;
            childChunk.indices = nextIndices;
            childChunk.numIndices = nextIndices.length;
            childChunk.indexBuffer.setData(childChunk.indices);
            changed = true;
            changedChunks.add(childChunk);
          }
        } else {
          const existingConnection = childChunk.missingConnections.find(
            (connection) =>
              connection.nodeId === normalizedNodeId &&
              connection.parentId === normalizedParentNodeId,
          );
          if (existingConnection === undefined) {
            childChunk.missingConnections.push({
              nodeId: normalizedNodeId,
              parentId: normalizedParentNodeId,
              vertexIndex: childEntry.vertexIndex,
              skeletonId: childSegmentId,
            });
            changed = true;
            changedChunks.add(childChunk);
          }
        }
      } else {
        const existingConnection = childChunk.missingConnections.find(
          (connection) =>
            connection.nodeId === normalizedNodeId &&
            connection.parentId === normalizedParentNodeId,
        );
        if (existingConnection === undefined) {
          childChunk.missingConnections.push({
            nodeId: normalizedNodeId,
            parentId: normalizedParentNodeId,
            vertexIndex: childEntry.vertexIndex,
            skeletonId: childSegmentId,
          });
          changed = true;
          changedChunks.add(childChunk);
        }
      }
    }

    if (!changed) {
      return false;
    }
    for (const chunk of changedChunks) {
      this.invalidateEditedChunk(chunk);
    }
    this.invalidateNodeLocatorIndex();
    this.requestFilteredDataRedraw();
    return true;
  }

  splitSkeletonAtNode(
    nodeId: number,
    newSegmentId: number | bigint,
    options: {
      lod?: number;
    } = {},
  ) {
    const descendantNodeIds = this.getLoadedChildNodeIds(nodeId, options);
    const segmentChanged =
      descendantNodeIds.length === 0
        ? false
        : this.setNodeSegmentIds(
            descendantNodeIds.map((descendantNodeId) => ({
              nodeId: descendantNodeId,
              segmentId: newSegmentId,
            })),
            options,
          );
    const parentChanged = this.setNodeParent(nodeId, undefined, options);
    return segmentChanged || parentChanged;
  }

  mergeSkeletonNodes(options: {
    parentNodeId: number;
    childNodeId: number;
    resultSegmentId: number | bigint;
    mergedSegmentId: number | bigint;
    lod?: number;
  }) {
    const { parentNodeId, childNodeId, resultSegmentId, mergedSegmentId, lod } =
      options;
    const mergedNodes = this.getNodes({
      lod,
      segmentId: BigInt(Number(mergedSegmentId)),
    });
    const segmentChanged =
      mergedNodes.length === 0
        ? false
        : this.setNodeSegmentIds(
            mergedNodes.map((node) => ({
              nodeId: node.nodeId,
              segmentId: resultSegmentId,
            })),
            { lod },
          );
    const parentChanged = this.setNodeParent(childNodeId, parentNodeId, {
      lod,
    });
    return segmentChanged || parentChanged;
  }

  setNodeSegmentIds(
    updates: Iterable<{
      nodeId: number;
      segmentId: number | bigint;
    }>,
    options: {
      lod?: number;
    } = {},
  ) {
    const segmentByNodeId = new Map<number, number>();
    for (const update of updates) {
      const nodeId = Number(update.nodeId);
      const segmentId = Number(update.segmentId);
      if (!Number.isFinite(nodeId) || !Number.isFinite(segmentId)) {
        continue;
      }
      const roundedNodeId = Math.round(nodeId);
      if (roundedNodeId <= 0) continue;
      segmentByNodeId.set(roundedNodeId, Math.round(segmentId));
    }
    if (segmentByNodeId.size === 0) {
      return false;
    }

    const targetLod = this.resolveTargetLod(options.lod);
    const selectedSources = this.getCurrentVisibleSourcesForEditing(targetLod);
    let changed = false;
    const changedChunks = new Set<SpatiallyIndexedSkeletonChunk>();
    for (const chunk of this.iterateVisibleCandidateChunks(
      selectedSources,
      targetLod,
    )) {
      const data = this.getChunkPositionAndSegmentArrays(chunk);
      if (data === undefined) continue;
      const { segmentIds } = data;
      let chunkChanged = false;
      for (const [nodeId, vertexIndex] of chunk.nodeMap.entries()) {
        const nextSegmentId = segmentByNodeId.get(nodeId);
        if (nextSegmentId === undefined) continue;
        if (vertexIndex < 0 || vertexIndex >= chunk.numVertices) {
          continue;
        }
        if (segmentIds[vertexIndex] === nextSegmentId) {
          continue;
        }
        segmentIds[vertexIndex] = nextSegmentId;
        changed = true;
        chunkChanged = true;
      }
      if (chunk.missingConnections.length === 0) {
        if (chunkChanged) {
          changedChunks.add(chunk);
        }
        continue;
      }
      let chunkConnectionsChanged = false;
      const nextConnections: typeof chunk.missingConnections = [];
      for (const connection of chunk.missingConnections) {
        const nextSegmentId = segmentByNodeId.get(connection.nodeId);
        if (
          nextSegmentId !== undefined &&
          Math.round(connection.skeletonId) !== nextSegmentId
        ) {
          chunkConnectionsChanged = true;
          nextConnections.push({
            ...connection,
            skeletonId: nextSegmentId,
          });
        } else {
          nextConnections.push(connection);
        }
      }
      if (chunkConnectionsChanged) {
        chunk.missingConnections = nextConnections;
        changed = true;
      }
      if (chunkChanged || chunkConnectionsChanged) {
        changedChunks.add(chunk);
      }
    }
    if (changed) {
      for (const chunk of changedChunks) {
        this.clearFilteredChunkData(chunk);
      }
      this.requestFilteredDataRedraw();
    }
    return changed;
  }

  setNodePosition(
    nodeId: number,
    position: ArrayLike<number>,
    options: {
      lod?: number;
    } = {},
  ): SpatiallyIndexedLocalMoveNodeResult {
    const x = Number(position[0]);
    const y = Number(position[1]);
    const z = Number(position[2]);
    if (!Number.isFinite(x) || !Number.isFinite(y) || !Number.isFinite(z)) {
      return {
        localUpdateApplied: false,
        changed: false,
      };
    }
    const targetLod = this.resolveTargetLod(options.lod);
    const selectedSources = this.getCurrentVisibleSourcesForEditing(targetLod);
    if (selectedSources.length === 0) {
      return {
        localUpdateApplied: false,
        changed: false,
      };
    }
    const selectedSourceIds = this.makeSourceIdSet(selectedSources);
    const nextPosition = new Float32Array([x, y, z]);
    this.ensureNodeLocatorIndex();
    const baseNodeState = this.getBaseNodeState(
      nodeId,
      selectedSourceIds,
      targetLod,
      { includePendingPosition: false, requireVisible: true },
    );
    if (baseNodeState === undefined) {
      return {
        localUpdateApplied: false,
        changed: false,
      };
    }
    if (this.positionsEqual(baseNodeState.position, nextPosition)) {
      return {
        localUpdateApplied: true,
        changed: false,
      };
    }
    const changedChunkData = new Map<
      SpatiallyIndexedSkeletonChunk,
      SpatiallyIndexedEditableChunkData
    >();
    const affectedChunks = this.collectAffectedChunksForNodePosition(
      nodeId,
      selectedSources,
      targetLod,
    );
    const parentByNodeId = new Map<number, number | undefined>(
      this.baseParentByNodeId,
    );
    for (const sourceEntry of selectedSources) {
      const sourceId = getObjectId(sourceEntry.chunkSource);
      const sourceIds = new Set([sourceId]);
      const oldEntries = this.getNodeLocatorEntries(
        nodeId,
        sourceIds,
        targetLod,
        { requireVisible: true },
      );
      const newChunk = this.findLoadedChunkForPosition(
        sourceEntry,
        nextPosition,
        targetLod,
      );
      if (
        oldEntries.length === 0 ||
        newChunk === undefined ||
        !this.isChunkVisibleForEditing(newChunk, targetLod)
      ) {
        return {
          localUpdateApplied: false,
          changed: false,
        };
      }
      const oldChunks = new Set<SpatiallyIndexedSkeletonChunk>();
      for (const entry of oldEntries) {
        oldChunks.add(entry.chunk);
      }
      const singleResidentChunk =
        oldChunks.size === 1 && oldChunks.has(newChunk);
      if (singleResidentChunk) {
        const oldChunkData =
          changedChunkData.get(newChunk) ?? this.getEditableChunkData(newChunk);
        if (oldChunkData === undefined) {
          return {
            localUpdateApplied: false,
            changed: false,
          };
        }
        const updated = updateNodePositionInSpatialChunk(
          oldChunkData,
          nodeId,
          nextPosition,
        );
        if (updated === undefined) {
          return {
            localUpdateApplied: false,
            changed: false,
          };
        }
        changedChunkData.set(newChunk, updated.data);
        affectedChunks.add(newChunk);
        continue;
      }
      for (const oldChunk of oldChunks) {
        const oldChunkData =
          changedChunkData.get(oldChunk) ?? this.getEditableChunkData(oldChunk);
        if (oldChunkData === undefined) {
          return {
            localUpdateApplied: false,
            changed: false,
          };
        }
        const removed = removeNodeFromSpatialChunk(oldChunkData, nodeId);
        if (removed === undefined) {
          return {
            localUpdateApplied: false,
            changed: false,
          };
        }
        changedChunkData.set(oldChunk, removed.data);
        affectedChunks.add(oldChunk);
      }
      const newChunkData =
        changedChunkData.get(newChunk) ?? this.getEditableChunkData(newChunk);
      if (newChunkData === undefined) {
        return {
          localUpdateApplied: false,
          changed: false,
        };
      }
      const appended = appendNodeToSpatialChunk(newChunkData, {
        nodeId,
        segmentId: baseNodeState.segmentId,
        position: nextPosition,
      });
      changedChunkData.set(newChunk, appended.data);
      affectedChunks.add(newChunk);
    }
    const rewrittenChunks = this.rebuildConnectionsForChunks(
      affectedChunks,
      targetLod,
      parentByNodeId,
      changedChunkData,
    );
    this.finalizeEditedChunks(
      rewrittenChunks.size === 0 ? affectedChunks : rewrittenChunks,
    );
    return {
      localUpdateApplied: true,
      changed: true,
    };
  }

  previewNodePosition(
    nodeId: number,
    position: ArrayLike<number>,
    options: {
      lod?: number;
    } = {},
  ) {
    const x = Number(position[0]);
    const y = Number(position[1]);
    const z = Number(position[2]);
    if (!Number.isFinite(x) || !Number.isFinite(y) || !Number.isFinite(z)) {
      return false;
    }
    const targetLod = this.resolveTargetLod(options.lod);
    const selectedSources = this.getCurrentVisibleSourcesForEditing(targetLod);
    const affectedChunks = this.collectAffectedChunksForNodePosition(
      nodeId,
      selectedSources,
      targetLod,
    );
    if (affectedChunks.size === 0) {
      return false;
    }
    this.applyNodePositionChangeToFilteredChunks(
      affectedChunks,
      nodeId,
      position,
    );
    this.redrawNeeded.dispatch();
    return true;
  }

  clearPreviewNodePosition(
    nodeId: number,
    options: {
      lod?: number;
    } = {},
  ) {
    const targetLod = this.resolveTargetLod(options.lod);
    const selectedSources = this.getCurrentVisibleSourcesForEditing(targetLod);
    const committedPosition = this.getCommittedNodePosition(
      nodeId,
      selectedSources,
      targetLod,
    );
    if (committedPosition === undefined) {
      return false;
    }
    const affectedChunks = this.collectAffectedChunksForNodePosition(
      nodeId,
      selectedSources,
      targetLod,
    );
    if (affectedChunks.size === 0) {
      return false;
    }
    this.applyNodePositionChangeToFilteredChunks(
      affectedChunks,
      nodeId,
      committedPosition,
    );
    this.redrawNeeded.dispatch();
    return true;
  }

  draw(
    renderContext: SliceViewPanelRenderContext | PerspectiveViewRenderContext,
    layer: RenderLayer,
    renderHelper: RenderHelper,
    browseRenderHelper: RenderHelper,
    renderOptions: ViewSpecificSkeletonRenderingOptions,
    attachment: VisibleLayerInfo<
      LayerView,
      ThreeDimensionalRenderLayerAttachmentState
    >,
    drawOptions?: {
      view?: SpatiallyIndexedSkeletonView;
      gridLevel?: number;
      lod?: number;
    },
  ) {
    const lineWidth = renderOptions.lineWidth.value;
    const { gl, displayState } = this;
    if (displayState.objectAlpha.value <= 0.0) {
      return;
    }
    const modelMatrix = update3dRenderLayerAttachment(
      displayState.transform.value,
      renderContext.projectionParameters.displayDimensionRenderInfo,
      attachment,
    );
    if (modelMatrix === undefined) return;

    const hasRegularSkeletonLayer = this.updateHasRegularSkeletonLayerWatchable(
      layer.userLayer,
    );
    const targetLod = drawOptions?.lod;
    const view = drawOptions?.view ?? "3d";
    const detailedInteractionActive = this.isDetailedInteractionActive();
    const isEditModeActive = this.editMode?.value === true;
    const pointDiameter = getSkeletonNodeDiameter(
      renderOptions.mode.value,
      lineWidth,
    );

    const activeRenderHelper = detailedInteractionActive
      ? renderHelper
      : browseRenderHelper;
    const edgeShaderResult = activeRenderHelper.edgeShaderGetter(
      renderContext.emitter,
    );
    const nodeShaderResult = activeRenderHelper.nodeShaderGetter(
      renderContext.emitter,
    );
    const { shader: edgeShader, parameters: edgeShaderParameters } =
      edgeShaderResult;
    const { shader: nodeShader, parameters: nodeShaderParameters } =
      nodeShaderResult;
    if (edgeShader === null || nodeShader === null) {
      return;
    }

    const { shaderControlState } = this.displayState.skeletonRenderingOptions;

    edgeShader.bind();
    activeRenderHelper.beginLayer(gl, edgeShader, renderContext, modelMatrix);
    activeRenderHelper.setEdgePickInstanceStride(gl, edgeShader, 0);
    setControlsInShader(
      gl,
      edgeShader,
      shaderControlState,
      edgeShaderParameters.parseResult.controls,
    );
    gl.uniform1f(edgeShader.uniform("uLineWidth"), lineWidth!);

    nodeShader.bind();
    activeRenderHelper.beginLayer(gl, nodeShader, renderContext, modelMatrix);
    gl.uniform1f(nodeShader.uniform("uNodeDiameter"), pointDiameter);
    activeRenderHelper.setNodePickInstanceStride(gl, nodeShader, 0);
    setControlsInShader(
      gl,
      nodeShader,
      shaderControlState,
      nodeShaderParameters.parseResult.controls,
    );

    const baseColor = new Float32Array([1, 1, 1, 1]);
    edgeShader.bind();
    activeRenderHelper.setColor(gl, edgeShader, baseColor);
    activeRenderHelper.enableDynamicSegmentAppearance(
      gl,
      edgeShader,
      hasRegularSkeletonLayer,
    );
    nodeShader.bind();
    activeRenderHelper.setColor(gl, nodeShader, baseColor);
    activeRenderHelper.enableDynamicSegmentAppearance(
      gl,
      nodeShader,
      hasRegularSkeletonLayer,
    );
    if (renderContext.emitPickID) {
      edgeShader.bind();
      activeRenderHelper.setPickID(gl, edgeShader, 0);
      activeRenderHelper.setEdgePickInstanceStride(gl, edgeShader, 0);
      nodeShader.bind();
      activeRenderHelper.setPickID(gl, nodeShader, 0);
      activeRenderHelper.setNodePickInstanceStride(gl, nodeShader, 0);
    }

    const selectedSources = this.selectSourcesForViewAndGrid(
      view,
      drawOptions?.gridLevel,
    );
    const selectedSourceIds = detailedInteractionActive
      ? this.makeSourceIdSet(selectedSources)
      : undefined;
    const nodeLocatorIndexKey = detailedInteractionActive
      ? this.makeNodeLocatorIndexKey(this.getIndexedSources())
      : undefined;
    for (const chunk of this.iterateCandidateChunks(
      selectedSources,
      targetLod,
      {
        view,
      },
    )) {
      const chunkToDraw = detailedInteractionActive
        ? this.updateChunkFilteredBuffer(
            chunk,
            selectedSourceIds!,
            targetLod,
            nodeLocatorIndexKey!,
          )
        : chunk;
      if (chunkToDraw === null) {
        continue;
      }
      if (renderContext.emitPickID) {
        let edgePickId = 0;
        let edgePickStride = 0;
        let nodePickId = 0;
        let nodePickStride = 0;
        if (detailedInteractionActive) {
          const filteredChunk = chunkToDraw as SkeletonChunkInterface;
          const pickEdgeSegmentIds = filteredChunk.pickEdgeSegmentIds;
          const pickNodeIds = filteredChunk.pickNodeIds;
          const pickSegmentIds = filteredChunk.pickSegmentIds;
          if (
            !isEditModeActive &&
            pickEdgeSegmentIds !== undefined &&
            pickEdgeSegmentIds.length > 0
          ) {
            const pickData: SpatiallyIndexedSkeletonPickData = {
              kind: "edge",
              segmentIds: pickEdgeSegmentIds,
            };
            edgePickId = renderContext.pickIDs.register(
              layer,
              pickEdgeSegmentIds.length,
              0n,
              pickData,
            );
            edgePickStride = 1;
          }
          if (
            pickNodeIds !== undefined &&
            filteredChunk.pickNodePositions !== undefined &&
            pickSegmentIds !== undefined &&
            filteredChunk.numVertices > 0
          ) {
            const pickData: SpatiallyIndexedSkeletonPickData = {
              kind: "node",
              nodeIds: pickNodeIds,
              nodePositions: filteredChunk.pickNodePositions,
              segmentIds: pickSegmentIds,
            };
            nodePickId = renderContext.pickIDs.register(
              layer,
              filteredChunk.numVertices,
              0n,
              pickData,
            );
            nodePickStride = 1;
          }
        } else {
          if (chunk.numIndices > 0) {
            edgePickId = renderContext.pickIDs.register(
              layer,
              chunk.numIndices / 2,
              0n,
              {
                kind: "segment-edge",
                chunk,
              } satisfies SpatiallyIndexedSkeletonPickData,
            );
            edgePickStride = 1;
          }
          if (chunk.numVertices > 0) {
            nodePickId = renderContext.pickIDs.register(
              layer,
              chunk.numVertices,
              0n,
              {
                kind: "segment-node",
                chunk,
              } satisfies SpatiallyIndexedSkeletonPickData,
            );
            nodePickStride = 1;
          }
        }
        edgeShader.bind();
        activeRenderHelper.setPickID(gl, edgeShader, edgePickId);
        activeRenderHelper.setEdgePickInstanceStride(
          gl,
          edgeShader,
          edgePickStride,
        );
        nodeShader.bind();
        activeRenderHelper.setPickID(gl, nodeShader, nodePickId);
        activeRenderHelper.setNodePickInstanceStride(
          gl,
          nodeShader,
          nodePickStride,
        );
      }
      activeRenderHelper.drawSkeleton(
        gl,
        edgeShader,
        nodeShader,
        chunkToDraw,
        renderContext.projectionParameters,
      );
    }

    activeRenderHelper.disableDynamicSegmentAppearance(gl, edgeShader);
    activeRenderHelper.disableDynamicSegmentAppearance(gl, nodeShader);
    activeRenderHelper.endLayer(gl, edgeShader, nodeShader);
  }

  updateChunkFilteredBuffer(
    chunk: SpatiallyIndexedSkeletonChunk,
    selectedSourceIds: ReadonlySet<string>,
    targetLod: number | undefined,
    nodeLocatorIndexKey: string,
  ): SkeletonChunkInterface | null {
    const getCachedFilteredChunk = ():
      | SkeletonChunkInterface
      | null
      | undefined => {
      if (
        chunk.filteredGeneration !== this.generation ||
        chunk.filteredNodeLocatorIndexKey !== nodeLocatorIndexKey
      ) {
        return undefined;
      }
      if (chunk.filteredEmpty) {
        return null;
      }
      if (
        chunk.filteredIndexBuffer &&
        chunk.filteredVertexAttributeTextures &&
        chunk.filteredPickNodeIds &&
        chunk.filteredPickNodePositions &&
        chunk.filteredPickSegmentIds &&
        chunk.filteredPickEdgeSegmentIds &&
        chunk.numFilteredIndices > 0 &&
        chunk.numFilteredVertices > 0
      ) {
        return {
          vertexAttributeTextures: chunk.filteredVertexAttributeTextures,
          indexBuffer: chunk.filteredIndexBuffer,
          numIndices: chunk.numFilteredIndices,
          numVertices: chunk.numFilteredVertices,
          pickNodeIds: chunk.filteredPickNodeIds,
          pickNodePositions: chunk.filteredPickNodePositions,
          pickSegmentIds: chunk.filteredPickSegmentIds,
          pickEdgeSegmentIds: chunk.filteredPickEdgeSegmentIds,
        };
      }
      return undefined;
    };
    const cachedFilteredChunk = getCachedFilteredChunk();
    if (cachedFilteredChunk !== undefined) {
      return cachedFilteredChunk;
    }
    this.ensureNodeLocatorIndex();
    let missingConnectionsHash = 0;
    const updateHash = (value: number) => {
      missingConnectionsHash =
        ((missingConnectionsHash * 1664525) ^ (value >>> 0)) >>> 0;
    };
    const gl = this.gl;
    const setEmptyFilteredChunkState = () => {
      chunk.filteredGeneration = this.generation;
      chunk.filteredMissingConnectionsHash = missingConnectionsHash;
      chunk.filteredNodeLocatorIndexKey = nodeLocatorIndexKey;
      chunk.numFilteredIndices = 0;
      chunk.numFilteredVertices = 0;
      chunk.filteredPickNodeIds = undefined;
      chunk.filteredPickNodePositions = undefined;
      chunk.filteredPickSegmentIds = undefined;
      chunk.filteredPickEdgeSegmentIds = undefined;
      chunk.filteredOldToNew = undefined;
      chunk.filteredExtraVertexMap = undefined;
      chunk.filteredEmpty = true;
    };
    const disposeFilteredTextures = () => {
      if (chunk.filteredVertexAttributeTextures) {
        for (const tex of chunk.filteredVertexAttributeTextures) {
          if (tex) gl.deleteTexture(tex);
        }
        chunk.filteredVertexAttributeTextures = undefined;
      }
    };

    const vertexAttributeOffsets = chunk.vertexAttributeOffsets;
    if (!vertexAttributeOffsets || vertexAttributeOffsets.length < 2) {
      disposeFilteredTextures();
      setEmptyFilteredChunkState();
      return null;
    }

    const data = this.getChunkPositionAndSegmentArrays(chunk);
    if (data === undefined) {
      disposeFilteredTextures();
      setEmptyFilteredChunkState();
      return null;
    }
    const { positions, segmentIds } = data;
    const selectedNodeId = this.selectedNodeId?.value;
    const oldToNew = new Int32Array(chunk.numVertices);
    for (let i = 0; i < chunk.numVertices; ++i) {
      oldToNew[i] = -1;
    }

    const residentNodes: SpatiallyIndexedResidentNodeState[] = [];
    const residentNodeIndex = new Map<number, number>();
    for (const [nodeId, vertexIndex] of chunk.nodeMap.entries()) {
      if (vertexIndex < 0 || vertexIndex >= chunk.numVertices) {
        continue;
      }
      const residentIndex = residentNodes.length;
      oldToNew[vertexIndex] = residentIndex;
      residentNodeIndex.set(nodeId, residentIndex);
      residentNodes.push({
        nodeId,
        segmentId: segmentIds[vertexIndex],
        position: this.getEffectiveNodePosition(positions, vertexIndex, nodeId),
        parentNodeId: this.getEffectiveParentNodeId(nodeId),
        selected: selectedNodeId !== undefined && selectedNodeId === nodeId,
      });
    }

    const filteredIndices: number[] = [];
    const extraPositions: number[] = [];
    const extraSegments: number[] = [];
    const extraSelected: number[] = [];
    const extraNodeIds: number[] = [];
    const extraVertexMap = new Map<number, number>();

    for (const residentNode of residentNodes) {
      const childIndex = residentNodeIndex.get(residentNode.nodeId);
      if (childIndex === undefined) {
        continue;
      }
      const parentNodeId = residentNode.parentNodeId;
      if (
        parentNodeId === undefined ||
        !Number.isFinite(parentNodeId) ||
        parentNodeId === residentNode.nodeId
      ) {
        continue;
      }
      updateHash(residentNode.nodeId);
      updateHash(parentNodeId);
      const localParentIndex = residentNodeIndex.get(parentNodeId);
      if (localParentIndex !== undefined) {
        updateHash(localParentIndex + 1);
        filteredIndices.push(childIndex, localParentIndex);
        continue;
      }
      const parentNode = this.getEffectiveNodeState(
        parentNodeId,
        selectedSourceIds,
        targetLod,
      );
      if (parentNode === undefined) {
        updateHash(0);
        continue;
      }
      let parentNew = extraVertexMap.get(parentNodeId);
      if (parentNew === undefined) {
        parentNew = residentNodes.length + extraPositions.length / 3;
        extraVertexMap.set(parentNodeId, parentNew);
        extraPositions.push(
          Number(parentNode.position[0]),
          Number(parentNode.position[1]),
          Number(parentNode.position[2]),
        );
        extraSegments.push(parentNode.segmentId);
        extraNodeIds.push(parentNodeId);
        extraSelected.push(
          selectedNodeId !== undefined && parentNodeId === selectedNodeId
            ? 1
            : 0,
        );
      }
      filteredIndices.push(childIndex, parentNew);
      updateHash(parentNew + 1);
    }

    if (
      chunk.filteredGeneration === this.generation &&
      chunk.filteredMissingConnectionsHash === missingConnectionsHash
    ) {
      chunk.filteredNodeLocatorIndexKey = nodeLocatorIndexKey;
      const reusableFilteredChunk = getCachedFilteredChunk();
      if (reusableFilteredChunk !== undefined) {
        return reusableFilteredChunk;
      }
    }

    if (residentNodes.length === 0 || filteredIndices.length === 0) {
      disposeFilteredTextures();
      setEmptyFilteredChunkState();
      return null;
    }

    const extraVertexCount = extraPositions.length / 3;
    const totalVertexCount = residentNodes.length + extraVertexCount;
    const filteredPositions = new Float32Array(totalVertexCount * 3);
    const filteredSegments = new Uint32Array(totalVertexCount);
    const filteredSelected = new Float32Array(totalVertexCount);
    const filteredPickNodeIds = new Int32Array(totalVertexCount);
    filteredPickNodeIds.fill(-1);
    const filteredPickNodePositions = new Float32Array(totalVertexCount * 3);
    const filteredPickSegmentIds = new Uint32Array(totalVertexCount);
    for (let i = 0; i < residentNodes.length; ++i) {
      const residentNode = residentNodes[i];
      const dstStart = i * 3;
      filteredPositions[dstStart] = Number(residentNode.position[0]);
      filteredPositions[dstStart + 1] = Number(residentNode.position[1]);
      filteredPositions[dstStart + 2] = Number(residentNode.position[2]);
      filteredSegments[i] = residentNode.segmentId;
      filteredSelected[i] = residentNode.selected ? 1 : 0;
      filteredPickNodeIds[i] = residentNode.nodeId;
      filteredPickNodePositions[dstStart] = filteredPositions[dstStart];
      filteredPickNodePositions[dstStart + 1] = filteredPositions[dstStart + 1];
      filteredPickNodePositions[dstStart + 2] = filteredPositions[dstStart + 2];
      filteredPickSegmentIds[i] = residentNode.segmentId;
    }
    for (let i = 0; i < extraVertexCount; ++i) {
      const dstVertex = residentNodes.length + i;
      const posStart = i * 3;
      const dstStart = dstVertex * 3;
      filteredPositions[dstStart] = extraPositions[posStart];
      filteredPositions[dstStart + 1] = extraPositions[posStart + 1];
      filteredPositions[dstStart + 2] = extraPositions[posStart + 2];
      filteredSegments[dstVertex] = extraSegments[i];
      filteredSelected[dstVertex] = extraSelected[i] ?? 0;
      filteredPickNodeIds[dstVertex] = extraNodeIds[i] ?? -1;
      filteredPickNodePositions[dstStart] = filteredPositions[dstStart];
      filteredPickNodePositions[dstStart + 1] = filteredPositions[dstStart + 1];
      filteredPickNodePositions[dstStart + 2] = filteredPositions[dstStart + 2];
      filteredPickSegmentIds[dstVertex] = extraSegments[i] ?? 0;
    }
    const filteredPickEdgeSegmentIds = new Uint32Array(
      filteredIndices.length / 2,
    );
    for (
      let edgeIndex = 0, indexOffset = 0;
      indexOffset < filteredIndices.length;
      indexOffset += 2, ++edgeIndex
    ) {
      const vertexA = filteredIndices[indexOffset];
      const vertexB = filteredIndices[indexOffset + 1];
      let segmentId = filteredPickSegmentIds[vertexA];
      if (!Number.isSafeInteger(segmentId) || segmentId <= 0) {
        segmentId = filteredPickSegmentIds[vertexB];
      }
      filteredPickEdgeSegmentIds[edgeIndex] = segmentId;
    }

    const posBytes = new Uint8Array(filteredPositions.buffer);
    const segBytes = new Uint8Array(filteredSegments.buffer);
    const selectedBytes = new Uint8Array(filteredSelected.buffer);
    const vertexBytes = new Uint8Array(
      posBytes.byteLength + segBytes.byteLength + selectedBytes.byteLength,
    );
    vertexBytes.set(posBytes, 0);
    vertexBytes.set(segBytes, posBytes.byteLength);
    vertexBytes.set(selectedBytes, posBytes.byteLength + segBytes.byteLength);
    const vertexOffsets = new Uint32Array([
      0,
      posBytes.byteLength,
      posBytes.byteLength + segBytes.byteLength,
    ]);

    disposeFilteredTextures();
    chunk.filteredVertexAttributeTextures = uploadVertexAttributesToGPU(
      gl,
      vertexBytes,
      vertexOffsets,
      this.filteredAttributeTextureFormats,
    );

    if (!chunk.filteredIndexBuffer) {
      chunk.filteredIndexBuffer = new GLBuffer(
        gl,
        WebGL2RenderingContext.ARRAY_BUFFER,
      );
    }
    chunk.filteredIndexBuffer.setData(new Uint32Array(filteredIndices));
    chunk.filteredGeneration = this.generation;
    chunk.filteredMissingConnectionsHash = missingConnectionsHash;
    chunk.filteredNodeLocatorIndexKey = nodeLocatorIndexKey;
    chunk.numFilteredIndices = filteredIndices.length;
    chunk.numFilteredVertices = totalVertexCount;
    chunk.filteredPickNodeIds = filteredPickNodeIds;
    chunk.filteredPickNodePositions = filteredPickNodePositions;
    chunk.filteredPickSegmentIds = filteredPickSegmentIds;
    chunk.filteredPickEdgeSegmentIds = filteredPickEdgeSegmentIds;
    chunk.filteredOldToNew = oldToNew;
    chunk.filteredExtraVertexMap = extraVertexMap;
    chunk.filteredEmpty = false;

    return {
      vertexAttributeTextures: chunk.filteredVertexAttributeTextures,
      indexBuffer: chunk.filteredIndexBuffer,
      numIndices: chunk.numFilteredIndices,
      numVertices: chunk.numFilteredVertices,
      pickNodeIds: filteredPickNodeIds,
      pickNodePositions: filteredPickNodePositions,
      pickSegmentIds: filteredPickSegmentIds,
      pickEdgeSegmentIds: filteredPickEdgeSegmentIds,
    };
  }

  isReady(
    transformedSources?: readonly TransformedSource[][],
    projectionParameters?: any,
    lod?: number | undefined,
  ) {
    if (
      transformedSources === undefined ||
      projectionParameters === undefined
    ) {
      return this.displayState.objectAlpha.value <= 0.0;
    }
    return this.areVisibleChunksReady(
      transformedSources,
      projectionParameters,
      lod,
    );
  }
}

export class PerspectiveViewSpatiallyIndexedSkeletonLayer extends PerspectiveViewRenderLayer {
  private renderHelper: RenderHelper;
  private browseRenderHelper: RenderHelper;
  private renderOptions: ViewSpecificSkeletonRenderingOptions;
  private transformedSources: TransformedSource[][] = [];
  backend: ChunkRenderLayerFrontend;

  constructor(public base: SpatiallyIndexedSkeletonLayer) {
    super();
    this.backend = base.backend;
    this.renderHelper = this.registerDisposer(new RenderHelper(base, false));
    this.browseRenderHelper = this.registerDisposer(
      new RenderHelper(base.chunkGeometryRenderLayerInterface, false),
    );
    this.renderOptions = base.displayState.skeletonRenderingOptions.params3d;

    this.layerChunkProgressInfo = base.layerChunkProgressInfo;
    this.registerDisposer(base);
    this.registerDisposer(base.redrawNeeded.add(this.redrawNeeded.dispatch));
    const { renderOptions } = this;
    this.registerDisposer(
      renderOptions.mode.changed.add(this.redrawNeeded.dispatch),
    );
    this.registerDisposer(
      renderOptions.lineWidth.changed.add(this.redrawNeeded.dispatch),
    );
    const histogram = (base.displayState as any)
      .spatialSkeletonGridRenderScaleHistogram3d as
      | RenderScaleHistogram
      | undefined;
    if (histogram !== undefined) {
      this.registerDisposer(histogram.visibility.add(this.visibility));
    }
  }

  attach(
    attachment: VisibleLayerInfo<
      PerspectivePanel,
      ThreeDimensionalRenderLayerAttachmentState
    >,
  ) {
    super.attach(attachment);

    // Manually add layer to backend
    const backend = this.backend;
    if (backend && backend.rpc) {
      backend.rpc.invoke(RENDERED_VIEW_ADD_LAYER_RPC_ID, {
        layer: backend.rpcId,
        view: attachment.view.rpcId,
      });
    }

    // Capture references to avoid losing 'this' context in callback
    const baseLayer = this.base;
    const redrawNeeded = this.redrawNeeded;

    attachment.registerDisposer(
      registerNested(
        (context, transform, displayDimensionRenderInfo) => {
          const transformedSources = getVolumetricTransformedSources(
            displayDimensionRenderInfo,
            transform,
            () => [
              baseLayer.getSources("3d").map((sourceEntry) => ({
                chunkSource: sourceEntry.chunkSource,
                chunkToMultiscaleTransform:
                  sourceEntry.chunkToMultiscaleTransform,
              })),
            ],
            attachment.messages,
            this,
          );
          for (const scales of transformedSources) {
            for (const tsource of scales) {
              context.registerDisposer(tsource.source);
            }
          }
          attachment.view.flushBackendProjectionParameters();
          this.transformedSources = transformedSources;
          baseLayer.rpc!.invoke(
            SPATIALLY_INDEXED_SKELETON_RENDER_LAYER_UPDATE_SOURCES_RPC_ID,
            {
              layer: baseLayer.backend.rpcId,
              view: attachment.view.rpcId,
              displayDimensionRenderInfo,
              sources: serializeAllTransformedSources(transformedSources),
            },
          );
          redrawNeeded.dispatch();
          return transformedSources;
        },
        baseLayer.displayState.transform,
        attachment.view.displayDimensionRenderInfo,
      ),
    );
  }

  get gl() {
    return this.base.gl;
  }

  get isTransparent() {
    return this.base.displayState.objectAlpha.value < 1.0;
  }

  getValueAt(position: Float32Array) {
    position;
    return undefined;
  }

  transformPickedValue(pickState: PickState) {
    const pickedSegmentId = pickState.pickedSpatialSkeletonSegmentId;
    if (
      typeof pickedSegmentId === "number" &&
      Number.isSafeInteger(pickedSegmentId)
    ) {
      return BigInt(pickedSegmentId);
    }
    return undefined;
  }

  updateMouseState(
    mouseState: MouseSelectionState,
    _pickedValue: bigint,
    pickedOffset: number,
    data: any,
  ) {
    const pickData = data as SpatiallyIndexedSkeletonPickData | undefined;
    if (pickData === undefined) return;
    if (pickData.kind === "node") {
      if (
        pickedOffset < 0 ||
        pickedOffset >= pickData.nodeIds.length ||
        pickedOffset >= pickData.segmentIds.length
      ) {
        return;
      }
      const nodeId = pickData.nodeIds[pickedOffset];
      if (!Number.isSafeInteger(nodeId) || nodeId <= 0) return;
      mouseState.pickedSpatialSkeletonNodeId = nodeId;
      snapMouseStateToSpatialSkeletonNode(
        mouseState,
        pickData.nodePositions,
        pickedOffset,
      );
      const segmentId = pickData.segmentIds[pickedOffset];
      if (Number.isSafeInteger(segmentId)) {
        mouseState.pickedSpatialSkeletonSegmentId = segmentId;
      }
      return;
    }
    if (pickData.kind === "edge") {
      if (pickedOffset < 0 || pickedOffset >= pickData.segmentIds.length) {
        return;
      }
      const segmentId = pickData.segmentIds[pickedOffset];
      if (Number.isSafeInteger(segmentId) && segmentId > 0) {
        mouseState.pickedSpatialSkeletonSegmentId = segmentId;
      }
      return;
    }
    if (pickData.kind === "segment-node" || pickData.kind === "segment-edge") {
      const segmentId = this.base.resolveSegmentPickFromChunk(
        pickData.chunk,
        pickedOffset,
        pickData.kind === "segment-node" ? "node" : "edge",
      );
      if (segmentId !== undefined) {
        mouseState.pickedSpatialSkeletonSegmentId = segmentId;
      }
    }
  }

  draw(
    renderContext: PerspectiveViewRenderContext,
    attachment: VisibleLayerInfo<
      PerspectivePanel,
      ThreeDimensionalRenderLayerAttachmentState
    >,
  ) {
    const pixelSizeWatchable = (this.base.displayState as any)
      .spatialSkeletonGridPixelSize3d as
      | WatchableValueInterface<number>
      | undefined;
    if (pixelSizeWatchable !== undefined) {
      const voxelPhysicalScales =
        renderContext.projectionParameters.displayDimensionRenderInfo
          ?.voxelPhysicalScales;
      if (voxelPhysicalScales !== undefined) {
        const { invViewMatrix } = renderContext.projectionParameters;
        let computedPixelSize = 0;
        for (let i = 0; i < 3; ++i) {
          const s = voxelPhysicalScales[i];
          const x = invViewMatrix[i];
          computedPixelSize += (s * x) ** 2;
        }
        const pixelSize = Math.sqrt(computedPixelSize);
        if (
          Number.isFinite(pixelSize) &&
          pixelSizeWatchable.value !== pixelSize
        ) {
          pixelSizeWatchable.value = pixelSize;
        }
      }
    }
    if (!renderContext.emitColor && renderContext.alreadyEmittedPickID) {
      return;
    }
    const displayState = this.base.displayState as any;
    const lodValue = displayState.skeletonLod?.value as number | undefined;
    this.base.updateVisibleChunksForView(
      "3d",
      this.transformedSources,
      renderContext.projectionParameters,
      lodValue,
    );
    const levels = displayState.spatialSkeletonGridLevels?.value as
      | Array<{ size: { x: number; y: number; z: number } }>
      | undefined;
    const histogram = displayState.spatialSkeletonGridRenderScaleHistogram3d as
      | RenderScaleHistogram
      | undefined;
    if (histogram !== undefined) {
      const frameNumber =
        this.base.chunkManager.chunkQueueManager.frameNumberCounter.frameNumber;
      const relative =
        displayState.spatialSkeletonGridResolutionRelative3d?.value === true;
      const pixelSize = Math.max(pixelSizeWatchable?.value ?? 1, 1e-6);
      updateSpatialSkeletonGridRenderScaleHistogram(
        histogram,
        frameNumber,
        this.transformedSources,
        renderContext.projectionParameters,
        this.base.localPosition.value,
        lodValue,
        levels,
        relative,
        pixelSize,
      );
    }
    this.base.draw(
      renderContext,
      this,
      this.renderHelper,
      this.browseRenderHelper,
      this.renderOptions,
      attachment,
      {
        view: "3d",
        gridLevel: displayState.spatialSkeletonGridLevel3d?.value as
          | number
          | undefined,
        lod: lodValue,
      },
    );
  }

  isReady(
    renderContext: PerspectiveViewReadyRenderContext,
    _attachment: VisibleLayerInfo<
      PerspectivePanel,
      ThreeDimensionalRenderLayerAttachmentState
    >,
  ) {
    const displayState = this.base.displayState as any;
    const lodValue = displayState.skeletonLod?.value as number | undefined;
    return this.base.isReady(
      this.transformedSources,
      renderContext.projectionParameters,
      lodValue,
    );
  }
}

export class SliceViewSpatiallyIndexedSkeletonLayer extends SliceViewRenderLayer {
  private renderOptions: ViewSpecificSkeletonRenderingOptions;
  constructor(public base: SpatiallyIndexedSkeletonLayer) {
    super(
      base.chunkManager,
      {
        getSources: () => {
          return [
            [
              {
                chunkSource: base.source,
                chunkToMultiscaleTransform: mat4.create(),
              },
            ],
          ];
        },
      } as any,
      {
        transform: base.displayState.transform,
        localPosition: (base.displayState as any).localPosition,
      },
    );
    // @ts-ignore
    this.renderHelper = this.registerDisposer(new RenderHelper(base, true));
    this.renderOptions = base.displayState.skeletonRenderingOptions.params2d;
    this.layerChunkProgressInfo = base.layerChunkProgressInfo;
    this.registerDisposer(base);
    const { renderOptions } = this;
    this.registerDisposer(
      renderOptions.mode.changed.add(this.redrawNeeded.dispatch),
    );
    this.registerDisposer(
      renderOptions.lineWidth.changed.add(this.redrawNeeded.dispatch),
    );
    this.registerDisposer(base.redrawNeeded.add(this.redrawNeeded.dispatch));
    this.initializeCounterpart();
  }
  get gl() {
    return this.base.gl;
  }

  getValueAt(position: Float32Array) {
    position;
    return undefined;
  }

  draw(renderContext: SliceViewRenderContext) {
    renderContext;
    // No-op for now to test data loading.
    // SliceViewRenderLayer draw is abstract, so we must implement it.
  }
}

export class SliceViewPanelSpatiallyIndexedSkeletonLayer extends SliceViewPanelRenderLayer {
  private renderHelper: RenderHelper;
  private browseRenderHelper: RenderHelper;
  private renderOptions: ViewSpecificSkeletonRenderingOptions;
  private transformedSources: TransformedSource[][] = [];
  constructor(public base: SpatiallyIndexedSkeletonLayer) {
    super();
    this.renderHelper = this.registerDisposer(new RenderHelper(base, true));
    this.browseRenderHelper = this.registerDisposer(
      new RenderHelper(base.chunkGeometryRenderLayerInterface, true),
    );
    this.renderOptions = base.displayState.skeletonRenderingOptions.params2d;
    this.layerChunkProgressInfo = base.layerChunkProgressInfo;
    this.registerDisposer(base);
    const { renderOptions } = this;
    this.registerDisposer(
      renderOptions.mode.changed.add(this.redrawNeeded.dispatch),
    );
    this.registerDisposer(
      renderOptions.lineWidth.changed.add(this.redrawNeeded.dispatch),
    );
    const gridLevel2d = (base.displayState as any).spatialSkeletonGridLevel2d;
    if (gridLevel2d?.changed) {
      this.registerDisposer(
        gridLevel2d.changed.add(this.redrawNeeded.dispatch),
      );
    }
    const lod2d = (base.displayState as any).spatialSkeletonLod2d;
    if (lod2d?.changed) {
      this.registerDisposer(lod2d.changed.add(this.redrawNeeded.dispatch));
    }
    const histogram = (base.displayState as any)
      .spatialSkeletonGridRenderScaleHistogram2d as
      | RenderScaleHistogram
      | undefined;
    if (histogram !== undefined) {
      this.registerDisposer(histogram.visibility.add(this.visibility));
    }
    this.registerDisposer(base.redrawNeeded.add(this.redrawNeeded.dispatch));
  }
  get gl() {
    return this.base.gl;
  }

  getValueAt(position: Float32Array) {
    position;
    return undefined;
  }

  transformPickedValue(pickState: PickState) {
    const pickedSegmentId = pickState.pickedSpatialSkeletonSegmentId;
    if (
      typeof pickedSegmentId === "number" &&
      Number.isSafeInteger(pickedSegmentId)
    ) {
      return BigInt(pickedSegmentId);
    }
    return undefined;
  }

  updateMouseState(
    mouseState: MouseSelectionState,
    _pickedValue: bigint,
    pickedOffset: number,
    data: any,
  ) {
    const pickData = data as SpatiallyIndexedSkeletonPickData | undefined;
    if (pickData === undefined) return;
    if (pickData.kind === "node") {
      if (
        pickedOffset < 0 ||
        pickedOffset >= pickData.nodeIds.length ||
        pickedOffset >= pickData.segmentIds.length
      ) {
        return;
      }
      const nodeId = pickData.nodeIds[pickedOffset];
      if (!Number.isSafeInteger(nodeId) || nodeId <= 0) return;
      mouseState.pickedSpatialSkeletonNodeId = nodeId;
      snapMouseStateToSpatialSkeletonNode(
        mouseState,
        pickData.nodePositions,
        pickedOffset,
      );
      const segmentId = pickData.segmentIds[pickedOffset];
      if (Number.isSafeInteger(segmentId)) {
        mouseState.pickedSpatialSkeletonSegmentId = segmentId;
      }
      return;
    }
    if (pickData.kind === "edge") {
      if (pickedOffset < 0 || pickedOffset >= pickData.segmentIds.length) {
        return;
      }
      const segmentId = pickData.segmentIds[pickedOffset];
      if (Number.isSafeInteger(segmentId) && segmentId > 0) {
        mouseState.pickedSpatialSkeletonSegmentId = segmentId;
      }
      return;
    }
    if (pickData.kind === "segment-node" || pickData.kind === "segment-edge") {
      const segmentId = this.base.resolveSegmentPickFromChunk(
        pickData.chunk,
        pickedOffset,
        pickData.kind === "segment-node" ? "node" : "edge",
      );
      if (segmentId !== undefined) {
        mouseState.pickedSpatialSkeletonSegmentId = segmentId;
      }
    }
  }

  attach(
    attachment: VisibleLayerInfo<
      SliceViewPanel,
      ThreeDimensionalRenderLayerAttachmentState
    >,
  ) {
    super.attach(attachment);
    const baseLayer = this.base;
    const redrawNeeded = this.redrawNeeded;
    attachment.registerDisposer(
      registerNested(
        (context, transform, displayDimensionRenderInfo) => {
          const transformedSources = getVolumetricTransformedSources(
            displayDimensionRenderInfo,
            transform,
            () => [
              baseLayer.getSources("2d").map((sourceEntry) => ({
                chunkSource: sourceEntry.chunkSource,
                chunkToMultiscaleTransform:
                  sourceEntry.chunkToMultiscaleTransform,
              })),
            ],
            attachment.messages,
            this,
          );
          for (const scales of transformedSources) {
            for (const tsource of scales) {
              context.registerDisposer(tsource.source);
            }
          }
          this.transformedSources = transformedSources;
          redrawNeeded.dispatch();
          return transformedSources;
        },
        baseLayer.displayState.transform,
        attachment.view.displayDimensionRenderInfo,
      ),
    );
  }

  draw(
    renderContext: SliceViewPanelRenderContext,
    attachment: VisibleLayerInfo<
      SliceViewPanel,
      ThreeDimensionalRenderLayerAttachmentState
    >,
  ) {
    const pixelSizeWatchable = (this.base.displayState as any)
      .spatialSkeletonGridPixelSize2d as
      | WatchableValueInterface<number>
      | undefined;
    if (pixelSizeWatchable !== undefined) {
      const pixelSize =
        renderContext.sliceView.projectionParameters.value.pixelSize;
      if (
        Number.isFinite(pixelSize) &&
        pixelSizeWatchable.value !== pixelSize
      ) {
        pixelSizeWatchable.value = pixelSize;
      }
    }
    const displayState = this.base.displayState as any;
    const lodValue = displayState.spatialSkeletonLod2d?.value as
      | number
      | undefined;
    this.base.updateVisibleChunksForView(
      "2d",
      this.transformedSources,
      renderContext.sliceView.projectionParameters.value,
      lodValue,
    );
    const levels = displayState.spatialSkeletonGridLevels?.value as
      | Array<{ size: { x: number; y: number; z: number } }>
      | undefined;
    const histogram = displayState.spatialSkeletonGridRenderScaleHistogram2d as
      | RenderScaleHistogram
      | undefined;
    if (histogram !== undefined) {
      const frameNumber =
        this.base.chunkManager.chunkQueueManager.frameNumberCounter.frameNumber;
      const relative =
        displayState.spatialSkeletonGridResolutionRelative2d?.value === true;
      const pixelSize = Math.max(pixelSizeWatchable?.value ?? 1, 1e-6);
      updateSpatialSkeletonGridRenderScaleHistogram(
        histogram,
        frameNumber,
        this.transformedSources,
        renderContext.sliceView.projectionParameters.value,
        this.base.localPosition.value,
        lodValue,
        levels,
        relative,
        pixelSize,
      );
    }
    this.base.draw(
      renderContext,
      this,
      this.renderHelper,
      this.browseRenderHelper,
      this.renderOptions,
      attachment,
      {
        view: "2d",
        gridLevel: displayState.spatialSkeletonGridLevel2d?.value as
          | number
          | undefined,
        lod: lodValue,
      },
    );
  }

  isReady(
    renderContext: SliceViewPanelReadyRenderContext,
    _attachment: VisibleLayerInfo<
      SliceViewPanel,
      ThreeDimensionalRenderLayerAttachmentState
    >,
  ) {
    const displayState = this.base.displayState as any;
    const lodValue = displayState.spatialSkeletonLod2d?.value as
      | number
      | undefined;
    return this.base.isReady(
      this.transformedSources,
      renderContext.projectionParameters,
      lodValue,
    );
  }
}

const emptyVertexAttributes = new Map<string, VertexAttributeInfo>();

function getAttributeTextureFormats(
  vertexAttributes: Map<string, VertexAttributeInfo>,
): TextureFormat[] {
  const attributeTextureFormats: TextureFormat[] = [
    vertexPositionTextureFormat,
  ];
  for (const info of vertexAttributes.values()) {
    attributeTextureFormats.push(
      computeTextureFormat(
        new TextureFormat(),
        info.dataType,
        info.numComponents,
      ),
    );
  }
  return attributeTextureFormats;
}

export type SkeletonSourceOptions = object;

export class SkeletonSource extends ChunkSource {
  private attributeTextureFormats_?: TextureFormat[];

  get attributeTextureFormats() {
    let attributeTextureFormats = this.attributeTextureFormats_;
    if (attributeTextureFormats === undefined) {
      attributeTextureFormats = this.attributeTextureFormats_ =
        getAttributeTextureFormats(this.vertexAttributes);
    }
    return attributeTextureFormats;
  }

  declare chunks: Map<string, SkeletonChunk>;
  getChunk(x: any) {
    return new SkeletonChunk(this, x);
  }

  get vertexAttributes(): Map<string, VertexAttributeInfo> {
    return emptyVertexAttributes;
  }
}
