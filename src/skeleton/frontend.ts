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

import type { Uint64Set } from "#src/uint64_set.js";
import { uploadVertexAttributesToGPU } from "#src/skeleton/gpu_upload_utils.js";

import { ChunkState, LayerChunkProgressInfo } from "#src/chunk_manager/base.js";
import type { ChunkManager } from "#src/chunk_manager/frontend.js";
import {
  Chunk,
  ChunkRenderLayerFrontend,
  ChunkSource,
} from "#src/chunk_manager/frontend.js";
import type { LayerView, UserLayer, VisibleLayerInfo } from "#src/layer/index.js";
import type { PerspectivePanel } from "#src/perspective_view/panel.js";
import type { PerspectiveViewRenderContext } from "#src/perspective_view/render_layer.js";
import { PerspectiveViewRenderLayer } from "#src/perspective_view/render_layer.js";
import type {
  RenderLayer,
  ThreeDimensionalRenderLayerAttachmentState,
} from "#src/renderlayer.js";
import { update3dRenderLayerAttachment } from "#src/renderlayer.js";
import {
  RenderScaleHistogram,
} from "#src/render_scale_statistics.js";
import {
  forEachVisibleSegment,
  getObjectKey,
  getVisibleSegments,
} from "#src/segmentation_display_state/base.js";
import type { SegmentationDisplayState3D, SegmentationDisplayState } from "#src/segmentation_display_state/frontend.js";
import {
  forEachVisibleSegmentToDraw,
  getBaseObjectColor,
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
import { DataType } from "#src/util/data_type.js";
import { RefCounted } from "#src/util/disposable.js";
import { mat4 } from "#src/util/geom.js";
import { verifyFinitePositiveFloat } from "#src/util/json.js";
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

const DEFAULT_FRAGMENT_MAIN = `void main() {
  emitDefault();
}
`;

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
  DataType.FLOAT32,
  1,
);
const segmentColorTextureFormat = computeTextureFormat(
  new TextureFormat(),
  DataType.FLOAT32,
  4,
);

interface SkeletonLayerInterface {
  vertexAttributes: VertexAttributeRenderInfo[];
  segmentColorAttributeIndex?: number;
  gl: GL;
  fallbackShaderParameters: WatchableValue<ShaderControlsBuilderState>;
  displayState: SkeletonLayerDisplayState;
}

interface SkeletonChunkInterface {
  vertexAttributeTextures: (WebGLTexture | null)[];
  indexBuffer: GLBuffer;
  numIndices: number;
  numVertices: number;
}

interface SkeletonChunkData {
  vertexAttributes: Uint8Array;
  indices: Uint32Array;
  numVertices: number;
  vertexAttributeOffsets: Uint32Array;
}

class RenderHelper extends RefCounted {
  private textureAccessHelper = new OneDimensionalTextureAccessHelper(
    "vertexData",
  );
  private vertexIdHelper;
  private segmentColorAttributeIndex: number | undefined;
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

  constructor(
    public base: SkeletonLayerInterface,
    public targetIsSliceView: boolean,
  ) {
    super();
    this.vertexIdHelper = this.registerDisposer(VertexIdHelper.get(this.gl));
    this.segmentColorAttributeIndex = base.segmentColorAttributeIndex;
    this.edgeShaderGetter = parameterizedEmitterDependentShaderGetter(
      this,
      this.gl,
      {
        memoizeKey: {
          type: "skeleton/SkeletonShaderManager/edge",
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
          defineLineShader(builder);
          builder.addAttribute("highp uvec2", "aVertexIndex");
          builder.addUniform("highp float", "uLineWidth");
          let vertexMain = `
highp vec3 vertexA = readAttribute0(aVertexIndex.x);
highp vec3 vertexB = readAttribute0(aVertexIndex.y);
emitLine(uProjection, vertexA, vertexB, uLineWidth);
highp uint lineEndpointIndex = getLineEndpointIndex();
highp uint vertexIndex = aVertexIndex.x * (1u - lineEndpointIndex) + aVertexIndex.y * lineEndpointIndex;
`;

          const segmentColorExpression = this.getSegmentColorExpression();
          const segmentAlphaExpression =
            this.segmentColorAttributeIndex === undefined
              ? "uColor.a"
              : `${segmentColorExpression}.a`;
          if (this.segmentColorAttributeIndex === undefined) {
            // Preserve legacy skeleton behavior where `uColor` is already
            // premultiplied by `objectAlpha` in `getObjectColor`.
            builder.addFragmentCode(`
vec4 segmentColor() {
  return ${segmentColorExpression};
}
void emitRGB(vec3 color) {
  emit(vec4(color * uColor.a, uColor.a * getLineAlpha() * ${this.getCrossSectionFadeFactor()}), uPickID);
}
void emitDefault() {
  emit(vec4(uColor.rgb, uColor.a * getLineAlpha() * ${this.getCrossSectionFadeFactor()}), uPickID);
}
`);
          } else {
            builder.addFragmentCode(`
vec4 segmentColor() {
  return ${segmentColorExpression};
}
void emitRGB(vec3 color) {
  highp float alpha = ${segmentAlphaExpression} * getLineAlpha() * ${this.getCrossSectionFadeFactor()};
  emit(vec4(color * alpha, alpha), uPickID);
}
void emitDefault() {
  vec4 baseColor = segmentColor();
  highp float alpha = baseColor.a * getLineAlpha() * ${this.getCrossSectionFadeFactor()};
  emit(vec4(baseColor.rgb * alpha, alpha), uPickID);
}
`);
          }
          builder.addFragmentCode(glsl_COLORMAPS);
          const { vertexAttributes } = this;
          const numAttributes = vertexAttributes.length;
          for (let i = 1; i < numAttributes; ++i) {
            const info = vertexAttributes[i];
            builder.addVarying(`highp ${info.glslDataType}`, `vCustom${i}`);
            vertexMain += `vCustom${i} = readAttribute${i}(vertexIndex);\n`;
            builder.addFragmentCode(`#define ${info.name} vCustom${i}\n`);
            builder.addFragmentCode(
              `#define prop_${info.name}() vCustom${i}\n`,
            );
          }
          builder.setVertexMain(vertexMain);
          addControlsToBuilder(shaderBuilderState, builder);
          const edgeFragmentCode = shaderCodeWithLineDirective(shaderBuilderState.parseResult.code);
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
          defineCircleShader(
            builder,
            /*crossSectionFade=*/ this.targetIsSliceView,
          );
          builder.addUniform("highp float", "uNodeDiameter");
          let vertexMain = `
highp uint vertexIndex = uint(gl_InstanceID);
highp vec3 vertexPosition = readAttribute0(vertexIndex);
emitCircle(uProjection * vec4(vertexPosition, 1.0), uNodeDiameter, 0.0);
`;

          const segmentColorExpression = this.getSegmentColorExpression();
          if (this.segmentColorAttributeIndex === undefined) {
            // Preserve legacy skeleton behavior for non-spatial skeletons.
            builder.addFragmentCode(`
vec4 segmentColor() {
  return ${segmentColorExpression};
}
void emitRGBA(vec4 color) {
  vec4 borderColor = color;
  emit(getCircleColor(color, borderColor), uPickID);
}
void emitRGB(vec3 color) {
  emitRGBA(vec4(color, 1.0));
}
void emitDefault() {
  emitRGBA(uColor);
}
`);
          } else {
            builder.addFragmentCode(`
vec4 segmentColor() {
  return ${segmentColorExpression};
}
void emitRGBA(vec4 color) {
  vec4 borderColor = color;
  vec4 circleColor = getCircleColor(color, borderColor);
  emit(vec4(circleColor.rgb * circleColor.a, circleColor.a), uPickID);
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
            builder.addVarying(`highp ${info.glslDataType}`, `vCustom${i}`);
            vertexMain += `vCustom${i} = readAttribute${i}(vertexIndex);\n`;
            builder.addFragmentCode(`#define ${info.name} vCustom${i}\n`);
            builder.addFragmentCode(
              `#define prop_${info.name}() vCustom${i}\n`,
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
    const a = (color as Float32Array).length >= 4
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

  drawSkeleton(
    gl: GL,
    edgeShader: ShaderProgram,
    nodeShader: ShaderProgram | null,
    skeletonChunk: SkeletonChunkInterface,
    projectionParameters: { width: number; height: number },
  ) {
    const { vertexAttributes } = this;
    const numAttributes = vertexAttributes.length;
    const { vertexAttributeTextures } = skeletonChunk;
    for (let i = 0; i < numAttributes; ++i) {
      const textureUnit =
        WebGL2RenderingContext.TEXTURE0 +
        edgeShader.textureUnit(vertexAttributeSamplerSymbols[i]);
      gl.activeTexture(textureUnit);
      gl.bindTexture(
        WebGL2RenderingContext.TEXTURE_2D,
        vertexAttributeTextures[i],
      );
    }

    // Draw edges
    {
      edgeShader.bind();
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
      initializeCircleShader(nodeShader, projectionParameters, {
        featherWidthInPixels: this.targetIsSliceView ? 1.0 : 0.0,
      });
      drawCircles(nodeShader.gl, 2, skeletonChunk.numVertices);
    }
  }

  endLayer(gl: GL, shader: ShaderProgram) {
    const { vertexAttributes } = this;
    const numAttributes = vertexAttributes.length;
    for (let i = 0; i < numAttributes; ++i) {
      const curTextureUnit =
        shader.textureUnit(vertexAttributeSamplerSymbols[i]) +
        WebGL2RenderingContext.TEXTURE0;
      gl.activeTexture(curTextureUnit);
      gl.bindTexture(gl.TEXTURE_2D, null);
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
        glslDataType:
          info.numComponents > 1 ? `vec${info.numComponents}` : "float",
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
    let pointDiameter: number;
    if (renderOptions.mode.value === SkeletonRenderMode.LINES_AND_POINTS) {
      pointDiameter = Math.max(5, lineWidth * 2);
    } else {
      pointDiameter = lineWidth;
    }

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
    renderHelper.endLayer(gl, edgeShader);
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
    default:
      throw new Error(
        `Data type not supported by WebGL: ${DataType[dataType]}`,
      );
  }
}

const vertexPositionAttribute: VertexAttributeRenderInfo = {
  dataType: DataType.FLOAT32,
  numComponents: 3,
  name: "",
  webglDataType: WebGL2RenderingContext.FLOAT,
  glslDataType: "vec3",
};

const segmentAttribute: VertexAttributeRenderInfo = {
  dataType: DataType.FLOAT32,
  numComponents: 1,
  name: "segment",
  webglDataType: WebGL2RenderingContext.FLOAT,
  glslDataType: "float",
};

const segmentColorAttribute: VertexAttributeRenderInfo = {
  dataType: DataType.FLOAT32,
  numComponents: 4,
  name: "segmentColorAttr",
  webglDataType: WebGL2RenderingContext.FLOAT,
  glslDataType: "vec4",
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

export class SpatiallyIndexedSkeletonChunk extends SliceViewChunk implements SkeletonChunkInterface {
  declare source: SpatiallyIndexedSkeletonSource;
  vertexAttributes: Uint8Array;
  indices: Uint32Array;
  indexBuffer: GLBuffer;
  numIndices: number;
  numVertices: number;
  vertexAttributeOffsets: Uint32Array;
  vertexAttributeTextures: (WebGLTexture | null)[];
  missingConnections: Array<{ nodeId: number; parentId: number; vertexIndex: number; skeletonId: number }> = [];
  nodeMap: Map<number, number> = new Map(); // Maps node ID to vertex index
  lod: number | undefined;

  // Filtering support
  filteredIndexBuffer: GLBuffer | undefined;
  filteredGeneration: number = -1;
  filteredSkipVisibleSegments = false;
  filteredMissingConnectionsHash = 0;
  numFilteredIndices: number = 0;
  numFilteredVertices: number = 0;
  filteredVertexAttributeTextures?: (WebGLTexture | null)[];

  constructor(source: SpatiallyIndexedSkeletonSource, chunkData: SkeletonChunkData) {
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

  freeGPUMemory(gl: GL) {
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
  }
}

export interface SpatiallyIndexedSkeletonChunkSpecification extends SliceViewChunkSpecification {
  chunkLayout: ChunkLayout;
}

export class SpatiallyIndexedSkeletonSource extends SliceViewChunkSource<
  SpatiallyIndexedSkeletonChunkSpecification,
  SpatiallyIndexedSkeletonChunk
> {
  vertexAttributes: VertexAttributeRenderInfo[];
  private attributeTextureFormats_?: TextureFormat[];

  constructor(chunkManager: ChunkManager, options: any) {
    super(chunkManager, options);
    this.vertexAttributes = [vertexPositionAttribute, segmentAttribute];
  }

  get attributeTextureFormats() {
    let attributeTextureFormats = this.attributeTextureFormats_;
    if (attributeTextureFormats === undefined) {
      attributeTextureFormats = this.attributeTextureFormats_ =
        getAttributeTextureFormats(
          new Map([
            ["", vertexPositionAttribute],
            ["segment", segmentAttribute],
          ]),
        );
    }
    return attributeTextureFormats;
  }

  static encodeSpec(spec: SpatiallyIndexedSkeletonChunkSpecification) {
    const base = SliceViewChunkSource.encodeSpec(spec);
    return { ...base, chunkLayout: spec.chunkLayout.toObject() };
  }

  getChunk(chunkData: SkeletonChunkData) {
    return new SpatiallyIndexedSkeletonChunk(this, chunkData);
  }
}

export abstract class MultiscaleSpatiallyIndexedSkeletonSource extends MultiscaleSliceViewChunkSource<SpatiallyIndexedSkeletonSource> {
  getPerspectiveSources(): SliceViewSingleResolutionSource<SpatiallyIndexedSkeletonSource>[] {
    const sources = this.getSources({ view: "3d" } as any);
    const flattened: SliceViewSingleResolutionSource<SpatiallyIndexedSkeletonSource>[] = [];
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

  getSpatialSkeletonGridSizes(): { x: number; y: number; z: number }[] | undefined {
    return undefined;
  }
}

export class MultiscaleSliceViewSpatiallyIndexedSkeletonLayer extends SliceViewRenderLayer<
  SpatiallyIndexedSkeletonSource
> {
  private renderOptions: ViewSpecificSkeletonRenderingOptions;
  RPC_TYPE_ID = SPATIALLY_INDEXED_SKELETON_SLICEVIEW_RENDER_LAYER_RPC_ID;
  constructor(
    public chunkManager: ChunkManager,
    public multiscaleSource: MultiscaleSpatiallyIndexedSkeletonSource,
    public displayState: SegmentationDisplayState
  ) {
    const renderScaleTarget = (displayState as any)
      .renderScaleTarget as WatchableValueInterface<number>;
    const gridLevel2d = (displayState as any).spatialSkeletonGridLevel2d as
      | WatchableValueInterface<number>
      | undefined;
    const gridAwareRenderScaleTarget = new WatchableValue(
      renderScaleTarget.value,
    );
    super(chunkManager, multiscaleSource, {
      transform: (displayState as any).transform,
      localPosition: (displayState as any).localPosition,
      renderScaleTarget: gridAwareRenderScaleTarget,
    });
    this.renderOptions = (displayState as any).skeletonRenderingOptions.params2d;
    this.registerDisposer(
      this.renderOptions.mode.changed.add(
        this.redrawNeeded.dispatch,
      ),
    );
    this.registerDisposer(
      this.renderOptions.lineWidth.changed.add(this.redrawNeeded.dispatch),
    );
    this.registerDisposer(
      renderScaleTarget.changed.add(() => {
        gridAwareRenderScaleTarget.value = renderScaleTarget.value;
      }),
    );
    if (gridLevel2d !== undefined) {
      let gridInvalidationTick = 0;
      this.registerDisposer(
        gridLevel2d.changed.add(() => {
          // Toggle a tiny offset so backend/front-end visible source caches are invalidated
          // even when only the grid index changes.
          const baseValue = renderScaleTarget.value;
          gridInvalidationTick ^= 1;
          gridAwareRenderScaleTarget.value =
            baseValue + gridInvalidationTick * 1e-6;
        }),
      );
    }
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
    if (gridLevel !== undefined && sources.length > 0) {
      const gridIndexedSources = sources
        .map((tsource, scaleIndex) => ({
          tsource,
          scaleIndex,
          gridIndex: (tsource.source as any).parameters?.gridIndex as
            | number
            | undefined,
        }))
        .filter((entry) => entry.gridIndex !== undefined);
      if (gridIndexedSources.length === sources.length) {
        let minGridIndex = Number.POSITIVE_INFINITY;
        let maxGridIndex = Number.NEGATIVE_INFINITY;
        for (const entry of gridIndexedSources) {
          const index = entry.gridIndex as number;
          minGridIndex = Math.min(minGridIndex, index);
          maxGridIndex = Math.max(maxGridIndex, index);
        }
        const clampedGridLevel = Math.min(
          Math.max(gridLevel, minGridIndex),
          maxGridIndex,
        );
        let selectedEntry =
          gridIndexedSources.find(
            (entry) => entry.gridIndex === clampedGridLevel,
          ) ?? gridIndexedSources[0];
        if (selectedEntry.gridIndex !== clampedGridLevel) {
          let bestDistance = Number.POSITIVE_INFINITY;
          for (const entry of gridIndexedSources) {
            const distance = Math.abs(
              (entry.gridIndex as number) - clampedGridLevel,
            );
            if (distance < bestDistance) {
              bestDistance = distance;
              selectedEntry = entry;
            }
          }
        }
        return [selectedEntry.tsource];
      }
    }
    return super.filterVisibleSources(sliceView, sources);
  }

  // Draw is no-op as per SliceViewSpatiallyIndexedSkeletonLayer pattern
  draw(renderContext: SliceViewRenderContext) {
    renderContext;
  }
}

type SpatiallyIndexedSkeletonSourceEntry =
  SliceViewSingleResolutionSource<SpatiallyIndexedSkeletonSource>;

interface SpatiallyIndexedGlobalNodeLookupEntry {
  chunk: SpatiallyIndexedSkeletonChunk;
  vertexIndex: number;
}

interface SpatiallyIndexedSkeletonLayerOptions {
  gridLevel?: WatchableValueInterface<number>;
  lod?: WatchableValueInterface<number>;
  sources2d?: SpatiallyIndexedSkeletonSourceEntry[];
}

type SpatiallyIndexedSkeletonView = "2d" | "3d";

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

export class SpatiallyIndexedSkeletonLayer extends RefCounted implements SkeletonLayerInterface {
  layerChunkProgressInfo = new LayerChunkProgressInfo();
  redrawNeeded = new NullarySignal();
  vertexAttributes: VertexAttributeRenderInfo[];
  segmentColorAttributeIndex: number | undefined;
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
    segmentColorTextureFormat,
  ];
  private regularSkeletonLayerWatchable = new WatchableValue(false);
  private regularSkeletonLayerUserLayer: UserLayer | undefined;
  private removeRegularSkeletonLayerUserLayerListener:
    | (() => boolean)
    | undefined;
  private cachedGlobalNodeLookupKey: string | undefined;
  private cachedGlobalNodeLookupChunkSignatures = new Map<
    SpatiallyIndexedSkeletonChunk,
    string
  >();
  private cachedGlobalNodeLookup = new Map<
    number,
    SpatiallyIndexedGlobalNodeLookupEntry
  >();
  gridLevel: WatchableValueInterface<number>;
  lod: WatchableValueInterface<number>;

  private markFilteredDataDirty() {
    this.generation++;
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

  private updateHasRegularSkeletonLayerWatchable(userLayer: UserLayer | undefined) {
    if (this.regularSkeletonLayerUserLayer !== userLayer) {
      this.removeRegularSkeletonLayerUserLayerListener?.();
      this.removeRegularSkeletonLayerUserLayerListener = undefined;
      this.regularSkeletonLayerUserLayer = userLayer;
      if (userLayer !== undefined) {
        const update = () => {
          const nextValue = this.computeHasRegularSkeletonLayer(userLayer);
          if (this.regularSkeletonLayerWatchable.value !== nextValue) {
            this.regularSkeletonLayerWatchable.value = nextValue;
            this.markFilteredDataDirty();
          }
        };
        update();
        this.removeRegularSkeletonLayerUserLayerListener =
          userLayer.layersChanged.add(update);
      } else if (this.regularSkeletonLayerWatchable.value) {
        this.regularSkeletonLayerWatchable.value = false;
        this.markFilteredDataDirty();
      }
    }
    return this.regularSkeletonLayerWatchable.value;
  }

  private lodMatches(chunk: SpatiallyIndexedSkeletonChunk, targetLod: number | undefined) {
    if (targetLod === undefined || chunk.lod === undefined) {
      return true;
    }
    return Math.abs(chunk.lod - targetLod) < 1e-6;
  }

  private makeGlobalNodeLookupCacheKey(
    selectedSources: SpatiallyIndexedSkeletonSourceEntry[],
    targetLod: number | undefined,
  ) {
    const sourceKey = selectedSources
      .map((entry) => getObjectId(entry.chunkSource))
      .join(",");
    const lodKey =
      targetLod === undefined ? "none" : Math.round(targetLod * 1e6) / 1e6;
    return `${lodKey}|${sourceKey}`;
  }

  private computeGlobalNodeLookupChunkSignatures(
    selectedSources: SpatiallyIndexedSkeletonSourceEntry[],
    targetLod: number | undefined,
  ) {
    const signatures = new Map<SpatiallyIndexedSkeletonChunk, string>();
    for (const sourceEntry of selectedSources) {
      const chunks = sourceEntry.chunkSource.chunks;
      for (const chunk of chunks.values()) {
        const typedChunk = chunk as SpatiallyIndexedSkeletonChunk;
        if (!this.lodMatches(typedChunk, targetLod)) continue;
        if (typedChunk.state !== ChunkState.GPU_MEMORY) continue;
        const nodeMap = typedChunk.nodeMap;
        const signature = `${getObjectId(nodeMap)}:${nodeMap.size}`;
        signatures.set(typedChunk, signature);
      }
    }
    return signatures;
  }

  private chunkSignaturesEqual(
    nextSignatures: Map<SpatiallyIndexedSkeletonChunk, string>,
  ) {
    const prevSignatures = this.cachedGlobalNodeLookupChunkSignatures;
    if (prevSignatures.size !== nextSignatures.size) return false;
    for (const [chunk, signature] of nextSignatures) {
      if (prevSignatures.get(chunk) !== signature) return false;
    }
    return true;
  }

  private rebuildGlobalNodeLookup(
    selectedSources: SpatiallyIndexedSkeletonSourceEntry[],
    targetLod: number | undefined,
  ) {
    this.cachedGlobalNodeLookup.clear();
    for (const sourceEntry of selectedSources) {
      const chunks = sourceEntry.chunkSource.chunks;
      for (const chunk of chunks.values()) {
        const typedChunk = chunk as SpatiallyIndexedSkeletonChunk;
        if (!this.lodMatches(typedChunk, targetLod)) continue;
        if (typedChunk.state !== ChunkState.GPU_MEMORY) continue;
        for (const [nodeId, vertexIndex] of typedChunk.nodeMap.entries()) {
          this.cachedGlobalNodeLookup.set(nodeId, {
            chunk: typedChunk,
            vertexIndex,
          });
        }
      }
    }
  }

  private getGlobalNodeLookup(
    selectedSources: SpatiallyIndexedSkeletonSourceEntry[],
    targetLod: number | undefined,
  ) {
    const key = this.makeGlobalNodeLookupCacheKey(selectedSources, targetLod);
    const nextSignatures = this.computeGlobalNodeLookupChunkSignatures(
      selectedSources,
      targetLod,
    );
    const shouldRebuild =
      this.cachedGlobalNodeLookupKey !== key ||
      !this.chunkSignaturesEqual(nextSignatures);
    if (shouldRebuild) {
      this.cachedGlobalNodeLookupKey = key;
      this.cachedGlobalNodeLookupChunkSignatures = nextSignatures;
      this.rebuildGlobalNodeLookup(selectedSources, targetLod);
    }
    return this.cachedGlobalNodeLookup;
  }

  get visibility() {
    return this.displayState.objectAlpha;
  }

  sources: SpatiallyIndexedSkeletonSourceEntry[];
  sources2d: SpatiallyIndexedSkeletonSourceEntry[];
  source: SpatiallyIndexedSkeletonSource;

  constructor(
    public chunkManager: ChunkManager,
    sources: SpatiallyIndexedSkeletonSourceEntry[] | SpatiallyIndexedSkeletonSource,
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
      this.cachedGlobalNodeLookup.clear();
      this.cachedGlobalNodeLookupChunkSignatures.clear();
      this.cachedGlobalNodeLookupKey = undefined;
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
      throw new Error("SpatiallyIndexedSkeletonLayer requires at least one source.");
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
      segmentColorAttribute,
    ];
    this.segmentColorAttributeIndex = this.vertexAttributes.length - 1;
    const markDirty = () => this.markFilteredDataDirty();
    // Monitor visible segment changes to update filtered buffers.
    this.registerDisposer(
      registerNested((context, segmentationGroup) => {
        context.registerDisposer(
          segmentationGroup.visibleSegments.changed.add(() =>
            markDirty(),
          ),
        );
        context.registerDisposer(
          segmentationGroup.temporaryVisibleSegments.changed.add(() =>
            markDirty(),
          ),
        );
        context.registerDisposer(
          segmentationGroup.useTemporaryVisibleSegments.changed.add(() =>
            markDirty(),
          ),
        );
      }, this.displayState.segmentationGroupState),
    );
    // Monitor segment color changes to update filtered buffers.
    this.registerDisposer(
      registerNested((context, colorGroupState) => {
        context.registerDisposer(
          colorGroupState.segmentColorHash.changed.add(() =>
            markDirty(),
          ),
        );
        context.registerDisposer(
          colorGroupState.segmentDefaultColor.changed.add(() =>
            markDirty(),
          ),
        );
        context.registerDisposer(
          colorGroupState.segmentStatedColors.changed.add(() =>
            markDirty(),
          ),
        );
      }, this.displayState.segmentationColorGroupState),
    );
    this.registerDisposer(
      displayState.objectAlpha.changed.add(() =>
        markDirty(),
      ),
    );
    if (this.gridLevel !== undefined) {
      this.registerDisposer(
        this.gridLevel.changed.add(() =>
          markDirty(),
        ),
      );
    }
    if (displayState.hiddenObjectAlpha) {
      this.registerDisposer(
        displayState.hiddenObjectAlpha.changed.add(() =>
          markDirty(),
        ),
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
      SharedWatchableValue.makeFromExisting(rpc, displayState.renderScaleTarget),
    );
    
    const skeletonLodWatchable = this.registerDisposer(
      SharedWatchableValue.makeFromExisting(rpc, this.lod),
    );

    const skeletonGridLevelWatchable = this.registerDisposer(
      SharedWatchableValue.makeFromExisting(
        rpc,
        this.gridLevel,
      ),
    );
    
    sharedObject.initializeCounterpart(rpc, {
      chunkManager: chunkManager.rpcId,
      localPosition: this.registerDisposer(
        SharedWatchableValue.makeFromExisting(
          rpc,
          this.localPosition,
        ),
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

  private selectSourcesForViewAndGrid(
    view: SpatiallyIndexedSkeletonView,
    gridLevel: number | undefined,
  ) {
    const sources = this.getSources(view);
    if (sources.length === 0) return sources;
    const viewFiltered = sources.filter((entry) => {
      const params = (entry.chunkSource as any).parameters;
      const sourceView = params?.view as string | undefined;
      return sourceView === undefined || sourceView === view;
    });
    if (gridLevel === undefined || viewFiltered.length === 0) {
      return viewFiltered;
    }
    const gridIndexedSources = viewFiltered
      .map((entry) => ({
        entry,
        gridIndex: (entry.chunkSource as any).parameters?.gridIndex as
          | number
          | undefined,
      }))
      .filter((entry) => entry.gridIndex !== undefined);
    if (gridIndexedSources.length !== viewFiltered.length) {
      return viewFiltered;
    }
    let minGridIndex = Number.POSITIVE_INFINITY;
    let maxGridIndex = Number.NEGATIVE_INFINITY;
    for (const entry of gridIndexedSources) {
      const index = entry.gridIndex as number;
      minGridIndex = Math.min(minGridIndex, index);
      maxGridIndex = Math.max(maxGridIndex, index);
    }
    const clampedGridLevel = Math.min(
      Math.max(gridLevel, minGridIndex),
      maxGridIndex,
    );
    let selectedEntry =
      gridIndexedSources.find(
        (entry) => entry.gridIndex === clampedGridLevel,
      ) ?? gridIndexedSources[0];
    if (selectedEntry.gridIndex !== clampedGridLevel) {
      let bestDistance = Number.POSITIVE_INFINITY;
      for (const entry of gridIndexedSources) {
        const distance = Math.abs(
          (entry.gridIndex as number) - clampedGridLevel,
        );
        if (distance < bestDistance) {
          bestDistance = distance;
          selectedEntry = entry;
        }
      }
    }
    return [selectedEntry.entry];
  }

  draw(
    renderContext: SliceViewPanelRenderContext | PerspectiveViewRenderContext,
    _layer: RenderLayer,
    renderHelper: RenderHelper,
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
      _layer.userLayer,
    );
    let pointDiameter: number;
    if (renderOptions.mode.value === SkeletonRenderMode.LINES_AND_POINTS) {
      pointDiameter = Math.max(5, lineWidth * 2);
    } else {
      pointDiameter = lineWidth;
    }

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
      return;
    }

    const { shaderControlState } = this.displayState.skeletonRenderingOptions;

    edgeShader.bind();
    renderHelper.beginLayer(gl, edgeShader, renderContext, modelMatrix);
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
    setControlsInShader(
      gl,
      nodeShader,
      shaderControlState,
      nodeShaderParameters.parseResult.controls,
    );

    const visibleSegments = getVisibleSegments(
      displayState.segmentationGroupState.value,
    );
    const baseColor = new Float32Array([1, 1, 1, 1]);
    edgeShader.bind();
    renderHelper.setColor(gl, edgeShader, baseColor);
    nodeShader.bind();
    renderHelper.setColor(gl, nodeShader, baseColor);

    const targetLod = drawOptions?.lod;
    const view = drawOptions?.view ?? "3d";
    const selectedSources = this.selectSourcesForViewAndGrid(
      view,
      drawOptions?.gridLevel,
    );
    const globalNodeMap = this.getGlobalNodeLookup(selectedSources, targetLod);
    for (const sourceEntry of selectedSources) {
      const chunks = sourceEntry.chunkSource.chunks;
      for (const chunk of chunks.values()) {
        const typedChunk = chunk as SpatiallyIndexedSkeletonChunk;
        if (!this.lodMatches(typedChunk, targetLod)) continue;
        if (typedChunk.state !== ChunkState.GPU_MEMORY || typedChunk.numIndices === 0) {
          continue;
        }
        const filteredChunk = this.updateChunkFilteredBuffer(
          typedChunk,
          visibleSegments,
          hasRegularSkeletonLayer,
          globalNodeMap,
        );
        if (filteredChunk === null) {
          continue;
        }
        renderHelper.drawSkeleton(
          gl,
          edgeShader,
          nodeShader,
          filteredChunk,
          renderContext.projectionParameters,
        );
      }
    }

    renderHelper.endLayer(gl, edgeShader);
  }

  updateChunkFilteredBuffer(
    chunk: SpatiallyIndexedSkeletonChunk,
    visibleSegments: Uint64Set,
    skipVisibleSegments: boolean,
    globalNodeMap?: Map<
      number,
      { chunk: SpatiallyIndexedSkeletonChunk; vertexIndex: number }
    >,
  ): SkeletonChunkInterface | null {
    let missingConnectionsHash = 0;
    if (chunk.missingConnections.length > 0 && globalNodeMap) {
      for (const conn of chunk.missingConnections) {
        const parentNode = globalNodeMap.get(conn.parentId);
        const parentMarker = parentNode ? parentNode.vertexIndex + 1 : 0;
        missingConnectionsHash =
          ((missingConnectionsHash * 1664525) ^
            ((conn.parentId >>> 0) + parentMarker)) >>>
          0;
      }
    }
    if (
      chunk.filteredGeneration === this.generation &&
      chunk.filteredSkipVisibleSegments === skipVisibleSegments &&
      chunk.filteredMissingConnectionsHash === missingConnectionsHash &&
      chunk.filteredIndexBuffer &&
      chunk.filteredVertexAttributeTextures &&
      chunk.numFilteredIndices > 0 &&
      chunk.numFilteredVertices > 0
    ) {
      return {
        vertexAttributeTextures: chunk.filteredVertexAttributeTextures,
        indexBuffer: chunk.filteredIndexBuffer,
        numIndices: chunk.numFilteredIndices,
        numVertices: chunk.numFilteredVertices,
      };
    }

    const gl = this.gl;
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
      chunk.filteredGeneration = this.generation;
      chunk.filteredSkipVisibleSegments = skipVisibleSegments;
      chunk.filteredMissingConnectionsHash = missingConnectionsHash;
      chunk.numFilteredIndices = 0;
      chunk.numFilteredVertices = 0;
      return null;
    }

    const posOffset = vertexAttributeOffsets[0];
    const segmentOffset = vertexAttributeOffsets[1];
    const positions = new Float32Array(
      chunk.vertexAttributes.buffer,
      chunk.vertexAttributes.byteOffset + posOffset,
      chunk.numVertices * 3,
    );
    const segmentIds = new Float32Array(
      chunk.vertexAttributes.buffer,
      chunk.vertexAttributes.byteOffset + segmentOffset,
      chunk.numVertices,
    );
    const segmentInfo = new Map<
      number,
      { include: boolean; color: Float32Array }
    >();

    const getSegmentInfo = (segmentId: number) => {
      let info = segmentInfo.get(segmentId);
      if (info) return info;
      const segmentBigInt = BigInt(segmentId);
      const isVisible = visibleSegments.has(segmentBigInt);
      const alphaForSegment = isVisible
        ? this.displayState.objectAlpha.value
        : this.displayState.hiddenObjectAlpha?.value ?? 0;
      const effectiveAlpha =
        skipVisibleSegments && isVisible ? 0 : alphaForSegment;
      const color = new Float32Array(4);
      getBaseObjectColor(this.displayState, segmentBigInt, color);
      // Encode effectiveAlpha into the alpha channel; do not pre-scale RGB here.
      color[3] = effectiveAlpha;
      const include = effectiveAlpha > 0;
      info = { include, color };
      segmentInfo.set(segmentId, info);
      return info;
    };

    const oldToNew = new Int32Array(chunk.numVertices);
    oldToNew.fill(-1);
    const vertexList: number[] = [];
    for (let v = 0; v < chunk.numVertices; ++v) {
      const rawId = segmentIds[v];
      const segmentId = Math.round(rawId);
      if (!Number.isFinite(segmentId)) {
        continue;
      }
      const info = getSegmentInfo(segmentId);
      if (info.include) {
        oldToNew[v] = vertexList.length;
        vertexList.push(v);
      }
    }

    if (vertexList.length === 0) {
      disposeFilteredTextures();
      chunk.filteredGeneration = this.generation;
      chunk.filteredSkipVisibleSegments = skipVisibleSegments;
      chunk.filteredMissingConnectionsHash = missingConnectionsHash;
      chunk.numFilteredIndices = 0;
      chunk.numFilteredVertices = 0;
      return null;
    }

    const filteredIndices: number[] = [];
    const indices = chunk.indices;
    for (let i = 0; i < chunk.numIndices; i += 2) {
      const aOld = indices[i];
      const bOld = indices[i + 1];
      const aNew = oldToNew[aOld];
      const bNew = oldToNew[bOld];
      if (aNew >= 0 && bNew >= 0) {
        filteredIndices.push(aNew, bNew);
      }
    }

    const interChunkIndices: number[] = [];
    const extraPositions: number[] = [];
    const extraSegments: number[] = [];
    const extraColors: number[] = [];
    const extraVertexMap = new Map<number, number>();
    const chunkPositionCache = new Map<
      SpatiallyIndexedSkeletonChunk,
      Float32Array
    >();
    const getChunkPositions = (targetChunk: SpatiallyIndexedSkeletonChunk) => {
      const cached = chunkPositionCache.get(targetChunk);
      if (cached) return cached;
      const offsets = targetChunk.vertexAttributeOffsets;
      if (!offsets || offsets.length < 1) {
        return null;
      }
      const positionArray = new Float32Array(
        targetChunk.vertexAttributes.buffer,
        targetChunk.vertexAttributes.byteOffset + offsets[0],
        targetChunk.numVertices * 3,
      );
      chunkPositionCache.set(targetChunk, positionArray);
      return positionArray;
    };

    if (chunk.missingConnections.length > 0 && globalNodeMap) {
      for (const conn of chunk.missingConnections) {
        const segmentId = Math.round(conn.skeletonId);
        if (!Number.isFinite(segmentId)) {
          continue;
        }
        const info = getSegmentInfo(segmentId);
        if (!info.include) {
          continue;
        }
        const childNew = oldToNew[conn.vertexIndex];
        if (childNew < 0) {
          continue;
        }
        const parentNode = globalNodeMap.get(conn.parentId);
        if (!parentNode) {
          continue;
        }
        let parentNew = -1;
        if (parentNode.chunk === chunk) {
          const localParent = oldToNew[parentNode.vertexIndex];
          if (localParent >= 0) {
            parentNew = localParent;
          } else {
            continue;
          }
        } else {
          const cached = extraVertexMap.get(conn.parentId);
          if (cached !== undefined) {
            parentNew = cached;
          } else {
            const parentPositions = getChunkPositions(parentNode.chunk);
            if (!parentPositions) {
              continue;
            }
            const posIndex = parentNode.vertexIndex * 3;
            const extraIndex = extraPositions.length / 3;
            parentNew = vertexList.length + extraIndex;
            extraVertexMap.set(conn.parentId, parentNew);
            extraPositions.push(
              parentPositions[posIndex],
              parentPositions[posIndex + 1],
              parentPositions[posIndex + 2],
            );
            extraSegments.push(segmentId);
            extraColors.push(
              info.color[0],
              info.color[1],
              info.color[2],
              info.color[3],
            );
          }
        }
        if (parentNew >= 0) {
          interChunkIndices.push(childNew, parentNew);
        }
      }
    }

    if (interChunkIndices.length > 0) {
      filteredIndices.push(...interChunkIndices);
    }

    if (filteredIndices.length === 0) {
      disposeFilteredTextures();
      chunk.filteredGeneration = this.generation;
      chunk.filteredSkipVisibleSegments = skipVisibleSegments;
      chunk.filteredMissingConnectionsHash = missingConnectionsHash;
      chunk.numFilteredIndices = 0;
      chunk.numFilteredVertices = 0;
      return null;
    }

    const extraVertexCount = extraPositions.length / 3;
    const totalVertexCount = vertexList.length + extraVertexCount;
    const filteredPositions = new Float32Array(totalVertexCount * 3);
    const filteredSegments = new Float32Array(totalVertexCount);
    const filteredColors = new Float32Array(totalVertexCount * 4);
    for (let i = 0; i < vertexList.length; ++i) {
      const oldIndex = vertexList[i];
      const srcStart = oldIndex * 3;
      const dstStart = i * 3;
      filteredPositions[dstStart] = positions[srcStart];
      filteredPositions[dstStart + 1] = positions[srcStart + 1];
      filteredPositions[dstStart + 2] = positions[srcStart + 2];
      const rawId = segmentIds[oldIndex];
      const segmentId = Math.round(rawId);
      filteredSegments[i] = rawId;
      const info = segmentInfo.get(segmentId);
      if (info) {
        filteredColors.set(info.color, i * 4);
      }
    }
    for (let i = 0; i < extraVertexCount; ++i) {
      const dstVertex = vertexList.length + i;
      const posStart = i * 3;
      const dstStart = dstVertex * 3;
      filteredPositions[dstStart] = extraPositions[posStart];
      filteredPositions[dstStart + 1] = extraPositions[posStart + 1];
      filteredPositions[dstStart + 2] = extraPositions[posStart + 2];
      filteredSegments[dstVertex] = extraSegments[i];
      const colorStart = i * 4;
      filteredColors[dstVertex * 4] = extraColors[colorStart];
      filteredColors[dstVertex * 4 + 1] = extraColors[colorStart + 1];
      filteredColors[dstVertex * 4 + 2] = extraColors[colorStart + 2];
      filteredColors[dstVertex * 4 + 3] = extraColors[colorStart + 3];
    }

    const posBytes = new Uint8Array(filteredPositions.buffer);
    const segBytes = new Uint8Array(filteredSegments.buffer);
    const colorBytes = new Uint8Array(filteredColors.buffer);
    const vertexBytes = new Uint8Array(
      posBytes.byteLength + segBytes.byteLength + colorBytes.byteLength,
    );
    vertexBytes.set(posBytes, 0);
    vertexBytes.set(segBytes, posBytes.byteLength);
    vertexBytes.set(colorBytes, posBytes.byteLength + segBytes.byteLength);
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
    chunk.filteredSkipVisibleSegments = skipVisibleSegments;
    chunk.filteredMissingConnectionsHash = missingConnectionsHash;
    chunk.numFilteredIndices = filteredIndices.length;
    chunk.numFilteredVertices = totalVertexCount;

    return {
      vertexAttributeTextures: chunk.filteredVertexAttributeTextures,
      indexBuffer: chunk.filteredIndexBuffer,
      numIndices: chunk.numFilteredIndices,
      numVertices: chunk.numFilteredVertices,
    };
  }

  isReady() {
    return true;
  }
}

export class PerspectiveViewSpatiallyIndexedSkeletonLayer extends PerspectiveViewRenderLayer {
  private renderHelper: RenderHelper;
  private renderOptions: ViewSpecificSkeletonRenderingOptions;
  private transformedSources: TransformedSource[][] = [];
  backend: ChunkRenderLayerFrontend;
  
  constructor(public base: SpatiallyIndexedSkeletonLayer) {
    super();
    this.backend = base.backend;
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

  draw(
    renderContext: PerspectiveViewRenderContext,
    attachment: VisibleLayerInfo<
      PerspectivePanel,
      ThreeDimensionalRenderLayerAttachmentState
    >,
  ) {
    const pixelSizeWatchable = (this.base.displayState as any)
      .spatialSkeletonGridPixelSize3d as WatchableValueInterface<number>
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
    const levels = displayState.spatialSkeletonGridLevels
      ?.value as Array<{ size: { x: number; y: number; z: number } }>
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

  isReady() {
    return this.base.isReady();
  }
}

export class SliceViewSpatiallyIndexedSkeletonLayer extends SliceViewRenderLayer {
  private renderOptions: ViewSpecificSkeletonRenderingOptions;
  constructor(public base: SpatiallyIndexedSkeletonLayer) {
    super(base.chunkManager, {
      getSources: () => {
        return [[{
          chunkSource: base.source,
          chunkToMultiscaleTransform: mat4.create(),
        }]];
      }
    } as any, {
      transform: base.displayState.transform,
      localPosition: (base.displayState as any).localPosition,
    });
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

  draw(renderContext: SliceViewRenderContext) {
    renderContext;
    // No-op for now to test data loading. 
    // SliceViewRenderLayer draw is abstract, so we must implement it.
  }
}

export class SliceViewPanelSpatiallyIndexedSkeletonLayer extends SliceViewPanelRenderLayer {
  private renderHelper: RenderHelper;
  private renderOptions: ViewSpecificSkeletonRenderingOptions;
  private transformedSources: TransformedSource[][] = [];
  constructor(public base: SpatiallyIndexedSkeletonLayer) {
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
    const gridLevel2d = (base.displayState as any).spatialSkeletonGridLevel2d;
    if (gridLevel2d?.changed) {
      this.registerDisposer(
        gridLevel2d.changed.add(this.redrawNeeded.dispatch),
      );
    }
    const lod2d = (base.displayState as any).spatialSkeletonLod2d;
    if (lod2d?.changed) {
      this.registerDisposer(
        lod2d.changed.add(this.redrawNeeded.dispatch),
      );
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
      .spatialSkeletonGridPixelSize2d as WatchableValueInterface<number>
      | undefined;
    if (pixelSizeWatchable !== undefined) {
      const pixelSize =
        renderContext.sliceView.projectionParameters.value.pixelSize;
      if (Number.isFinite(pixelSize) && pixelSizeWatchable.value !== pixelSize) {
        pixelSizeWatchable.value = pixelSize;
      }
    }
    const displayState = this.base.displayState as any;
    const lodValue = displayState.spatialSkeletonLod2d?.value as
      | number
      | undefined;
    const levels = displayState.spatialSkeletonGridLevels
      ?.value as Array<{ size: { x: number; y: number; z: number } }>
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

  isReady() {
    return this.base.isReady();
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
