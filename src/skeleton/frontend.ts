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
import type { LayerView, VisibleLayerInfo } from "#src/layer/index.js";
import type { PerspectivePanel } from "#src/perspective_view/panel.js";
import type { PerspectiveViewRenderContext } from "#src/perspective_view/render_layer.js";
import { PerspectiveViewRenderLayer } from "#src/perspective_view/render_layer.js";
import type {
  RenderLayer,
  ThreeDimensionalRenderLayerAttachmentState,
} from "#src/renderlayer.js";
import { update3dRenderLayerAttachment } from "#src/renderlayer.js";
import {
  forEachVisibleSegment,
  getObjectKey,
} from "#src/segmentation_display_state/base.js";
import type { SegmentationDisplayState3D } from "#src/segmentation_display_state/frontend.js";
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
} from "#src/skeleton/base.js";
import { RENDERED_VIEW_ADD_LAYER_RPC_ID } from "#src/render_layer_common.js";
import type { SliceViewPanel } from "#src/sliceview/panel.js";
import { SharedWatchableValue } from "#src/shared_watchable_value.js";
import type { RPC } from "#src/worker_rpc.js";
import {
  getVolumetricTransformedSources,
  serializeAllTransformedSources,
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
} from "#src/sliceview/frontend.js";
import type { SliceViewChunkSpecification } from "#src/sliceview/base.js";
import { ChunkLayout } from "#src/sliceview/chunk_layout.js";
import {
  TrackableValue,
  WatchableValue,
  WatchableValueInterface,
  registerNested,
} from "#src/trackable_value.js";
import { DataType } from "#src/util/data_type.js";
import { RefCounted } from "#src/util/disposable.js";
import type { vec3 } from "#src/util/geom.js";
import { mat4 } from "#src/util/geom.js";
import { verifyFinitePositiveFloat } from "#src/util/json.js";
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

interface SkeletonLayerInterface {
  vertexAttributes: VertexAttributeRenderInfo[];
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
  get vertexAttributes(): VertexAttributeRenderInfo[] {
    return this.base.vertexAttributes;
  }

  defineCommonShader(builder: ShaderBuilder) {
    defineVertexId(builder);
    builder.addUniform("highp vec4", "uColor");
    builder.addUniform("highp mat4", "uProjection");
    builder.addUniform("highp uint", "uPickID");
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

          builder.addFragmentCode(`
vec4 segmentColor() {
  return uColor;
}
void emitRGB(vec3 color) {
  emit(vec4(color * uColor.a, uColor.a * getLineAlpha() * ${this.getCrossSectionFadeFactor()}), uPickID);
}
void emitDefault() {
  emit(vec4(uColor.rgb, uColor.a * getLineAlpha() * ${this.getCrossSectionFadeFactor()}), uPickID);
}
`);
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

          builder.addFragmentCode(`
vec4 segmentColor() {
  return uColor;
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

  setColor(gl: GL, shader: ShaderProgram, color: vec3) {
    gl.uniform4fv(shader.uniform("uColor"), color);
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
      const glError = gl.getError();
      if (glError !== gl.NO_ERROR) {
        console.error('[SKELETON-RENDER] WebGL error after drawLines:', glError);
      }
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
    const { gl, source, displayState } = this;
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
          renderHelper.setColor(gl, edgeShader, <vec3>(<Float32Array>color));
          nodeShader.bind();
          renderHelper.setColor(gl, nodeShader, <vec3>(<Float32Array>color));
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

  // Filtering support
  filteredIndexBuffer: GLBuffer | undefined;
  filteredGeneration: number = -1;
  numFilteredIndices: number = 0;
  // Cache of per-segment compacted vertex textures for 2D rendering (no shader changes)
  filteredVertexTextureCache?: Map<number, { textures: (WebGLTexture | null)[]; numVertices: number; generation: number }>;

  constructor(source: SpatiallyIndexedSkeletonSource, chunkData: SkeletonChunkData) {
    super(source, chunkData);
    this.vertexAttributes = chunkData.vertexAttributes;
    const indices = (this.indices = chunkData.indices);
    this.numVertices = chunkData.numVertices;
    this.numIndices = indices.length;
    this.vertexAttributeOffsets = chunkData.vertexAttributeOffsets;
    this.missingConnections = (chunkData as any).missingConnections || [];

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

    // Dispose cached per-segment textures
    if (this.filteredVertexTextureCache) {
      for (const entry of this.filteredVertexTextureCache.values()) {
        for (const tex of entry.textures) {
          if (tex) gl.deleteTexture(tex);
        }
      }
      this.filteredVertexTextureCache.clear();
      this.filteredVertexTextureCache = undefined;
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

export class SpatiallyIndexedSkeletonLayer extends RefCounted implements SkeletonLayerInterface {
  layerChunkProgressInfo = new LayerChunkProgressInfo();
  redrawNeeded = new NullarySignal();
  vertexAttributes: VertexAttributeRenderInfo[];
  fallbackShaderParameters = new WatchableValue(
    getFallbackBuilderState(parseShaderUiControls(DEFAULT_FRAGMENT_MAIN)),
  );
  backend: ChunkRenderLayerFrontend;
  localPosition: WatchableValueInterface<Float32Array>;
  rpc: RPC | undefined;

  private generation = 0;

  get visibility() {
    return this.displayState.objectAlpha;
  }

  constructor(
    public chunkManager: ChunkManager,
    public source: SpatiallyIndexedSkeletonSource,
    public displayState: SkeletonLayerDisplayState & {
      localPosition: WatchableValueInterface<Float32Array>;
    },
  ) {
    super();
    this.localPosition = displayState.localPosition;
    registerRedrawWhenSegmentationDisplayState3DChanged(displayState, this);
    this.displayState.shaderError.value = undefined;
    const { skeletonRenderingOptions: renderingOptions } = displayState;
    this.registerDisposer(
      renderingOptions.shader.changed.add(() => {
        this.displayState.shaderError.value = undefined;
        this.redrawNeeded.dispatch();
      }),
    );

    this.vertexAttributes = source.vertexAttributes;
    // Monitor visible segments changes to update generation
    this.registerDisposer(
      registerNested((context, segmentationGroup) => {
        context.registerDisposer(
          segmentationGroup.visibleSegments.changed.add(() => {
            this.generation++;
            this.redrawNeeded.dispatch();
          })
        );
      }, this.displayState.segmentationGroupState)
    );

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
    
    sharedObject.initializeCounterpart(rpc, {
      chunkManager: chunkManager.rpcId,
      localPosition: this.registerDisposer(
        SharedWatchableValue.makeFromExisting(
          rpc,
          this.localPosition,
        ),
      ).rpcId,
      renderScaleTarget: renderScaleTargetWatchable.rpcId,
    });
    this.backend = sharedObject;
  }

  get gl() {
    return this.chunkManager.chunkQueueManager.gl;
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
  ) {
    const lineWidth = renderOptions.lineWidth.value;
    const { gl, source, displayState } = this;
    if (displayState.objectAlpha.value <= 0.0) {
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

    // Render all chunks with proper segment colors
    const chunks = source.chunks;
    const tempColor = new Float32Array(4);
    tempColor[3] = 1.0; // Set alpha to 1.0

    // Build global node map for inter-chunk connections
    const globalNodeMap = new Map<number, { chunk: SpatiallyIndexedSkeletonChunk; vertexIndex: number }>();

    for (const chunk of chunks.values()) {
      if (chunk.state === ChunkState.GPU_MEMORY) {
        // Populate global map with all nodes from this chunk
        for (const [nodeId, vertexIndex] of chunk.nodeMap.entries()) {
          globalNodeMap.set(nodeId, { chunk, vertexIndex });
        }
      }
    }

    // Render chunks with intra-chunk and inter-chunk connections
    for (const chunk of chunks.values()) {
      if (chunk.state === ChunkState.GPU_MEMORY && chunk.numIndices > 0) {
        // Extract segment IDs from vertex attributes
        const attrOffset = chunk.vertexAttributeOffsets && chunk.vertexAttributeOffsets.length > 1
          ? chunk.vertexAttributeOffsets[1]
          : 0;
        const segmentIds = new Float32Array(
          chunk.vertexAttributes.buffer,
          chunk.vertexAttributes.byteOffset + attrOffset,
          chunk.numVertices,
        );

        // Get unique segment IDs in this chunk for color verification
        const uniqueSegmentIds = new Set<number>();
        for (let i = 0; i < chunk.numVertices; ++i) {
          uniqueSegmentIds.add(Math.round(segmentIds[i]));
        }

        // Group indices by segment ID
        const segmentIndicesMap = new Map<number, number[]>();
        const indices = chunk.indices;

        for (let i = 0; i < chunk.numIndices; i += 2) {
          const idx0 = indices[i];
          const idx1 = indices[i + 1];
          const segmentId = Math.round(segmentIds[idx0]);

          if (!segmentIndicesMap.has(segmentId)) {
            segmentIndicesMap.set(segmentId, []);
          }
          segmentIndicesMap.get(segmentId)!.push(idx0, idx1);
        }

        // Render each segment with its own color
        for (const [segmentId, segmentIndicesList] of segmentIndicesMap) {
          const bigintId = BigInt(segmentId);
          const color = getBaseObjectColor(displayState, bigintId, tempColor);

          // Set color for this segment
          edgeShader.bind();
          renderHelper.setColor(gl, edgeShader, <vec3>(<Float32Array>color));
          nodeShader.bind();
          renderHelper.setColor(gl, nodeShader, <vec3>(<Float32Array>color));

          // Build per-segment vertex list and remap indices (2D only optimization)
          let originalIndexBuffer = chunk.indexBuffer;
          let originalNumIndices = chunk.numIndices;
          let originalTextures = chunk.vertexAttributeTextures;
          let originalNumVertices = chunk.numVertices;

          // Collect all vertex indices belonging to this segment
          const vertexList: number[] = [];
          const oldToNew = new Map<number, number>();
          for (let v = 0; v < chunk.numVertices; ++v) {
            if (Math.round(segmentIds[v]) === segmentId) {
              oldToNew.set(v, vertexList.length);
              vertexList.push(v);
            }
          }

          // Remap the edge indices to the compact index space for this segment
          const remappedIndices: number[] = [];
          for (let i = 0; i < segmentIndicesList.length; i += 2) {
            const aOld = segmentIndicesList[i];
            const bOld = segmentIndicesList[i + 1];
            const aNew = oldToNew.get(aOld);
            const bNew = oldToNew.get(bOld);
            if (aNew !== undefined && bNew !== undefined) {
              remappedIndices.push(aNew, bNew);
            }
          }

          if (remappedIndices.length === 0 || vertexList.length === 0) {
            continue;
          }

          // Build or retrieve cached compacted vertex textures for this segment
          if (renderHelper.targetIsSliceView) {
            if (!chunk.filteredVertexTextureCache) {
              chunk.filteredVertexTextureCache = new Map();
            }
            let cacheEntry = chunk.filteredVertexTextureCache.get(segmentId);
            if (!cacheEntry || cacheEntry.generation !== this.generation) {
              // Create compacted attribute arrays: positions (vec3) + segment (float)
              const posOffset = chunk.vertexAttributeOffsets ? chunk.vertexAttributeOffsets[0] : 0;
              const posAll = new Float32Array(
                chunk.vertexAttributes.buffer,
                chunk.vertexAttributes.byteOffset + posOffset,
                chunk.numVertices * 3,
              );

              const posFiltered = new Float32Array(vertexList.length * 3);
              for (let j = 0; j < vertexList.length; ++j) {
                const vOld = vertexList[j];
                const srcStart = vOld * 3;
                const dstStart = j * 3;
                posFiltered[dstStart] = posAll[srcStart];
                posFiltered[dstStart + 1] = posAll[srcStart + 1];
                posFiltered[dstStart + 2] = posAll[srcStart + 2];
              }

              const segFiltered = new Float32Array(vertexList.length);
              for (let j = 0; j < vertexList.length; ++j) {
                const vOld = vertexList[j];
                segFiltered[j] = segmentIds[vOld];
              }

              // Concatenate into a single byte array with offsets [0, posBytes]
              const posBytes = new Uint8Array(posFiltered.buffer.slice(0));
              const segBytes = new Uint8Array(segFiltered.buffer.slice(0));
              const vertexBytes = new Uint8Array(posBytes.byteLength + segBytes.byteLength);
              vertexBytes.set(posBytes, 0);
              vertexBytes.set(segBytes, posBytes.byteLength);
              const vertexOffsets = new Uint32Array([0, posBytes.byteLength]);

              const filteredTextures = uploadVertexAttributesToGPU(
                gl,
                vertexBytes,
                vertexOffsets,
                this.source.attributeTextureFormats,
              );

              cacheEntry = {
                textures: filteredTextures,
                numVertices: vertexList.length,
                generation: this.generation,
              };
              chunk.filteredVertexTextureCache.set(segmentId, cacheEntry);
            }

            // Create remapped index buffer for this segment
            const segmentIndexBuffer = GLBuffer.fromData(
              gl,
              new Uint32Array(remappedIndices),
              WebGL2RenderingContext.ARRAY_BUFFER,
              WebGL2RenderingContext.STATIC_DRAW,
            );

            // Temporarily swap buffers and textures to filtered versions
            originalIndexBuffer = chunk.indexBuffer;
            originalNumIndices = chunk.numIndices;
            originalTextures = chunk.vertexAttributeTextures;
            originalNumVertices = chunk.numVertices;

            chunk.indexBuffer = segmentIndexBuffer;
            chunk.numIndices = remappedIndices.length;
            chunk.vertexAttributeTextures = cacheEntry.textures;
            chunk.numVertices = cacheEntry.numVertices;

            // Render this segment (2D filtered)
            renderHelper.drawSkeleton(
              gl,
              edgeShader,
              nodeShader,
              chunk,
              renderContext.projectionParameters,
            );

            // Restore original state and dispose temp index buffer
            chunk.indexBuffer = originalIndexBuffer;
            chunk.numIndices = originalNumIndices;
            chunk.vertexAttributeTextures = originalTextures;
            chunk.numVertices = originalNumVertices;
            segmentIndexBuffer.dispose();
          } else {
            // 3D path: keep existing behavior (no filtering)
            const segmentIndexBuffer = GLBuffer.fromData(
              gl,
              new Uint32Array(segmentIndicesList),
              WebGL2RenderingContext.ARRAY_BUFFER,
              WebGL2RenderingContext.STATIC_DRAW,
            );
            const originalIndexBuffer3d = chunk.indexBuffer;
            const originalNumIndices3d = chunk.numIndices;
            chunk.indexBuffer = segmentIndexBuffer;
            chunk.numIndices = segmentIndicesList.length;
            renderHelper.drawSkeleton(
              gl,
              edgeShader,
              nodeShader,
              chunk,
              renderContext.projectionParameters,
            );
            chunk.indexBuffer = originalIndexBuffer3d;
            chunk.numIndices = originalNumIndices3d;
            segmentIndexBuffer.dispose();
          }
        }

        // Handle inter-chunk connections
        if (chunk.missingConnections.length > 0) {
          const interChunkEdges: Array<{ segmentId: number; childIdx: number; parentChunk: SpatiallyIndexedSkeletonChunk; parentIdx: number }> = [];

          for (const conn of chunk.missingConnections) {
            const parentNode = globalNodeMap.get(conn.parentId);
            if (parentNode) {
              // We found the parent in another chunk, store edge info
              interChunkEdges.push({
                segmentId: conn.skeletonId,
                childIdx: conn.vertexIndex,
                parentChunk: parentNode.chunk,
                parentIdx: parentNode.vertexIndex,
              });
            }
          }

          if (interChunkEdges.length > 0) {
            // console.log(`Found ${interChunkEdges.length} inter-chunk connections in chunk`);
            // TODO: Render inter-chunk edges by creating combined vertex buffers
            // This requires combining vertex data from different chunks
          }
        }
      }
    }

    renderHelper.endLayer(gl, edgeShader);
  }

  updateChunkFilteredBuffer(
    chunk: SpatiallyIndexedSkeletonChunk,
    visibleSegments: Uint64Set
  ) {
    if (chunk.filteredGeneration === this.generation && chunk.filteredIndexBuffer) {
      return;
    }

    const gl = this.gl;
    if (!chunk.filteredIndexBuffer) {
      chunk.filteredIndexBuffer = new GLBuffer(gl, WebGL2RenderingContext.ELEMENT_ARRAY_BUFFER);
    }
    // chunk.vertexAttributeOffsets[0] is start of POSITION (attribute 0).
    // chunk.vertexAttributeOffsets[1] is start of SEGMENT (attribute 1).

    const attrOffset = chunk.vertexAttributeOffsets ? chunk.vertexAttributeOffsets[1] : 0;
    const attrBytes = chunk.vertexAttributes;
    const ids = new Float32Array(
      attrBytes.buffer,
      attrBytes.byteOffset + attrOffset,
      chunk.numVertices,
    );

    const indices = chunk.indices;
    const filteredIndices: number[] = [];
    const numIndices = chunk.numIndices;

    // Check visibility
    for (let i = 0; i < numIndices; i += 2) {
      const idxA = indices[i];
      const idxB = indices[i + 1];
      const idA = ids[idxA];
      if (SpatiallyIndexedSkeletonLayer.isSegmentVisible(idA, visibleSegments)) {
        filteredIndices.push(idxA, idxB);
      }
    }

    chunk.filteredIndexBuffer.setData(new Uint32Array(filteredIndices));
    chunk.filteredGeneration = this.generation;
    chunk.numFilteredIndices = filteredIndices.length;
  }


  static isSegmentVisible(id: number, visibleSegments: Uint64Set): boolean {
    const id64 = BigInt(Math.round(id));
    return visibleSegments.has(id64);
  }

  isReady() {
    return true;
  }
}

export class PerspectiveViewSpatiallyIndexedSkeletonLayer extends PerspectiveViewRenderLayer {
  private renderHelper: RenderHelper;
  private renderOptions: ViewSpecificSkeletonRenderingOptions;
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
            () => [[{
              chunkSource: baseLayer.source,
              chunkToMultiscaleTransform: mat4.create(),
            }]],
            attachment.messages,
            this,
          );
          for (const scales of transformedSources) {
            for (const tsource of scales) {
              context.registerDisposer(tsource.source);
            }
          }
          attachment.view.flushBackendProjectionParameters();
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
    if (!renderContext.emitColor && renderContext.alreadyEmittedPickID) {
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
    this.registerDisposer(base.redrawNeeded.add(this.redrawNeeded.dispatch));
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
