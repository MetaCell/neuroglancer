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

import { ChunkState } from "#/chunk_manager/base";
import { ChunkRenderLayerFrontend } from "#/chunk_manager/frontend";
import { CoordinateSpace } from "#/coordinate_transform";
import { VisibleLayerInfo } from "#/layer";
import { PerspectivePanel } from "#/perspective_view/panel";
import {
  PerspectiveViewReadyRenderContext,
  PerspectiveViewRenderContext,
  PerspectiveViewRenderLayer,
} from "#/perspective_view/render_layer";
import { RenderLayerTransformOrError } from "#/render_coordinate_transform";
import {
  numRenderScaleHistogramBins,
  RenderScaleHistogram,
  renderScaleHistogramBinSize,
} from "#/render_scale_statistics";
import { SharedWatchableValue } from "#/shared_watchable_value";
import { getNormalizedChunkLayout } from "#/sliceview/base";
import {
  FrontendTransformedSource,
  getVolumetricTransformedSources,
  serializeAllTransformedSources,
} from "#/sliceview/frontend";
import { SliceViewRenderLayer } from "#/sliceview/renderlayer";
import {
  ChunkFormat,
  defineChunkDataShaderAccess,
  MultiscaleVolumeChunkSource,
  VolumeChunk,
  VolumeChunkSource,
} from "#/sliceview/volume/frontend";
import {
  makeCachedDerivedWatchableValue,
  NestedStateManager,
  registerNested,
  WatchableValueInterface,
} from "#/trackable_value";
import { getFrustrumPlanes, mat4, vec3 } from "#/util/geom";
import { getObjectId } from "#/util/object_id";
import {
  forEachVisibleVolumeRenderingChunk,
  getVolumeRenderingNearFarBounds,
  HistogramInformation,
  VOLUME_RENDERING_RENDER_LAYER_RPC_ID,
  VOLUME_RENDERING_RENDER_LAYER_UPDATE_SOURCES_RPC_ID,
} from "#/volume_rendering/base";
import { drawBoxes, glsl_getBoxFaceVertexPosition } from "#/webgl/bounding_box";
import { glsl_COLORMAPS } from "#/webgl/colormaps";
import {
  ParameterizedContextDependentShaderGetter,
  parameterizedContextDependentShaderGetter,
  ParameterizedShaderGetterResult,
  shaderCodeWithLineDirective,
  WatchableShaderError,
} from "#/webgl/dynamic_shader";
import { ShaderBuilder, ShaderModule, ShaderProgram } from "#/webgl/shader";
import {
  addControlsToBuilder,
  setControlsInShader,
  ShaderControlsBuilderState,
  ShaderControlState,
} from "#/webgl/shader_ui_controls";
import { defineVertexId, VertexIdHelper } from "#/webgl/vertex_id";
import { clampToInterval } from "#/util/lerp";
import {
  TrackableVolumeRenderingModeValue,
  VOLUME_RENDERING_MODES,
} from "#/volume_rendering/trackable_volume_rendering_mode";
import {
  DepthStencilRenderbuffer,
  FramebufferConfiguration,
  OffscreenCopyHelper,
  TextureBuffer,
  makeTextureBuffers,
} from "src/webgl/offscreen";
import { Buffer, getMemoizedBuffer } from "#/webgl/buffer";
import { drawQuads } from "#/webgl/quad";

export const VOLUME_RENDERING_DEPTH_SAMPLES_DEFAULT_VALUE = 64;
const VOLUME_RENDERING_DEPTH_SAMPLES_LOG_SCALE_ORIGIN = 1;
const VOLUME_RENDERING_RESOLUTION_INDICATOR_BAR_HEIGHT = 10;

const maxProjectionCopyIntensitySamplerTextureUnit = Symbol(
  "maxProjectionCopyIntensityTextureUnit",
);
const maxProjectionCopyColorSamplerTextureUnit = Symbol(
  "maxProjectionCopyColorTextureUnit",
);
const maxProjectionIntensitySamplerTextureUnit = Symbol(
  "maxProjectionIntensityTextureUnit",
);
const maxProjectionColorSamplerTextureUnit = Symbol(
  "maxProjectionColorTextureUnit",
);

type TransformedVolumeSource = FrontendTransformedSource<
  SliceViewRenderLayer,
  VolumeChunkSource
>;

interface VolumeRenderingAttachmentState {
  sources: NestedStateManager<TransformedVolumeSource[][]>;
}

export interface VolumeRenderingRenderLayerOptions {
  multiscaleSource: MultiscaleVolumeChunkSource;
  transform: WatchableValueInterface<RenderLayerTransformOrError>;
  shaderError: WatchableShaderError;
  shaderControlState: ShaderControlState;
  channelCoordinateSpace: WatchableValueInterface<CoordinateSpace>;
  localPosition: WatchableValueInterface<Float32Array>;
  depthSamplesTarget: WatchableValueInterface<number>;
  chunkResolutionHistogram: RenderScaleHistogram;
  mode: TrackableVolumeRenderingModeValue;
}

interface VolumeRenderingShaderParameters {
  numChannelDimensions: number;
  mode: VOLUME_RENDERING_MODES;
}

const tempMat4 = mat4.create();
const tempVisibleVolumetricClippingPlanes = new Float32Array(24);

export function getVolumeRenderingDepthSamplesBoundsLogScale(): [
  number,
  number,
] {
  const logScaleMax = Math.round(
    VOLUME_RENDERING_DEPTH_SAMPLES_LOG_SCALE_ORIGIN +
      numRenderScaleHistogramBins * renderScaleHistogramBinSize,
  );
  return [VOLUME_RENDERING_DEPTH_SAMPLES_LOG_SCALE_ORIGIN, logScaleMax];
}

function clampAndRoundResolutionTargetValue(value: number) {
  const logScaleDepthSamplesBounds =
    getVolumeRenderingDepthSamplesBoundsLogScale();
  const depthSamplesBounds: [number, number] = [
    2 ** logScaleDepthSamplesBounds[0],
    2 ** logScaleDepthSamplesBounds[1] - 1,
  ];
  return clampToInterval(depthSamplesBounds, Math.round(value)) as number;
}

// function defineMaxProjectionCopyShader(builder: ShaderBuilder) {
//   builder.addAttribute("int", "aDummyVertexId", 0);
//   builder.addOutputBuffer("vec4", "v4f_fragData0", 0);
//   builder.addOutputBuffer("vec4", "v4f_fragData1", 1);
//   builder.setFragmentMain(`
// vec4 v0 = getValue0();
// vec4 v1 = getValue1();

// v4f_fragData0 = v0;
// v4f_fragData1 = v1;
// `);
// }

function defineMaxProjectionCopyShader(builder: ShaderBuilder) {
  builder.addAttribute("int", "aDummyVertexId", 0);
  builder.addAttribute("vec2", "aVertexPosition");
  builder.addVarying("vec2", "vTexCoord");
  builder.addOutputBuffer("vec4", "v4f_fragData0", 0);
  builder.addOutputBuffer("vec4", "v4f_fragData1", 1);
  builder.addTextureSampler(
    "sampler2D",
    "uSamplerIntensity",
    maxProjectionCopyIntensitySamplerTextureUnit,
  );
  builder.addTextureSampler(
    "sampler2D",
    "uSamplerColor",
    maxProjectionCopyColorSamplerTextureUnit,
  );
  builder.setVertexMain(`
gl_Position = vec4(aVertexPosition, 0.0, 1.0);
vTexCoord = 0.5 * (aVertexPosition + 1.0);
`);
  builder.setFragmentMain(`
vec2 uv = vTexCoord;
vec4 max = texture(uSamplerIntensity, uv);
vec4 max2 = texture(uSamplerColor, uv);

v4f_fragData0 = max2;
v4f_fragData1 = max;
`);
}

export class VolumeRenderingRenderLayer extends PerspectiveViewRenderLayer {
  multiscaleSource: MultiscaleVolumeChunkSource;
  transform: WatchableValueInterface<RenderLayerTransformOrError>;
  channelCoordinateSpace: WatchableValueInterface<CoordinateSpace>;
  localPosition: WatchableValueInterface<Float32Array>;
  shaderControlState: ShaderControlState;
  depthSamplesTarget: WatchableValueInterface<number>;
  chunkResolutionHistogram: RenderScaleHistogram;
  mode: TrackableVolumeRenderingModeValue;
  backend: ChunkRenderLayerFrontend;
  private vertexIdHelper: VertexIdHelper;
  private maxProjectionCopyHelper_: OffscreenCopyHelper;
  protected maxProjectionConfiguration_:
    | FramebufferConfiguration<TextureBuffer>
    | undefined;
  private textureVertexBufferArray: Float32Array;
  private textureVertexBuffer: Buffer;
  private maxProjectionCopyShader: ShaderProgram;

  private shaderGetter: ParameterizedContextDependentShaderGetter<
    { emitter: ShaderModule; chunkFormat: ChunkFormat },
    ShaderControlsBuilderState,
    VolumeRenderingShaderParameters
  >;

  get gl() {
    return this.multiscaleSource.chunkManager.gl;
  }

  get isTransparent() {
    return true;
  }

  get isVolumeRendering() {
    return true;
  }

  private get maxProjectionConfiguration() {
    let maxProjectionConfiguration = this.maxProjectionConfiguration_;
    if (maxProjectionConfiguration === undefined) {
      maxProjectionConfiguration = this.maxProjectionConfiguration_ =
        this.registerDisposer(
          new FramebufferConfiguration(this.gl, {
            colorBuffers: makeTextureBuffers(
              this.gl,
              2,
              this.gl.RGBA32F,
              this.gl.RGBA,
              this.gl.FLOAT,
            ),
            depthBuffer: new DepthStencilRenderbuffer(this.gl),
          }),
        );
    }
    return maxProjectionConfiguration;
  }

  // private get maxProjectionCopyHelper() {
  //   let maxProjectionCopyHelper = this.maxProjectionCopyHelper_;
  //   if (maxProjectionCopyHelper === undefined) {
  //     maxProjectionCopyHelper = this.maxProjectionCopyHelper_ =
  //       this.registerDisposer(
  //         OffscreenCopyHelper.get(this.gl, defineMaxProjectionCopyShader, 2),
  //       );
  //   }
  //   return maxProjectionCopyHelper;
  // }

  constructor(options: VolumeRenderingRenderLayerOptions) {
    super();
    this.multiscaleSource = options.multiscaleSource;
    this.transform = options.transform;
    this.channelCoordinateSpace = options.channelCoordinateSpace;
    this.shaderControlState = options.shaderControlState;
    this.localPosition = options.localPosition;
    this.depthSamplesTarget = options.depthSamplesTarget;
    this.chunkResolutionHistogram = options.chunkResolutionHistogram;
    this.mode = options.mode;
    this.registerDisposer(
      this.chunkResolutionHistogram.visibility.add(this.visibility),
    );
    const extraParameters = this.registerDisposer(
      makeCachedDerivedWatchableValue(
        (space: CoordinateSpace, mode: VOLUME_RENDERING_MODES) => ({
          numChannelDimensions: space.rank,
          mode,
        }),
        [this.channelCoordinateSpace, this.mode],
      ),
    );
    this.textureVertexBufferArray = new Float32Array([
      // Triangle 1 - top left, top-rght, bottom-right
      -1, 1, 1, 1, 1, -1,
      // Triangle 2 - top-left, bottom-right, bottom-left
      -1, 1, 1, -1, -1, -1,
    ]);
    this.textureVertexBuffer = this.registerDisposer(
      getMemoizedBuffer(
        this.gl,
        WebGL2RenderingContext.ARRAY_BUFFER,
        () => this.textureVertexBufferArray,
      ),
    ).value;
    this.textureVertexBuffer.setData(this.textureVertexBufferArray);

    this.maxProjectionCopyShader = this.registerDisposer(
      (() => {
        const builder = new ShaderBuilder(this.gl);
        defineMaxProjectionCopyShader(builder);
        return builder.build();
      })(),
    );
    this.shaderGetter = parameterizedContextDependentShaderGetter(
      this,
      this.gl,
      {
        memoizeKey: "VolumeRenderingRenderLayer",
        parameters: options.shaderControlState.builderState,
        getContextKey: ({ emitter, chunkFormat }) =>
          `${getObjectId(emitter)}:${chunkFormat.shaderKey}`,
        shaderError: options.shaderError,
        extraParameters: extraParameters,
        defineShader: (
          builder,
          { emitter, chunkFormat },
          shaderBuilderState,
          shaderParametersState,
        ) => {
          if (shaderBuilderState.parseResult.errors.length !== 0) {
            throw new Error("Invalid UI control specification");
          }
          defineVertexId(builder);
          builder.addFragmentCode(`
#define VOLUME_RENDERING true
`);

          emitter(builder);
          // Near limit in [0, 1] as fraction of full limit.
          builder.addUniform("highp float", "uNearLimitFraction");
          // Far limit in [0, 1] as fraction of full limit.
          builder.addUniform("highp float", "uFarLimitFraction");
          builder.addUniform("highp int", "uMaxSteps");

          // Specifies translation of the current chunk.
          builder.addUniform("highp vec3", "uTranslation");

          // Matrix by which computed vertices will be transformed.
          builder.addUniform("highp mat4", "uModelViewProjectionMatrix");
          builder.addUniform("highp mat4", "uInvModelViewProjectionMatrix");

          // Chunk size in voxels.
          builder.addUniform("highp vec3", "uChunkDataSize");

          builder.addUniform("highp vec3", "uLowerClipBound");
          builder.addUniform("highp vec3", "uUpperClipBound");

          builder.addUniform("highp float", "uBrightnessFactor");
          builder.addVarying("highp vec4", "vNormalizedPosition");

          let glsl_emitRBGA = `
void emitRGBA(vec4 rgba) {
  float alpha = rgba.a * uBrightnessFactor;
  outputColor += vec4(rgba.rgb * alpha, alpha);
}
void emitGrayscale(float value) {
  emitRGBA(vec4(value, value, value, value));
}
void emitRGB(vec3 rgb) {
  emitRGBA(vec4(rgb, 1.0));
}
void emitTransparent() {
  emitRGBA(vec4(0.0, 0.0, 0.0, 0.0));
}
`;
        let glsl_colorInit = `
  outputColor = vec4(0.0, 0.0, 0.0, 0.0);
`
          if (this.mode.value === VOLUME_RENDERING_MODES.MAX) {
            builder.addTextureSampler(
              "sampler2D",
              "uIntensitySampler",
              maxProjectionIntensitySamplerTextureUnit,
            );
            builder.addTextureSampler(
              "sampler2D",
              "uColorSampler",
              maxProjectionColorSamplerTextureUnit,
            );
            glsl_emitRBGA = `
void emitRGBA(float intensity, vec4 rgba) {
  if (intensity > maxIntensity) {
    maxIntensity = intensity;
    outputColor = vec4(rgba.rgb * rgba.a, rgba.a);
  }
}
void emitGrayscale(float value) {
  emitRGBA(value, vec4(value, value, value, value));
}
void emitTransparent() {
  emitRGBA(0.0, vec4(0.0, 0.0, 0.0, 0.0));
}
`;
            glsl_colorInit = `
  vec3 firstPosition = mix(nearPoint, farPoint, uNearLimitFraction + float(startStep) * stepSize);
  vec4 firstClipSpacePosition = uModelViewProjectionMatrix * vec4(firstPosition, 1.0);
  vec2 firstUV = firstClipSpacePosition.xy / firstClipSpacePosition.w;
  firstUV = 0.5 * firstUV + 0.5;
  vec4 firstTexIntensity = texture(uIntensitySampler, firstUV);
  vec4 firstTexColor = texture(uColorSampler, firstUV);
  maxIntensity = firstTexIntensity.r;
  outputColor = firstTexColor;
`
          }
          builder.addVertexCode(glsl_getBoxFaceVertexPosition);

          builder.setVertexMain(`
vec3 boxVertex = getBoxFaceVertexPosition(gl_VertexID);
vec3 position = max(uLowerClipBound, min(uUpperClipBound, uTranslation + boxVertex * uChunkDataSize));
vNormalizedPosition = gl_Position = uModelViewProjectionMatrix * vec4(position, 1.0);
gl_Position.z = 0.0;
`);
          builder.addFragmentCode(`
vec3 curChunkPosition;
vec4 outputColor;
void userMain();
float maxIntensity;
vec4 maxColor;
`);
          defineChunkDataShaderAccess(
            builder,
            chunkFormat,
            shaderParametersState.numChannelDimensions,
            "curChunkPosition",
          );
          builder.addFragmentCode(`
${glsl_emitRBGA}
`);
          builder.setFragmentMainFunction(`
void main() {
  vec2 normalizedPosition = vNormalizedPosition.xy / vNormalizedPosition.w;
  vec4 nearPointH = uInvModelViewProjectionMatrix * vec4(normalizedPosition, -1.0, 1.0);
  vec4 farPointH = uInvModelViewProjectionMatrix * vec4(normalizedPosition, 1.0, 1.0);
  vec3 nearPoint = nearPointH.xyz / nearPointH.w;
  vec3 farPoint = farPointH.xyz / farPointH.w;
  vec3 rayVector = farPoint - nearPoint;
  vec3 boxStart = max(uLowerClipBound, uTranslation);
  vec3 boxEnd = min(boxStart + uChunkDataSize, uUpperClipBound);
  float intersectStart = uNearLimitFraction;
  float intersectEnd = uFarLimitFraction;
  for (int i = 0; i < 3; ++i) {
    float startPt = nearPoint[i];
    float endPt = farPoint[i];
    float boxLower = boxStart[i];
    float boxUpper = boxEnd[i];
    float r = rayVector[i];
    float startFraction;
    float endFraction;
    if (startPt >= boxLower && startPt <= boxUpper) {
      startFraction = 0.0;
    } else {
      startFraction = min((boxLower - startPt) / r, (boxUpper - startPt) / r);
    }
    if (endPt >= boxLower && endPt <= boxUpper) {
      endFraction = 1.0;
    } else {
      endFraction = max((boxLower - startPt) / r, (boxUpper - startPt) / r);
    }
    intersectStart = max(intersectStart, startFraction);
    intersectEnd = min(intersectEnd, endFraction);
  }
  float stepSize = (uFarLimitFraction - uNearLimitFraction) / float(uMaxSteps - 1);
  int startStep = int(floor((intersectStart - uNearLimitFraction) / stepSize));
  int endStep = min(uMaxSteps, int(floor((intersectEnd - uNearLimitFraction) / stepSize)) + 1);
  ${glsl_colorInit}
  for (int step = startStep; step < endStep; ++step) {
    vec3 position = mix(nearPoint, farPoint, uNearLimitFraction + float(step) * stepSize);
    // vec4 clipSpacePosition = uModelViewProjectionMatrix * vec4(position, 1.0);
    // vec2 uv = clipSpacePosition.xy / clipSpacePosition.w;
    // uv = 0.5 * uv + 0.5;
    // bufferIntensity = texture(uIntensitySampler, uv).r;
    // bufferColor = texture(uColorSampler, uv);
    curChunkPosition = position - uTranslation;
    userMain();
  }
  emit(outputColor, 0u);
}
`);
          builder.addFragmentCode(glsl_COLORMAPS);
          addControlsToBuilder(shaderBuilderState, builder);
          builder.addFragmentCode(
            "\n#define main userMain\n" +
              shaderCodeWithLineDirective(shaderBuilderState.parseResult.code) +
              "\n#undef main\n",
          );
        },
      },
    );
    this.vertexIdHelper = this.registerDisposer(VertexIdHelper.get(this.gl));

    this.registerDisposer(
      this.depthSamplesTarget.changed.add(this.redrawNeeded.dispatch),
    );
    this.registerDisposer(
      this.shaderControlState.changed.add(this.redrawNeeded.dispatch),
    );
    this.registerDisposer(
      this.localPosition.changed.add(this.redrawNeeded.dispatch),
    );
    this.registerDisposer(
      this.transform.changed.add(this.redrawNeeded.dispatch),
    );
    this.registerDisposer(this.mode.changed.add(this.redrawNeeded.dispatch));
    this.registerDisposer(
      this.shaderControlState.fragmentMain.changed.add(
        this.redrawNeeded.dispatch,
      ),
    );
    const { chunkManager } = this.multiscaleSource;
    const sharedObject = this.registerDisposer(
      new ChunkRenderLayerFrontend(this.layerChunkProgressInfo),
    );
    const rpc = chunkManager.rpc!;
    sharedObject.RPC_TYPE_ID = VOLUME_RENDERING_RENDER_LAYER_RPC_ID;
    sharedObject.initializeCounterpart(rpc, {
      chunkManager: chunkManager.rpcId,
      localPosition: this.registerDisposer(
        SharedWatchableValue.makeFromExisting(rpc, this.localPosition),
      ).rpcId,
      renderScaleTarget: this.registerDisposer(
        SharedWatchableValue.makeFromExisting(rpc, this.depthSamplesTarget),
      ).rpcId,
    });
    this.backend = sharedObject;
  }

  get dataType() {
    return this.multiscaleSource.dataType;
  }

  attach(
    attachment: VisibleLayerInfo<
      PerspectivePanel,
      VolumeRenderingAttachmentState
    >,
  ) {
    super.attach(attachment);
    attachment.state = {
      sources: attachment.registerDisposer(
        registerNested(
          (context, transform, displayDimensionRenderInfo) => {
            const transformedSources = getVolumetricTransformedSources(
              displayDimensionRenderInfo,
              transform,
              (options) => this.multiscaleSource.getSources(options),
              attachment.messages,
              this,
            ) as TransformedVolumeSource[][];
            for (const scales of transformedSources) {
              for (const tsource of scales) {
                context.registerDisposer(tsource.source);
              }
            }
            attachment.view.flushBackendProjectionParameters();
            this.backend.rpc!.invoke(
              VOLUME_RENDERING_RENDER_LAYER_UPDATE_SOURCES_RPC_ID,
              {
                layer: this.backend.rpcId,
                view: attachment.view.rpcId,
                sources: serializeAllTransformedSources(transformedSources),
              },
            );
            this.redrawNeeded.dispatch();
            return transformedSources;
          },
          this.transform,
          attachment.view.displayDimensionRenderInfo,
        ),
      ),
    };
  }

  get chunkManager() {
    return this.multiscaleSource.chunkManager;
  }

  draw(
    renderContext: PerspectiveViewRenderContext,
    attachment: VisibleLayerInfo<
      PerspectivePanel,
      VolumeRenderingAttachmentState
    >,
  ) {
    if (!renderContext.emitColor) return;
    const allSources = attachment.state!.sources.value;
    if (allSources.length === 0) return;
    let curPhysicalSpacing = 0;
    let curOptimalSamples = 0;
    let curHistogramInformation: HistogramInformation = {
      spatialScales: new Map(),
      activeIndex: 0,
    };
    let shader: ShaderProgram | null = null;
    let prevChunkFormat: ChunkFormat | undefined | null;
    let shaderResult: ParameterizedShaderGetterResult<
      ShaderControlsBuilderState,
      VolumeRenderingShaderParameters
    >;
    // Size of chunk (in voxels) in the "display" subspace of the chunk coordinate space.
    const chunkDataDisplaySize = vec3.create();

    const { gl } = this;
    this.vertexIdHelper.enable();

    const { chunkResolutionHistogram: renderScaleHistogram } = this;
    renderScaleHistogram.begin(
      this.chunkManager.chunkQueueManager.frameNumberCounter.frameNumber,
    );

    const endShader = () => {
      if (shader === null) return;
      if (prevChunkFormat !== null) {
        prevChunkFormat!.endDrawing(gl, shader);
      }
      if (presentCount !== 0 || notPresentCount !== 0) {
        let index = curHistogramInformation.spatialScales.size - 1;
        const alreadyStoredSamples = new Set<number>([
          clampAndRoundResolutionTargetValue(curOptimalSamples),
        ]);
        curHistogramInformation.spatialScales.forEach(
          (optimalSamples, physicalSpacing) => {
            const roundedSamples =
              clampAndRoundResolutionTargetValue(optimalSamples);
            if (
              index !== curHistogramInformation.activeIndex &&
              !alreadyStoredSamples.has(roundedSamples)
            ) {
              renderScaleHistogram.add(
                physicalSpacing,
                optimalSamples,
                0,
                VOLUME_RENDERING_RESOLUTION_INDICATOR_BAR_HEIGHT,
                true,
              );
              alreadyStoredSamples.add(roundedSamples);
            }
            index--;
          },
        );
        renderScaleHistogram.add(
          curPhysicalSpacing,
          curOptimalSamples,
          presentCount,
          notPresentCount,
        );
      }
    };
    let newSource = true;

    const { projectionParameters } = renderContext;

    let chunks: Map<string, VolumeChunk>;
    let presentCount = 0;
    let notPresentCount = 0;
    let chunkDataSize: Uint32Array | undefined;

    const chunkRank = this.multiscaleSource.rank;
    const chunkPosition = vec3.create();

    gl.enable(WebGL2RenderingContext.CULL_FACE);
    gl.cullFace(WebGL2RenderingContext.FRONT);

    forEachVisibleVolumeRenderingChunk(
      renderContext.projectionParameters,
      this.localPosition.value,
      this.depthSamplesTarget.value,
      allSources[0],
      (
        transformedSource,
        ignored1,
        physicalSpacing,
        optimalSamples,
        ignored2,
        histogramInformation,
      ) => {
        ignored1;
        ignored2;
        curPhysicalSpacing = physicalSpacing;
        curOptimalSamples = optimalSamples;
        curHistogramInformation = histogramInformation;
        const chunkLayout = getNormalizedChunkLayout(
          projectionParameters,
          transformedSource.chunkLayout,
        );
        const source = transformedSource.source as VolumeChunkSource;
        const { fixedPositionWithinChunk, chunkDisplayDimensionIndices } =
          transformedSource;
        for (const chunkDim of chunkDisplayDimensionIndices) {
          fixedPositionWithinChunk[chunkDim] = 0;
        }
        const chunkFormat = source.chunkFormat;
        if (chunkFormat !== prevChunkFormat) {
          prevChunkFormat = chunkFormat;
          endShader();
          shaderResult = this.shaderGetter({
            emitter: renderContext.emitter,
            chunkFormat: chunkFormat!,
          });
          shader = shaderResult.shader;
          if (shader !== null) {
            shader.bind();
            if (chunkFormat !== null) {
              setControlsInShader(
                gl,
                shader,
                this.shaderControlState,
                shaderResult.parameters.parseResult.controls,
              );
              chunkFormat.beginDrawing(gl, shader);
              chunkFormat.beginSource(gl, shader);
            }
          }
        }
        chunkDataSize = undefined;
        if (shader === null) return;
        chunks = source.chunks;
        chunkDataDisplaySize.fill(1);

        // Compute projection matrix that transforms chunk layout coordinates to device
        // coordinates.
        const modelViewProjection = mat4.multiply(
          tempMat4,
          projectionParameters.viewProjectionMat,
          chunkLayout.transform,
        );
        gl.uniformMatrix4fv(
          shader.uniform("uModelViewProjectionMatrix"),
          false,
          modelViewProjection,
        );
        const clippingPlanes = tempVisibleVolumetricClippingPlanes;
        getFrustrumPlanes(clippingPlanes, modelViewProjection);
        mat4.invert(modelViewProjection, modelViewProjection);
        gl.uniformMatrix4fv(
          shader.uniform("uInvModelViewProjectionMatrix"),
          false,
          modelViewProjection,
        );
        const { near, far, adjustedNear, adjustedFar } =
          getVolumeRenderingNearFarBounds(
            clippingPlanes,
            transformedSource.lowerClipDisplayBound,
            transformedSource.upperClipDisplayBound,
          );
        const step =
          (adjustedFar - adjustedNear) / (this.depthSamplesTarget.value - 1);
        const brightnessFactor = step / (far - near);
        gl.uniform1f(shader.uniform("uBrightnessFactor"), brightnessFactor);
        const nearLimitFraction = (adjustedNear - near) / (far - near);
        const farLimitFraction = (adjustedFar - near) / (far - near);
        gl.uniform1f(shader.uniform("uNearLimitFraction"), nearLimitFraction);
        gl.uniform1f(shader.uniform("uFarLimitFraction"), farLimitFraction);
        gl.uniform1i(
          shader.uniform("uMaxSteps"),
          this.depthSamplesTarget.value,
        );
        gl.uniform3fv(
          shader.uniform("uLowerClipBound"),
          transformedSource.lowerClipDisplayBound,
        );
        gl.uniform3fv(
          shader.uniform("uUpperClipBound"),
          transformedSource.upperClipDisplayBound,
        );
      },
      (transformedSource) => {
        if (shader === null) return;
        const key = transformedSource.curPositionInChunks.join();
        const chunk = chunks.get(key);
        if (chunk !== undefined && chunk.state === ChunkState.GPU_MEMORY) {
          const originalChunkSize = transformedSource.chunkLayout.size;
          const newChunkDataSize = chunk.chunkDataSize;
          const {
            chunkDisplayDimensionIndices,
            fixedPositionWithinChunk,
            chunkTransform: { channelToChunkDimensionIndices },
          } = transformedSource;
          if (newChunkDataSize !== chunkDataSize) {
            chunkDataSize = newChunkDataSize;

            for (let i = 0; i < 3; ++i) {
              const chunkDim = chunkDisplayDimensionIndices[i];
              chunkDataDisplaySize[i] =
                chunkDim === -1 || chunkDim >= chunkRank
                  ? 1
                  : chunkDataSize[chunkDim];
            }
            gl.uniform3fv(
              shader.uniform("uChunkDataSize"),
              chunkDataDisplaySize,
            );
          }
          const { chunkGridPosition } = chunk;
          for (let i = 0; i < 3; ++i) {
            const chunkDim = chunkDisplayDimensionIndices[i];
            chunkPosition[i] =
              chunkDim === -1 || chunkDim >= chunkRank
                ? 0
                : originalChunkSize[i] * chunkGridPosition[chunkDim];
          }
          if (prevChunkFormat != null) {
            prevChunkFormat.bindChunk(
              gl,
              shader!,
              chunk,
              fixedPositionWithinChunk,
              chunkDisplayDimensionIndices,
              channelToChunkDimensionIndices,
              newSource,
            );
          }
          newSource = false;
          gl.uniform3fv(shader.uniform("uTranslation"), chunkPosition);
          if (this.mode.value === VOLUME_RENDERING_MODES.MAX) {
            gl.blendFunc(
              WebGL2RenderingContext.ONE,
              WebGL2RenderingContext.ZERO,
            );
            const maxProjectionHelper = renderContext.maxProjectionHelper!;
            maxProjectionHelper.bindMaxProjectionBuffer();
            const intensityTextureUnit = shader.textureUnit(
              maxProjectionIntensitySamplerTextureUnit,
            );
            const colorTextureUnit = shader.textureUnit(
              maxProjectionColorSamplerTextureUnit,
            );
            gl.activeTexture(
              WebGL2RenderingContext.TEXTURE0 + intensityTextureUnit,
            );
            gl.bindTexture(
              WebGL2RenderingContext.TEXTURE_2D,
              (
                maxProjectionHelper.transparentConfiguration
                  .colorBuffers[1] as TextureBuffer
              ).texture,
            );
            gl.activeTexture(
              WebGL2RenderingContext.TEXTURE0 + colorTextureUnit,
            );
            gl.bindTexture(
              WebGL2RenderingContext.TEXTURE_2D,
              (
                maxProjectionHelper.transparentConfiguration
                  .colorBuffers[0] as TextureBuffer
              ).texture,
            );
            drawBoxes(gl, 1, 1);
            gl.activeTexture(
              WebGL2RenderingContext.TEXTURE0 + intensityTextureUnit,
            );
            gl.bindTexture(WebGL2RenderingContext.TEXTURE_2D, null);
            gl.activeTexture(
              WebGL2RenderingContext.TEXTURE0 + colorTextureUnit,
            );
            gl.bindTexture(WebGL2RenderingContext.TEXTURE_2D, null);
            // Setup state
            const localShader = this.maxProjectionCopyShader;
            localShader.bind();
            //this.vertexIdHelper.disable();
            renderContext.bindFramebuffer();
            gl.blendFunc(
              WebGL2RenderingContext.ONE,
              WebGL2RenderingContext.ZERO,
            );
            const aVertexPosition = localShader.attribute("aVertexPosition");
            this.textureVertexBuffer.bindToVertexAttrib(
              aVertexPosition,
              2,
              WebGL2RenderingContext.FLOAT,
            );
            const maxProjectionCopyIntensityTextureUnit =
              localShader.textureUnit(
                maxProjectionCopyIntensitySamplerTextureUnit,
              );
            const maxProjectionCopyColorTextureUnit = localShader.textureUnit(
              maxProjectionCopyColorSamplerTextureUnit,
            );
            gl.activeTexture(
              WebGL2RenderingContext.TEXTURE0 +
                maxProjectionCopyIntensityTextureUnit,
            );
            gl.bindTexture(
              WebGL2RenderingContext.TEXTURE_2D,
              (
                maxProjectionHelper.maxProjectionConfiguration
                  .colorBuffers[1] as TextureBuffer
              ).texture,
            );
            gl.activeTexture(
              WebGL2RenderingContext.TEXTURE0 +
                maxProjectionCopyColorTextureUnit,
            );
            gl.bindTexture(
              WebGL2RenderingContext.TEXTURE_2D,
              (
                maxProjectionHelper.maxProjectionConfiguration
                  .colorBuffers[0] as TextureBuffer
              ).texture,
            );

            //Copy the result so far over
            // this.maxProjectionCopyHelper.draw(
            //   (
            //     maxProjectionHelper.maxProjectionConfiguration
            //       .colorBuffers[0] as TextureBuffer
            //   ).texture,
            //   (
            //     maxProjectionHelper.maxProjectionConfiguration
            //       .colorBuffers[1] as TextureBuffer
            //   ).texture,
            // );
            drawQuads(gl, 1, 1);

            // // Restore state
            gl.disableVertexAttribArray(aVertexPosition);
            gl.activeTexture(
              WebGL2RenderingContext.TEXTURE0 +
                maxProjectionCopyIntensityTextureUnit,
            );
            gl.bindTexture(WebGL2RenderingContext.TEXTURE_2D, null);
            gl.activeTexture(
              WebGL2RenderingContext.TEXTURE0 +
                maxProjectionCopyColorTextureUnit,
            );
            gl.bindTexture(WebGL2RenderingContext.TEXTURE_2D, null);
            maxProjectionHelper.bindMaxProjectionBuffer();
            const chunkFormat = transformedSource.source.chunkFormat;
            // TODO (skm can I overcome calling this?)
            chunkFormat.beginDrawing(gl, shader);
            shader.bind();
            //this.vertexIdHelper.enable();
          } else {
            drawBoxes(gl, 1, 1);
          }
          ++presentCount;
        } else {
          ++notPresentCount;
        }
      },
    );
    gl.disable(WebGL2RenderingContext.CULL_FACE);
    endShader();
    this.vertexIdHelper.disable();
  }

  isReady(
    renderContext: PerspectiveViewReadyRenderContext,
    attachment: VisibleLayerInfo<
      PerspectivePanel,
      VolumeRenderingAttachmentState
    >,
  ) {
    const allSources = attachment.state!.sources.value;
    if (allSources.length === 0) return true;
    let missing = false;
    forEachVisibleVolumeRenderingChunk(
      renderContext.projectionParameters,
      this.localPosition.value,
      this.depthSamplesTarget.value,
      allSources[0],
      () => {},
      (tsource) => {
        const chunk = tsource.source.chunks.get(
          tsource.curPositionInChunks.join(),
        );
        if (chunk === undefined || chunk.state !== ChunkState.GPU_MEMORY) {
          missing = true;
        }
      },
    );
    return missing;
  }
}
