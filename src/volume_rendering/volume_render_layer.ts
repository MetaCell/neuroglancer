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
import { ShaderModule, ShaderProgram } from "#/webgl/shader";
import {
  addControlsToBuilder,
  setControlsInShader,
  ShaderControlsBuilderState,
  ShaderControlState,
} from "#/webgl/shader_ui_controls";
import { defineVertexId, VertexIdHelper } from "#/webgl/vertex_id";
import { clampToInterval } from "#/util/lerp";

export const VOLUME_RENDERING_DEPTH_SAMPLES_DEFAULT_VALUE = 64;
const VOLUME_RENDERING_DEPTH_SAMPLES_LOG_SCALE_ORIGIN = 1;
const VOLUME_RENDERING_RESOLUTION_INDICATOR_BAR_HEIGHT = 10;

const depthSamplerTextureUnit = Symbol("depthSamplerTextureUnit");

export const glsl_emitRGBAVolumeRendering = `
void emitRGBA(vec4 rgba) {
  float correctedAlpha = clamp(rgba.a * uBrightnessFactor * uGain, 0.0, 1.0);
  float weightedAlpha = correctedAlpha * computeOITWeight(correctedAlpha, depthAtRayPosition);
  outputColor += vec4(rgba.rgb * weightedAlpha, weightedAlpha);
  revealage *= 1.0 - correctedAlpha;
}
`;

type TransformedVolumeSource = FrontendTransformedSource<
  SliceViewRenderLayer,
  VolumeChunkSource
>;

interface VolumeRenderingAttachmentState {
  sources: NestedStateManager<TransformedVolumeSource[][]>;
}

export interface VolumeRenderingRenderLayerOptions {
  gain: WatchableValueInterface<number>;
  multiscaleSource: MultiscaleVolumeChunkSource;
  transform: WatchableValueInterface<RenderLayerTransformOrError>;
  shaderError: WatchableShaderError;
  shaderControlState: ShaderControlState;
  channelCoordinateSpace: WatchableValueInterface<CoordinateSpace>;
  localPosition: WatchableValueInterface<Float32Array>;
  depthSamplesTarget: WatchableValueInterface<number>;
  chunkResolutionHistogram: RenderScaleHistogram;
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

export class VolumeRenderingRenderLayer extends PerspectiveViewRenderLayer {
  gain: WatchableValueInterface<number>;
  multiscaleSource: MultiscaleVolumeChunkSource;
  transform: WatchableValueInterface<RenderLayerTransformOrError>;
  channelCoordinateSpace: WatchableValueInterface<CoordinateSpace>;
  localPosition: WatchableValueInterface<Float32Array>;
  shaderControlState: ShaderControlState;
  depthSamplesTarget: WatchableValueInterface<number>;
  chunkResolutionHistogram: RenderScaleHistogram;
  backend: ChunkRenderLayerFrontend;
  private vertexIdHelper: VertexIdHelper;

  private shaderGetter: ParameterizedContextDependentShaderGetter<
    { emitter: ShaderModule; chunkFormat: ChunkFormat; wireFrame: boolean },
    ShaderControlsBuilderState,
    number
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

  constructor(options: VolumeRenderingRenderLayerOptions) {
    super();
    this.gain = options.gain;
    this.multiscaleSource = options.multiscaleSource;
    this.transform = options.transform;
    this.channelCoordinateSpace = options.channelCoordinateSpace;
    this.shaderControlState = options.shaderControlState;
    this.localPosition = options.localPosition;
    this.depthSamplesTarget = options.depthSamplesTarget;
    this.chunkResolutionHistogram = options.chunkResolutionHistogram;
    this.registerDisposer(
      this.chunkResolutionHistogram.visibility.add(this.visibility),
    );
    const numChannelDimensions = this.registerDisposer(
      makeCachedDerivedWatchableValue(
        (space) => space.rank,
        [this.channelCoordinateSpace],
      ),
    );
    this.shaderGetter = parameterizedContextDependentShaderGetter(
      this,
      this.gl,
      {
        memoizeKey: "VolumeRenderingRenderLayer",
        parameters: options.shaderControlState.builderState,
        getContextKey: ({ emitter, chunkFormat, wireFrame }) =>
          `${getObjectId(emitter)}:${chunkFormat.shaderKey}:${wireFrame}`,
        shaderError: options.shaderError,
        extraParameters: numChannelDimensions,
        defineShader: (
          builder,
          { emitter, chunkFormat, wireFrame },
          shaderBuilderState,
          numChannelDimensions,
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
          builder.addUniform("highp float", "uChunkNumber");

          builder.addUniform("highp vec3", "uLowerClipBound");
          builder.addUniform("highp vec3", "uUpperClipBound");

          builder.addUniform("highp float", "uBrightnessFactor");
          builder.addUniform("highp float", "uGain");
          builder.addVarying("highp vec4", "vNormalizedPosition");
          builder.addTextureSampler(
            "sampler2D",
            "uDepthSampler",
            depthSamplerTextureUnit,
          );
          builder.addVertexCode(glsl_getBoxFaceVertexPosition);

          builder.setVertexMain(`
vec3 boxVertex = getBoxFaceVertexPosition(gl_VertexID);
vec3 position = max(uLowerClipBound, min(uUpperClipBound, uTranslation + boxVertex * uChunkDataSize));
vNormalizedPosition = gl_Position = uModelViewProjectionMatrix * vec4(position, 1.0);
gl_Position.z = 0.0;
`);
          builder.addFragmentCode(`
vec3 curChunkPosition;
float depthAtRayPosition;
vec4 outputColor;
float revealage;
void userMain();
`);
          defineChunkDataShaderAccess(
            builder,
            chunkFormat,
            numChannelDimensions,
            "curChunkPosition",
          );
          builder.addFragmentCode([
            glsl_emitRGBAVolumeRendering,
            `
void emitRGB(vec3 rgb) {
  emitRGBA(vec4(rgb, 1.0));
}
void emitGrayscale(float value) {
  emitRGBA(vec4(value, value, value, value));
}
void emitTransparent() {
  emitRGBA(vec4(0.0, 0.0, 0.0, 0.0));
}
float computeDepthFromClipSpace(vec4 clipSpacePosition) {
  float NDCDepthCoord = clipSpacePosition.z / clipSpacePosition.w;
  return (NDCDepthCoord + 1.0) * 0.5;
}
vec2 computeUVFromClipSpace(vec4 clipSpacePosition) {
  vec2 NDCPosition = clipSpacePosition.xy / clipSpacePosition.w;
  return (NDCPosition + 1.0) * 0.5;
}
`,
          ]);
          if (wireFrame) {
            builder.setFragmentMainFunction(`
void main() {
  outputColor = vec4(uChunkNumber, uChunkNumber, uChunkNumber, 1.0);
  emit(outputColor, 0u);
}
`);
          } else {
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
  outputColor = vec4(0, 0, 0, 0);
  revealage = 1.0;
  for (int step = startStep; step < endStep; ++step) {
    vec3 position = mix(nearPoint, farPoint, uNearLimitFraction + float(step) * stepSize);
    vec4 clipSpacePosition = uModelViewProjectionMatrix * vec4(position, 1.0);
    depthAtRayPosition = computeDepthFromClipSpace(clipSpacePosition);
    vec2 uv = computeUVFromClipSpace(clipSpacePosition);
    float depthInBuffer = texture(uDepthSampler, uv).r;
    bool rayPositionBehindOpaqueObject = (1.0 - depthAtRayPosition) < depthInBuffer;
    if (rayPositionBehindOpaqueObject) {
      break;
    }
    curChunkPosition = position - uTranslation;
    userMain();
  }
  emitAccumAndRevealage(outputColor, 1.0 - revealage, 0u);
}
`);
          }
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
    this.registerDisposer(this.gain.changed.add(this.redrawNeeded.dispatch));
    this.registerDisposer(
      this.shaderControlState.changed.add(this.redrawNeeded.dispatch),
    );
    this.registerDisposer(
      this.localPosition.changed.add(this.redrawNeeded.dispatch),
    );
    this.registerDisposer(
      this.transform.changed.add(this.redrawNeeded.dispatch),
    );
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
      number
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
      shader.unbindTransferFunctionTextures();
      if (prevChunkFormat !== null) {
        prevChunkFormat!.endDrawing(gl, shader);
      }
      const depthTextureUnit = shader.textureUnit(depthSamplerTextureUnit);
      gl.activeTexture(WebGL2RenderingContext.TEXTURE0 + depthTextureUnit);
      gl.bindTexture(WebGL2RenderingContext.TEXTURE_2D, null);
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
    let chunkNumber = 1;

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
            wireFrame: renderContext.wireFrame,
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
              if (
                renderContext.depthBufferTexture !== undefined &&
                renderContext.depthBufferTexture !== null
              ) {
                const depthTextureUnit = shader.textureUnit(
                  depthSamplerTextureUnit,
                );
                gl.activeTexture(
                  WebGL2RenderingContext.TEXTURE0 + depthTextureUnit,
                );
                gl.bindTexture(
                  WebGL2RenderingContext.TEXTURE_2D,
                  renderContext.depthBufferTexture,
                );
              } else {
                throw new Error(
                  "Depth buffer texture ID for volume rendering is undefined or null",
                );
              }
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
        const optimalSampleRate = optimalSamples;
        const actualSampleRate = this.depthSamplesTarget.value;
        const brightnessFactor = optimalSampleRate / actualSampleRate;
        gl.uniform1f(shader.uniform("uBrightnessFactor"), brightnessFactor);
        const nearLimitFraction = (adjustedNear - near) / (far - near);
        const farLimitFraction = (adjustedFar - near) / (far - near);
        gl.uniform1f(shader.uniform("uNearLimitFraction"), nearLimitFraction);
        gl.uniform1f(shader.uniform("uFarLimitFraction"), farLimitFraction);
        gl.uniform1f(shader.uniform("uGain"), Math.exp(this.gain.value));
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
          if (renderContext.wireFrame) {
            const normChunkNumber = chunkNumber / chunks.size;
            gl.uniform1f(shader.uniform("uChunkNumber"), normChunkNumber);
            ++chunkNumber;
          }
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
          drawBoxes(gl, 1, 1);
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