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

import {
  Chunk,
  ChunkRenderLayerBackend,
  ChunkSource,
  withChunkManager,
} from "#src/chunk_manager/backend.js";
import { ChunkState } from "#src/chunk_manager/base.js";
import { decodeVertexPositionsAndIndices } from "#src/mesh/backend.js";
import { withSegmentationLayerBackendState } from "#src/segmentation_display_state/backend.js";
import {
  forEachVisibleSegment,
  getObjectKey,
} from "#src/segmentation_display_state/base.js";
import {
  SKELETON_LAYER_RPC_ID,
  SPATIALLY_INDEXED_SKELETON_RENDER_LAYER_RPC_ID,
  SPATIALLY_INDEXED_SKELETON_RENDER_LAYER_UPDATE_SOURCES_RPC_ID,
  SPATIALLY_INDEXED_SKELETON_SLICEVIEW_RENDER_LAYER_RPC_ID,
} from "#src/skeleton/base.js";
import type { TypedNumberArray } from "#src/util/array.js";
import type { Endianness } from "#src/util/endian.js";
import {
  getBasePriority,
  getPriorityTier,
  withSharedVisibility,
} from "#src/visibility_priority/backend.js";

import {
  SliceViewChunk,
  SliceViewChunkSourceBackend,
  SliceViewRenderLayerBackend,
} from "#src/sliceview/backend.js";
import {
  SliceViewChunkSpecification,
  forEachVisibleVolumetricChunk,
  type SliceViewBase,
  type SliceViewProjectionParameters,
  type TransformedSource,
} from "#src/sliceview/base.js";
import { SCALE_PRIORITY_MULTIPLIER } from "#src/sliceview/backend.js";
import type { RenderLayerBackendAttachment } from "#src/render_layer_backend.js";
import { RenderLayerBackend } from "#src/render_layer_backend.js";
import type { RenderedViewBackend } from "#src/render_layer_backend.js";
import type { SharedWatchableValue } from "#src/shared_watchable_value.js";
import {
  type DisplayDimensionRenderInfo,
  validateDisplayDimensionRenderInfoProperty,
} from "#src/navigation_state.js";
import {
  freeSkeletonChunkSystemMemory,
  getVertexAttributeBytes,
  serializeSkeletonChunkData,
  type SkeletonChunkData,
} from "#src/skeleton/skeleton_chunk_serialization.js";
import type { RPC } from "#src/worker_rpc.js";
import { registerRPC, registerSharedObject } from "#src/worker_rpc.js";
import { deserializeTransformedSources } from "#src/sliceview/backend.js";
import { debounce } from "lodash-es";


export interface SpatiallyIndexedSkeletonChunkSpecification extends SliceViewChunkSpecification {
  chunkLayout: any;
}

const SKELETON_CHUNK_PRIORITY = 60;
const SPATIALLY_INDEXED_SKELETON_LOD_DEBOUNCE_MS = 300;

registerRPC(
  SPATIALLY_INDEXED_SKELETON_RENDER_LAYER_UPDATE_SOURCES_RPC_ID,
  function (x) {
    const view = this.get(x.view) as RenderedViewBackend;
    const layer = this.get(
      x.layer,
    ) as SpatiallyIndexedSkeletonRenderLayerBackend;
    const attachment = layer.attachments.get(
      view,
    )! as RenderLayerBackendAttachment<
      RenderedViewBackend,
      SpatiallyIndexedSkeletonRenderLayerAttachmentState
    >;
    attachment.state!.transformedSources = deserializeTransformedSources<
      SpatiallyIndexedSkeletonSourceBackend,
      SpatiallyIndexedSkeletonRenderLayerBackend
    >(this, x.sources, layer);
    attachment.state!.displayDimensionRenderInfo = x.displayDimensionRenderInfo;
    layer.chunkManager.scheduleUpdateChunkPriorities();
  },
);

// Chunk that contains the skeleton of a single object.
export class SkeletonChunk extends Chunk implements SkeletonChunkData {
  objectId: bigint = 0n;
  vertexPositions: Float32Array | null = null;
  vertexAttributes: TypedNumberArray[] | null = null;
  indices: Uint32Array | null = null;

  initializeSkeletonChunk(key: string, objectId: bigint) {
    super.initialize(key);
    this.objectId = objectId;
  }

  freeSystemMemory() {
    freeSkeletonChunkSystemMemory(this);
  }

  serialize(msg: any, transfers: any[]) {
    super.serialize(msg, transfers);
    serializeSkeletonChunkData(this, msg, transfers);
    freeSkeletonChunkSystemMemory(this);
  }

  downloadSucceeded() {
    this.systemMemoryBytes = this.gpuMemoryBytes =
      this.indices!.byteLength + getVertexAttributeBytes(this);
    super.downloadSucceeded();
  }
}

export class SkeletonSource extends ChunkSource {
  declare chunks: Map<string, SkeletonChunk>;
  getChunk(objectId: bigint) {
    const key = getObjectKey(objectId);
    let chunk = this.chunks.get(key);
    if (chunk === undefined) {
      chunk = this.getNewChunk_(SkeletonChunk);
      chunk.initializeSkeletonChunk(key, objectId);
      this.addChunk(chunk);
    }
    return chunk;
  }
}

@registerSharedObject(SKELETON_LAYER_RPC_ID)
export class SkeletonLayer extends withSegmentationLayerBackendState(
  withSharedVisibility(withChunkManager(ChunkRenderLayerBackend)),
) {
  source: SkeletonSource;

  constructor(rpc: RPC, options: any) {
    super(rpc, options);
    this.source = this.registerDisposer(
      rpc.getRef<SkeletonSource>(options.source),
    );
    this.registerDisposer(
      this.chunkManager.recomputeChunkPriorities.add(() => {
        this.updateChunkPriorities();
      }),
    );
  }

  private updateChunkPriorities() {
    const visibility = this.visibility.value;
    if (visibility === Number.NEGATIVE_INFINITY) {
      return;
    }
    this.chunkManager.registerLayer(this);
    const priorityTier = getPriorityTier(visibility);
    const basePriority = getBasePriority(visibility);
    const { source, chunkManager } = this;
    forEachVisibleSegment(this, (objectId) => {
      const chunk = source.getChunk(objectId);
      ++this.numVisibleChunksNeeded;
      if (chunk.state === ChunkState.GPU_MEMORY) {
        ++this.numVisibleChunksAvailable;
      }
      chunkManager.requestChunk(
        chunk,
        priorityTier,
        basePriority + SKELETON_CHUNK_PRIORITY,
      );
    });
  }
}

/**
 * Extracts vertex positions and edge vertex indices of the specified endianness from `data'.
 *
 * See documentation of decodeVertexPositionsAndIndices.
 */
export function decodeSkeletonVertexPositionsAndIndices(
  chunk: SkeletonChunk,
  data: ArrayBuffer,
  endianness: Endianness,
  vertexByteOffset: number,
  numVertices: number,
  indexByteOffset?: number,
  numEdges?: number,
) {
  const meshData = decodeVertexPositionsAndIndices(
    /*verticesPerPrimitive=*/ 2,
    data,
    endianness,
    vertexByteOffset,
    numVertices,
    indexByteOffset,
    numEdges,
  );
  chunk.vertexPositions = meshData.vertexPositions as Float32Array;
  chunk.indices = meshData.indices as Uint32Array;
}

export class SpatiallyIndexedSkeletonChunk extends SliceViewChunk implements SkeletonChunkData {
  vertexPositions: Float32Array | null = null;
  vertexAttributes: TypedNumberArray[] | null = null;
  indices: Uint32Array | null = null;
  lod: number = 0;
  missingConnections: Array<{ nodeId: number; parentId: number; vertexIndex: number; skeletonId: number }> = [];
  nodeMap: Map<number, number> = new Map(); // Maps node ID to vertex index

  freeSystemMemory() {
    freeSkeletonChunkSystemMemory(this);
  }

  serialize(msg: any, transfers: any[]) {
    super.serialize(msg, transfers);
    serializeSkeletonChunkData(this, msg, transfers);
    freeSkeletonChunkSystemMemory(this);
  }

  downloadSucceeded() {
    this.systemMemoryBytes = this.gpuMemoryBytes =
      this.indices!.byteLength + getVertexAttributeBytes(this);
    super.downloadSucceeded();
  }
}

export class SpatiallyIndexedSkeletonSourceBackend extends SliceViewChunkSourceBackend<SpatiallyIndexedSkeletonChunkSpecification, SpatiallyIndexedSkeletonChunk> {
  chunkConstructor = SpatiallyIndexedSkeletonChunk;
  currentLod: number = 0;

  getChunk(chunkGridPosition: Float32Array) {
    const lodValue = this.currentLod;
    const key = `${chunkGridPosition.join()}:${lodValue}`;
    let chunk = this.chunks.get(key);
    if (chunk === undefined) {
      chunk = this.getNewChunk_(this.chunkConstructor) as SpatiallyIndexedSkeletonChunk;
      chunk.initializeVolumeChunk(key, chunkGridPosition);
      chunk.lod = lodValue;
      this.addChunk(chunk);
    }
    return chunk;
  }
}

interface SpatiallyIndexedSkeletonRenderLayerAttachmentState {
  displayDimensionRenderInfo: DisplayDimensionRenderInfo;
  transformedSources: TransformedSource<SpatiallyIndexedSkeletonRenderLayerBackend, SpatiallyIndexedSkeletonSourceBackend>[][];
}

@registerSharedObject(SPATIALLY_INDEXED_SKELETON_RENDER_LAYER_RPC_ID)
export class SpatiallyIndexedSkeletonRenderLayerBackend extends withChunkManager(
  RenderLayerBackend,
) {
  localPosition: SharedWatchableValue<Float32Array>;
  renderScaleTarget: SharedWatchableValue<number>;
  skeletonLod: SharedWatchableValue<number>;
  skeletonGridLevel: SharedWatchableValue<number>;

  constructor(rpc: RPC, options: any) {
    super(rpc, options);
    this.renderScaleTarget = rpc.get(options.renderScaleTarget);
    this.localPosition = rpc.get(options.localPosition);
    this.skeletonLod = rpc.get(options.skeletonLod);
    this.skeletonGridLevel = rpc.get(options.skeletonGridLevel);
    const scheduleUpdateChunkPriorities = () =>
      this.chunkManager.scheduleUpdateChunkPriorities();
    this.registerDisposer(
      this.localPosition.changed.add(scheduleUpdateChunkPriorities),
    );
    this.registerDisposer(
      this.renderScaleTarget.changed.add(scheduleUpdateChunkPriorities),
    );
    this.registerDisposer(
      this.skeletonGridLevel.changed.add(scheduleUpdateChunkPriorities),
    );
    
    // Debounce LOD changes to avoid making requests for every slider value
    const debouncedLodUpdate = debounce(() => {
      scheduleUpdateChunkPriorities();
    }, SPATIALLY_INDEXED_SKELETON_LOD_DEBOUNCE_MS);
    
    this.registerDisposer(
      this.skeletonLod.changed.add(() => {
        // Trigger a reschedule; LOD-specific chunks are keyed by LOD.
        debouncedLodUpdate();
      }),
    );
    this.registerDisposer(
      this.chunkManager.recomputeChunkPriorities.add(() =>
        this.recomputeChunkPriorities(),
      ),
    );
  }

  attach(
    attachment: RenderLayerBackendAttachment<
      RenderedViewBackend,
      SpatiallyIndexedSkeletonRenderLayerAttachmentState
    >,
  ) {
    const scheduleUpdateChunkPriorities = () =>
      this.chunkManager.scheduleUpdateChunkPriorities();
    const { view } = attachment;
    attachment.registerDisposer(scheduleUpdateChunkPriorities);
    attachment.registerDisposer(
      view.projectionParameters.changed.add(scheduleUpdateChunkPriorities),
    );
    attachment.registerDisposer(
      view.visibility.changed.add(scheduleUpdateChunkPriorities),
    );
    attachment.state = {
      displayDimensionRenderInfo:
        view.projectionParameters.value.displayDimensionRenderInfo,
      transformedSources: [],
    };
  }

  private recomputeChunkPriorities() {
    this.chunkManager.registerLayer(this);
    for (const attachment of this.attachments.values()) {
      const { view } = attachment;
      const visibility = view.visibility.value;
      if (visibility === Number.NEGATIVE_INFINITY) {
        continue;
      }
      const attachmentState =
        attachment.state! as SpatiallyIndexedSkeletonRenderLayerAttachmentState;
      const { transformedSources } = attachmentState;
      if (
        transformedSources.length === 0 ||
        !validateDisplayDimensionRenderInfoProperty(
          attachmentState,
          view.projectionParameters.value.displayDimensionRenderInfo,
        )
      ) {
        continue;
      }
      const priorityTier = getPriorityTier(visibility);
      const basePriority = getBasePriority(visibility);

    const projectionParameters = view.projectionParameters.value;
      const { chunkManager } = this;
      const sliceProjectionParameters =
        projectionParameters as SliceViewProjectionParameters;
      const pixelSize =
        "pixelSize" in sliceProjectionParameters
          ? sliceProjectionParameters.pixelSize
          : undefined;
      let resolvedPixelSize = pixelSize;
      if (resolvedPixelSize === undefined) {
        const voxelPhysicalScales =
          projectionParameters.displayDimensionRenderInfo?.voxelPhysicalScales;
        if (voxelPhysicalScales) {
          let computedPixelSize = 0;
          const { invViewMatrix } = projectionParameters;
          for (let i = 0; i < 3; ++i) {
            const s = voxelPhysicalScales[i];
            const x = invViewMatrix[i];
            computedPixelSize += (s * x) ** 2;
          }
          resolvedPixelSize = Math.sqrt(computedPixelSize);
        }
      }
      const renderScaleTarget = this.renderScaleTarget.value;
      const skeletonGridLevel = this.skeletonGridLevel.value;

      const selectScales = (
        scales: TransformedSource<
          SpatiallyIndexedSkeletonRenderLayerBackend,
          SpatiallyIndexedSkeletonSourceBackend
        >[],
      ): Array<{
        tsource: TransformedSource<
          SpatiallyIndexedSkeletonRenderLayerBackend,
          SpatiallyIndexedSkeletonSourceBackend
        >;
        scaleIndex: number;
      }> => {
        if (scales.length === 0) {
          return [];
        }
        const gridIndexedScales = scales
          .map((tsource, scaleIndex) => ({
            tsource,
            scaleIndex,
            gridIndex: (tsource.source as any).parameters?.gridIndex as
              | number
              | undefined,
          }))
          .filter((entry) => entry.gridIndex !== undefined);
        if (gridIndexedScales.length === scales.length) {
          let minGridIndex = Number.POSITIVE_INFINITY;
          let maxGridIndex = Number.NEGATIVE_INFINITY;
          for (const entry of gridIndexedScales) {
            const gridIndex = entry.gridIndex as number;
            minGridIndex = Math.min(minGridIndex, gridIndex);
            maxGridIndex = Math.max(maxGridIndex, gridIndex);
          }
          const clampedGridLevel = Math.min(
            Math.max(skeletonGridLevel, minGridIndex),
            maxGridIndex,
          );
          let selectedEntry =
            gridIndexedScales.find(
              (entry) => entry.gridIndex === clampedGridLevel,
            ) ?? gridIndexedScales[0];
          if (selectedEntry.gridIndex !== clampedGridLevel) {
            let bestDistance = Number.POSITIVE_INFINITY;
            for (const entry of gridIndexedScales) {
              const distance = Math.abs(
                (entry.gridIndex as number) - clampedGridLevel,
              );
              if (distance < bestDistance) {
                bestDistance = distance;
                selectedEntry = entry;
              }
            }
          }
          return [
            {
              tsource: selectedEntry.tsource,
              scaleIndex: selectedEntry.scaleIndex,
            },
          ];
        }
        if (resolvedPixelSize === undefined) {
          return scales.map((tsource, scaleIndex) => ({
            tsource,
            scaleIndex,
          }));
        }
        const pixelSizeWithMargin = resolvedPixelSize * 1.1;
        const smallestVoxelSize = scales[0].effectiveVoxelSize;
        const canImproveOnVoxelSize = (voxelSize: Float32Array) => {
          const targetSize = pixelSizeWithMargin * renderScaleTarget;
          for (let i = 0; i < 3; ++i) {
            const size = voxelSize[i];
            if (size > targetSize && size > 1.01 * smallestVoxelSize[i]) {
              return true;
            }
          }
          return false;
        };
        const improvesOnPrevVoxelSize = (
          voxelSize: Float32Array,
          prevVoxelSize: Float32Array,
        ) => {
          const targetSize = pixelSizeWithMargin * renderScaleTarget;
          for (let i = 0; i < 3; ++i) {
            const size = voxelSize[i];
            const prevSize = prevVoxelSize[i];
            if (
              Math.abs(targetSize - size) < Math.abs(targetSize - prevSize) &&
              size < 1.01 * prevSize
            ) {
              return true;
            }
          }
          return false;
        };

        const selected: Array<{
          tsource: TransformedSource<
            SpatiallyIndexedSkeletonRenderLayerBackend,
            SpatiallyIndexedSkeletonSourceBackend
          >;
          scaleIndex: number;
        }> = [];
        let scaleIndex = scales.length - 1;
        let prevVoxelSize: Float32Array | undefined;
        while (true) {
          const tsource = scales[scaleIndex];
          const selectionVoxelSize = tsource.effectiveVoxelSize;
          if (
            prevVoxelSize !== undefined &&
            !improvesOnPrevVoxelSize(selectionVoxelSize, prevVoxelSize)
          ) {
            break;
          }
          selected.push({ tsource, scaleIndex });
          if (scaleIndex === 0) break;
          if (!canImproveOnVoxelSize(selectionVoxelSize)) break;
          prevVoxelSize = selectionVoxelSize;
          --scaleIndex;
        }
        return selected;
      };

      const lodValue = this.skeletonLod.value;
      for (const scales of transformedSources) {
        const selectedScales = selectScales(scales);
        for (const { tsource, scaleIndex } of selectedScales) {
          const source =
            tsource.source as SpatiallyIndexedSkeletonSourceBackend;
          source.currentLod = lodValue;
          forEachVisibleVolumetricChunk(
            projectionParameters,
            this.localPosition.value,
            tsource,
            () => {
              const chunk = source.getChunk(tsource.curPositionInChunks);
              ++this.numVisibleChunksNeeded;
              if (chunk.state === ChunkState.GPU_MEMORY) {
                ++this.numVisibleChunksAvailable;
              }
              const priority = 0;
              chunkManager.requestChunk(
                chunk,
                priorityTier,
                basePriority + priority + SCALE_PRIORITY_MULTIPLIER * scaleIndex,
              );
            },
          );
        }
      }
    }
  }
}

@registerSharedObject(SPATIALLY_INDEXED_SKELETON_SLICEVIEW_RENDER_LAYER_RPC_ID)
export class SpatiallyIndexedSkeletonSliceViewRenderLayerBackend extends SliceViewRenderLayerBackend {
  skeletonGridLevel: SharedWatchableValue<number>;
  skeletonLod: SharedWatchableValue<number>;

  constructor(rpc: RPC, options: any) {
    super(rpc, options);
    this.skeletonGridLevel = rpc.get(options.skeletonGridLevel);
    this.skeletonLod = rpc.get(options.skeletonLod);
    const chunkManager = rpc.get(options.chunkManager);
    const scheduleUpdateChunkPriorities = () =>
      chunkManager.scheduleUpdateChunkPriorities();
    this.registerDisposer(
      this.skeletonGridLevel.changed.add(scheduleUpdateChunkPriorities),
    );
    // Debounce LOD changes to avoid making requests for every slider value.
    const debouncedLodUpdate = debounce(() => {
      scheduleUpdateChunkPriorities();
    }, SPATIALLY_INDEXED_SKELETON_LOD_DEBOUNCE_MS);
    
    this.registerDisposer(
      this.skeletonLod.changed.add(() => {
        debouncedLodUpdate();
      }),
    );
  }

  prepareChunkSourceForRequest(source: SpatiallyIndexedSkeletonSourceBackend) {
    source.currentLod = this.skeletonLod.value;
  }

  filterVisibleSources(
    sliceView: SliceViewBase,
    sources: readonly TransformedSource[],
  ): Iterable<TransformedSource> {
    const lodValue = this.skeletonLod.value;
    for (const tsource of sources) {
      const source = tsource.source as SpatiallyIndexedSkeletonSourceBackend;
      source.currentLod = lodValue;
    }

    const gridIndexedSources = sources
      .map((tsource, scaleIndex) => ({
        tsource,
        scaleIndex,
        gridIndex: (tsource.source as any).parameters?.gridIndex as
          | number
          | undefined,
      }))
      .filter((entry) => entry.gridIndex !== undefined);
    if (gridIndexedSources.length === sources.length && sources.length > 0) {
      let minGridIndex = Number.POSITIVE_INFINITY;
      let maxGridIndex = Number.NEGATIVE_INFINITY;
      for (const entry of gridIndexedSources) {
        const gridIndex = entry.gridIndex as number;
        minGridIndex = Math.min(minGridIndex, gridIndex);
        maxGridIndex = Math.max(maxGridIndex, gridIndex);
      }
      const clampedGridLevel = Math.min(
        Math.max(this.skeletonGridLevel.value, minGridIndex),
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
    return super.filterVisibleSources(sliceView, sources);
  }
}
