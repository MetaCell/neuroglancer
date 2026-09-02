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

import {
  AnnotationDisplayState,
  AnnotationLayerState,
} from "#src/annotation/annotation_layer_state.js";
import {
  type Annotation,
  type AnnotationReference,
  AnnotationType,
  LocalAnnotationSource,
  type Point,
  type PolyLine,
} from "#src/annotation/index.js";
import type { LoadedDataSubsource } from "#src/layer/layer_data_source.js";
import type { SegmentationUserLayer } from "#src/layer/segmentation/index.js";
import { RenderLayerRole } from "#src/renderlayer.js";
import {
  type SkeletonFindPathEndpoint,
  type SkeletonFindPathResultNode,
  type SkeletonFindPathState,
} from "#src/skeleton/find_path.js";
import { TrackableBoolean } from "#src/trackable_boolean.js";
import { WatchableValue } from "#src/trackable_value.js";
import { RefCounted } from "#src/util/disposable.js";

const ASSOCIATED_SEGMENTS_RELATIONSHIP = "associated segments";
const SPATIAL_SKELETON_FIND_PATH_SUBSOURCE_ID = "spatialSkeletonFindPath";

export const SPATIAL_SKELETON_FIND_PATH_SOURCE_DESCRIPTION = "find path source";
export const SPATIAL_SKELETON_FIND_PATH_TARGET_DESCRIPTION = "find path target";
export const SPATIAL_SKELETON_FIND_PATH_RESULT_DESCRIPTION = "find path result";

const SPATIAL_SKELETON_FIND_PATH_SHADER = `
void main() {
  setColor(defaultColor());
  setLineWidth(3.0);
  setPointMarkerSize(9.0);
  setPointMarkerBorderWidth(2.0);
  setPointMarkerBorderColor(vec4(0.0, 0.0, 0.0, 1.0));
}
`;

export interface SpatialSkeletonFindPathAnnotationControllerOptions {
  layer: SegmentationUserLayer;
  loadedSubsource: LoadedDataSubsource;
  state: SkeletonFindPathState;
}

type AnnotationKind = "source" | "target" | "result";

function positionsEqual(a: Float32Array, b: Float32Array) {
  return a.length === b.length && a.every((value, index) => value === b[index]);
}

function relatedSegmentsFor(segmentId: bigint) {
  return [BigUint64Array.of(segmentId)];
}

function hasRelatedSegment(
  relatedSegments: BigUint64Array[] | undefined,
  segmentId: bigint,
) {
  return (
    relatedSegments?.length === 1 &&
    relatedSegments[0].length === 1 &&
    relatedSegments[0][0] === segmentId
  );
}

/**
 * Owns the local, derived annotations used to display spatial-skeleton
 * find-path state for one loaded subsource.
 *
 * The find-path state belongs to the skeleton state and is serializable.
 * Annotation references and generated annotation IDs are deliberately kept
 * only on this controller.
 */
export class SpatialSkeletonFindPathAnnotationController extends RefCounted {
  readonly annotationSource: LocalAnnotationSource;
  readonly annotationState: AnnotationLayerState;

  private sourceReference: AnnotationReference | undefined;
  private targetReference: AnnotationReference | undefined;
  private resultReference: AnnotationReference | undefined;
  private readonly programmaticDeletionIds = new Set<string>();
  private readonly programmaticUpdateIds = new Set<string>();
  private disposing = false;

  constructor(
    private readonly options: SpatialSkeletonFindPathAnnotationControllerOptions,
  ) {
    super();

    const { layer, loadedSubsource, state } = options;
    const annotationSource = new LocalAnnotationSource(
      loadedSubsource.loadedDataSource.transform,
      new WatchableValue([]),
      [ASSOCIATED_SEGMENTS_RELATIONSHIP],
    );
    this.annotationSource = annotationSource;

    const displayState = new AnnotationDisplayState();
    displayState.color.value.set([1, 1, 1]);
    displayState.shader.value = SPATIAL_SKELETON_FIND_PATH_SHADER;
    displayState.relationshipStates.set(ASSOCIATED_SEGMENTS_RELATIONSHIP, {
      segmentationState: new WatchableValue(layer.displayState),
      showMatches: new TrackableBoolean(false),
    });

    const annotationState = new AnnotationLayerState({
      localPosition: layer.localPosition,
      transform: loadedSubsource.getRenderLayerTransform(),
      source: annotationSource,
      displayState,
      dataSource: loadedSubsource.loadedDataSource.layerDataSource,
      subsourceIndex: loadedSubsource.subsourceIndex,
      subsourceId: loadedSubsource.subsourceEntry.id,
      subsubsourceId: SPATIAL_SKELETON_FIND_PATH_SUBSOURCE_ID,
      role: RenderLayerRole.ANNOTATION,
    });
    // AnnotationLayerState does not otherwise own its independent display
    // state. Tie their lifetimes together so render layers cannot outlive it.
    annotationState.registerDisposer(displayState);
    this.annotationState = this.registerDisposer(annotationState);

    layer.addAnnotationLayerState(annotationState, loadedSubsource);

    this.registerDisposer(
      annotationSource.childDeleted.add((annotationId) => {
        this.handleAnnotationDeleted(annotationId);
      }),
    );
    this.registerDisposer(
      annotationSource.childUpdated.add((annotation) => {
        this.handleAnnotationUpdated(annotation);
      }),
    );
    this.registerDisposer(state.changed.add(() => this.synchronize()));
    this.synchronize();
  }

  private getReference(kind: AnnotationKind) {
    switch (kind) {
      case "source":
        return this.sourceReference;
      case "target":
        return this.targetReference;
      case "result":
        return this.resultReference;
    }
  }

  private setReference(
    kind: AnnotationKind,
    reference: AnnotationReference | undefined,
  ) {
    switch (kind) {
      case "source":
        this.sourceReference = reference;
        break;
      case "target":
        this.targetReference = reference;
        break;
      case "result":
        this.resultReference = reference;
        break;
    }
  }

  private removeAnnotation(kind: AnnotationKind) {
    const reference = this.getReference(kind);
    if (reference === undefined) return;
    this.programmaticDeletionIds.add(reference.id);
    try {
      this.annotationSource.delete(reference);
    } finally {
      this.programmaticDeletionIds.delete(reference.id);
    }
    this.setReference(kind, undefined);
    reference.dispose();
  }

  private handleAnnotationDeleted(annotationId: string) {
    if (this.disposing || this.programmaticDeletionIds.has(annotationId)) {
      return;
    }

    let kind: AnnotationKind | undefined;
    for (const candidate of ["source", "target", "result"] as const) {
      if (this.getReference(candidate)?.id === annotationId) {
        kind = candidate;
        break;
      }
    }
    if (kind === undefined) return;

    const reference = this.getReference(kind)!;
    this.setReference(kind, undefined);
    reference.dispose();

    const { state } = this.options;
    switch (kind) {
      case "source":
        state.setSource(undefined);
        break;
      case "target":
        state.setTarget(undefined);
        break;
      case "result":
        state.invalidateResult();
        break;
    }
  }

  private handleAnnotationUpdated(annotation: Annotation) {
    if (
      this.disposing ||
      this.programmaticUpdateIds.has(annotation.id) ||
      !(["source", "target", "result"] as const).some(
        (kind) => this.getReference(kind)?.id === annotation.id,
      )
    ) {
      return;
    }

    // These annotations are projections of the persisted find-path state,
    // rather than another editable copy of it.  Annotation editing tools may
    // still update this local source directly, so immediately restore the
    // canonical value.  synchronizeAnnotationUpdate suppresses the resulting
    // childUpdated notification to avoid a feedback loop.
    this.synchronize();
  }

  private synchronizeAnnotationUpdate(
    reference: AnnotationReference,
    annotation: Annotation,
  ) {
    this.programmaticUpdateIds.add(reference.id);
    try {
      this.annotationSource.update(reference, annotation);
    } finally {
      this.programmaticUpdateIds.delete(reference.id);
    }
  }

  private synchronizeEndpoint(
    kind: "source" | "target",
    endpoint: SkeletonFindPathEndpoint | undefined,
    description: string,
  ) {
    if (endpoint === undefined) {
      this.removeAnnotation(kind);
      return;
    }

    let reference = this.getReference(kind);
    const current = reference?.value;
    if (
      current?.type === AnnotationType.POINT &&
      current.description === description &&
      current.properties.length === 0 &&
      positionsEqual(current.point, endpoint.position) &&
      hasRelatedSegment(current.relatedSegments, endpoint.segmentId)
    ) {
      return;
    }

    const annotation: Point = {
      id: reference?.id ?? "",
      type: AnnotationType.POINT,
      point: new Float32Array(endpoint.position),
      properties: [],
      relatedSegments: relatedSegmentsFor(endpoint.segmentId),
      description,
    };
    if (reference === undefined) {
      reference = this.annotationSource.add(annotation);
      this.setReference(kind, reference);
    } else {
      this.synchronizeAnnotationUpdate(reference, annotation);
    }
  }

  private synchronizeResult(
    result: readonly SkeletonFindPathResultNode[] | undefined,
    segmentId: bigint | undefined,
  ) {
    // A polyline needs at least one edge, and an associated segment is needed
    // for the layer relationship filter.
    if (result === undefined || result.length < 2 || segmentId === undefined) {
      this.removeAnnotation("result");
      return;
    }

    let reference = this.resultReference;
    const current = reference?.value;
    if (
      current?.type === AnnotationType.POLYLINE &&
      current.description === SPATIAL_SKELETON_FIND_PATH_RESULT_DESCRIPTION &&
      current.properties.length === 0 &&
      current.points.length === result.length &&
      current.points.every((point, index) =>
        positionsEqual(point, result[index].position),
      ) &&
      hasRelatedSegment(current.relatedSegments, segmentId)
    ) {
      return;
    }

    const annotation: PolyLine = {
      id: reference?.id ?? "",
      type: AnnotationType.POLYLINE,
      points: result.map((node) => new Float32Array(node.position)),
      properties: [],
      relatedSegments: relatedSegmentsFor(segmentId),
      description: SPATIAL_SKELETON_FIND_PATH_RESULT_DESCRIPTION,
    };
    if (reference === undefined) {
      reference = this.annotationSource.add(annotation);
      this.resultReference = reference;
    } else {
      this.synchronizeAnnotationUpdate(reference, annotation);
    }
  }

  private synchronize() {
    if (this.disposing) return;
    const { state } = this.options;
    const { source, target } = state;
    this.synchronizeEndpoint(
      "source",
      source,
      SPATIAL_SKELETON_FIND_PATH_SOURCE_DESCRIPTION,
    );
    this.synchronizeEndpoint(
      "target",
      target,
      SPATIAL_SKELETON_FIND_PATH_TARGET_DESCRIPTION,
    );
    this.synchronizeResult(
      state.result,
      source?.segmentId ?? target?.segmentId,
    );
  }

  disposed() {
    this.disposing = true;
    const { state } = this.options;
    if (state.wasDisposed !== true) {
      state.invalidatePendingRequest();
    }
    for (const kind of ["source", "target", "result"] as const) {
      const reference = this.getReference(kind);
      this.setReference(kind, undefined);
      reference?.dispose();
    }
    super.disposed();
  }
}
