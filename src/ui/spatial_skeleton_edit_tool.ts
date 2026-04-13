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

import "#src/ui/spatial_skeleton_edit_tool.css";

import type { SegmentationUserLayer } from "#src/layer/segmentation/index.js";
import {
  executeSpatialSkeletonAddNode,
  executeSpatialSkeletonMerge,
  executeSpatialSkeletonMoveNode,
  executeSpatialSkeletonSplit,
} from "#src/layer/segmentation/spatial_skeleton_commands.js";
import {
  getSpatialSkeletonSegmentIdFromLayerSelectionState,
  hasSpatialSkeletonNodeSelection,
} from "#src/layer/segmentation/selection.js";
import { showSpatialSkeletonActionError } from "#src/layer/segmentation/spatial_skeleton_errors.js";
import { RenderedDataPanel } from "#src/rendered_data_panel.js";
import {
  addSegmentToVisibleSets,
  getVisibleSegments,
  removeSegmentFromVisibleSets,
} from "#src/segmentation_display_state/base.js";
import { setSpatialSkeletonModesToLinesAndPoints } from "#src/skeleton/edit_mode_rendering.js";
import type { SpatiallyIndexedSkeletonLayer } from "#src/skeleton/frontend.js";
import {
  PerspectiveViewSpatiallyIndexedSkeletonLayer,
  SliceViewPanelSpatiallyIndexedSkeletonLayer,
  SliceViewSpatiallyIndexedSkeletonLayer,
} from "#src/skeleton/frontend.js";
import { hasSpatialSkeletonTrueEndLabel } from "#src/skeleton/node_types.js";
import type { SpatiallyIndexedSkeletonSourceCapability } from "#src/skeleton/state.js";
import { StatusMessage } from "#src/status.js";
import type { ToolActivation } from "#src/ui/tool.js";
import {
  LayerTool,
  makeToolActivationStatusMessageWithHeader,
  registerTool,
} from "#src/ui/tool.js";
import type { SpatialSkeletonToolPointInfo } from "#src/ui/spatial_skeleton_tool_messages.js";
import {
  SPATIAL_SKELETON_SPLIT_BANNER_MESSAGE,
  getSpatialSkeletonEditBannerMessage,
  getSpatialSkeletonMergeBannerMessage,
  getSpatialSkeletonToolPointStatusFields,
} from "#src/ui/spatial_skeleton_tool_messages.js";
import type { ActionEvent } from "#src/util/event_action_map.js";
import { EventActionMap } from "#src/util/event_action_map.js";
import { removeChildren } from "#src/util/dom.js";
import type { vec3 } from "#src/util/geom.js";
import { startRelativeMouseDrag } from "#src/util/mouse_drag.js";

export const SPATIAL_SKELETON_EDIT_MODE_TOOL_ID = "spatialSkeletonEditMode";
export const SPATIAL_SKELETON_MERGE_MODE_TOOL_ID = "spatialSkeletonMergeMode";
export const SPATIAL_SKELETON_SPLIT_MODE_TOOL_ID = "spatialSkeletonSplitMode";

const SPATIAL_SKELETON_EDIT_STATUS_INPUT_EVENT_MAP = EventActionMap.fromObject({
  // Only expose the primary edit actions in the auto-generated subtitle.
  "at:control+mousedown0": "spatial-skeleton-add-node",
  "at:alt+mousedown0": "spatial-skeleton-move-node",
  "at:control+mousedown2": {
    action: "spatial-skeleton-pin-node",
    stopPropagation: true,
    preventDefault: true,
  },
});

const SPATIAL_SKELETON_EDIT_AUX_INPUT_EVENT_MAP = EventActionMap.fromObject({
  "at:dblclick0": {
    action: "spatial-skeleton-toggle-visible",
    stopPropagation: true,
    preventDefault: true,
  },
  "at:shift+control+mousedown2": {
    action: "spatial-skeleton-clear-node-selection",
    stopPropagation: true,
    preventDefault: true,
  },
});

const SPATIAL_SKELETON_PICK_INPUT_EVENT_MAP = EventActionMap.fromObject({
  "at:control+mousedown2": {
    action: "spatial-skeleton-pick-node",
    stopPropagation: true,
    preventDefault: true,
  },
});

const SPATIAL_SKELETON_PICK_AUX_INPUT_EVENT_MAP = EventActionMap.fromObject({
  "at:dblclick0": {
    action: "spatial-skeleton-toggle-visible",
    stopPropagation: true,
    preventDefault: true,
  },
  "at:shift+control+mousedown2": {
    action: "spatial-skeleton-clear-node-selection",
    stopPropagation: true,
    preventDefault: true,
  },
});

const DRAG_START_DISTANCE_PX = 4;

function renderSpatialSkeletonToolStatus(
  body: HTMLElement,
  options: {
    message: string;
    point?: SpatialSkeletonToolPointInfo;
  },
) {
  removeChildren(body);
  body.classList.add("neuroglancer-spatial-skeleton-tool-status");
  const message = document.createElement("span");
  message.className = "neuroglancer-spatial-skeleton-tool-status-message";
  message.textContent = options.message;
  body.appendChild(message);
  if (options.point === undefined) {
    return;
  }
  const point = document.createElement("span");
  point.className = "neuroglancer-spatial-skeleton-tool-status-point";
  for (const field of getSpatialSkeletonToolPointStatusFields(options.point)) {
    const fieldElement = document.createElement("span");
    fieldElement.className =
      "neuroglancer-spatial-skeleton-tool-status-point-field";
    const label = document.createElement("span");
    label.className =
      "neuroglancer-spatial-skeleton-tool-status-point-field-label";
    label.textContent = field.label;
    fieldElement.appendChild(label);
    const value = document.createElement("span");
    value.className =
      "neuroglancer-spatial-skeleton-tool-status-point-field-value";
    value.textContent = field.value;
    fieldElement.appendChild(value);
    point.appendChild(fieldElement);
  }
  body.appendChild(point);
}

abstract class SpatialSkeletonToolBase extends LayerTool<SegmentationUserLayer> {
  constructor(layer: SegmentationUserLayer) {
    super(layer, true);
  }

  protected getActiveSpatiallyIndexedSkeletonLayer() {
    const pickedLayer = this.mouseState.pickedRenderLayer;
    if (pickedLayer instanceof PerspectiveViewSpatiallyIndexedSkeletonLayer) {
      return pickedLayer.base;
    }
    if (pickedLayer instanceof SliceViewPanelSpatiallyIndexedSkeletonLayer) {
      return pickedLayer.base;
    }
    if (pickedLayer instanceof SliceViewSpatiallyIndexedSkeletonLayer) {
      return pickedLayer.base;
    }
    return this.layer.getSpatiallyIndexedSkeletonLayer();
  }

  protected getPickedSpatialSkeletonNode():
    | {
        nodeId: number;
        segmentId?: number;
      }
    | undefined {
    if (!this.mouseState.updateUnconditionally() || !this.mouseState.active) {
      return undefined;
    }
    const nodeIdRaw = this.mouseState.pickedSpatialSkeletonNodeId;
    if (
      typeof nodeIdRaw !== "number" ||
      !Number.isSafeInteger(nodeIdRaw) ||
      nodeIdRaw <= 0
    ) {
      return undefined;
    }
    const nodeId = nodeIdRaw;
    const segmentIdRaw = this.mouseState.pickedSpatialSkeletonSegmentId;
    return {
      nodeId,
      segmentId:
        typeof segmentIdRaw === "number" && Number.isSafeInteger(segmentIdRaw)
          ? segmentIdRaw
          : undefined,
    };
  }

  protected getPickedSpatialSkeletonSegment() {
    if (!this.mouseState.updateUnconditionally() || !this.mouseState.active) {
      return undefined;
    }
    const segmentIdRaw = this.mouseState.pickedSpatialSkeletonSegmentId;
    if (
      typeof segmentIdRaw !== "number" ||
      !Number.isSafeInteger(segmentIdRaw) ||
      segmentIdRaw <= 0
    ) {
      return undefined;
    }
    return segmentIdRaw;
  }

  protected selectSegmentByNumber(value: number) {
    if (!Number.isFinite(value)) return;
    this.layer.selectSegment(BigInt(Math.round(value)), false);
  }

  protected pinSegmentByNumber(value: number) {
    if (!Number.isFinite(value)) return;
    this.layer.selectSegment(BigInt(Math.round(value)), true);
  }

  protected ensureSegmentVisibleByNumber(value: number) {
    if (!Number.isFinite(value)) return;
    addSegmentToVisibleSets(
      this.layer.displayState.segmentationGroupState.value,
      BigInt(Math.round(value)),
    );
  }

  protected removeVisibleSegmentByNumber(
    value: number,
    options: {
      deselect?: boolean;
    } = {},
  ) {
    if (!Number.isFinite(value)) return;
    removeSegmentFromVisibleSets(
      this.layer.displayState.segmentationGroupState.value,
      BigInt(Math.round(value)),
      options,
    );
  }

  protected isSpatialSkeletonSegmentVisible(segmentId: number) {
    return getVisibleSegments(
      this.layer.displayState.segmentationGroupState.value,
    ).has(BigInt(Math.round(segmentId)));
  }

  protected describeVisibleSegmentRequirement(segmentId: number) {
    return `Only visible skeletons are editable. Make skeleton ${segmentId} visible in Seg tab or by double-clicking it in the viewer.`;
  }

  protected togglePickedSpatialSkeletonVisibility() {
    const pickedSegmentId = this.getPickedSpatialSkeletonSegment();
    if (pickedSegmentId === undefined) {
      return false;
    }
    const skeletonLayer = this.layer.getSpatiallyIndexedSkeletonLayer();
    const isVisible = this.isSpatialSkeletonSegmentVisible(pickedSegmentId);
    if (isVisible) {
      this.removeVisibleSegmentByNumber(pickedSegmentId, { deselect: true });
      const selectedNodeId = this.layer.selectedSpatialSkeletonNodeId.value;
      const selectedNode =
        selectedNodeId === undefined
          ? undefined
          : skeletonLayer?.getNode(selectedNodeId);
      if (selectedNode?.segmentId === pickedSegmentId) {
        this.layer.clearSpatialSkeletonNodeSelection(false);
      }
      const mergeAnchorNodeId =
        this.layer.spatialSkeletonState.mergeAnchorNodeId.value;
      const anchorSegmentId =
        mergeAnchorNodeId === undefined
          ? undefined
          : skeletonLayer?.getNode(mergeAnchorNodeId)?.segmentId ??
            this.layer.spatialSkeletonState.getCachedNode(mergeAnchorNodeId)
              ?.segmentId;
      if (anchorSegmentId === pickedSegmentId) {
        this.layer.clearSpatialSkeletonMergeAnchor();
      }
      const cachedSegmentIds = new Set<number>(
        [
          ...getVisibleSegments(
            this.layer.displayState.segmentationGroupState.value,
          ).keys(),
        ]
          .map((segmentId) => Number(segmentId))
          .filter(
            (segmentId) => Number.isSafeInteger(segmentId) && segmentId > 0,
          ),
      );
      for (const retainedSegmentId of skeletonLayer?.getRetainedOverlaySegmentIds() ??
        []) {
        cachedSegmentIds.add(retainedSegmentId);
      }
      this.layer.spatialSkeletonState.evictInactiveSegmentNodes(
        cachedSegmentIds,
      );
      StatusMessage.showTemporaryMessage(
        `Removed skeleton ${pickedSegmentId} from visible/editable skeletons.`,
      );
      return true;
    }
    this.ensureSegmentVisibleByNumber(pickedSegmentId);
    this.selectSegmentByNumber(pickedSegmentId);
    StatusMessage.showTemporaryMessage(
      `Made skeleton ${pickedSegmentId} visible/editable.`,
    );
    return true;
  }

  protected bindVisibilityToggleAction(activation: ToolActivation<this>) {
    activation.bindAction(
      "spatial-skeleton-toggle-visible",
      (event: ActionEvent<MouseEvent>) => {
        if (event.detail.button !== 0) return;
        event.stopPropagation();
        event.detail.preventDefault();
        this.togglePickedSpatialSkeletonVisibility();
      },
    );
  }

  protected moveViewToNodePosition(position: ArrayLike<number>) {
    const globalPosition = this.layer.manager.root.globalPosition;
    const nextGlobal = globalPosition.value.slice();
    const globalRank = Math.min(nextGlobal.length, 3);
    for (let i = 0; i < globalRank; ++i) {
      const value = Number(position[i]);
      if (Number.isFinite(value)) {
        nextGlobal[i] = value;
      }
    }
    globalPosition.value = nextGlobal;

    const localPosition = this.layer.localPosition;
    const nextLocal = localPosition.value.slice();
    const localRank = Math.min(nextLocal.length, 3);
    for (let i = 0; i < localRank; ++i) {
      const value = Number(position[i]);
      if (Number.isFinite(value)) {
        nextLocal[i] = value;
      }
    }
    localPosition.value = nextLocal;
  }

  protected resolvePickedNodeForAction(
    skeletonLayer: SpatiallyIndexedSkeletonLayer,
  ) {
    const pickedNode = this.resolvePickedNodeSelection(skeletonLayer);
    if (pickedNode === undefined) {
      return undefined;
    }
    if (pickedNode.segmentId !== undefined) {
      this.selectSegmentByNumber(pickedNode.segmentId);
    }
    this.layer.selectSpatialSkeletonNode(pickedNode.nodeId, false, {
      segmentId: pickedNode.segmentId,
      position: pickedNode.position,
    });
    return {
      nodeId: pickedNode.nodeId,
      segmentId: pickedNode.segmentId,
    };
  }

  protected resolvePickedNodeSelection(
    skeletonLayer: SpatiallyIndexedSkeletonLayer,
  ) {
    const nodeHit = this.getPickedSpatialSkeletonNode();
    if (nodeHit === undefined) {
      return undefined;
    }
    const resolvedNodeInfo =
      skeletonLayer.getNode(nodeHit.nodeId);
    return {
      nodeId: nodeHit.nodeId,
      segmentId: nodeHit.segmentId ?? resolvedNodeInfo?.segmentId,
      position: resolvedNodeInfo?.position,
    };
  }

  protected getSelectedSpatialSkeletonNodeSummary() {
    const nodeId = this.layer.selectedSpatialSkeletonNodeId.value;
    if (nodeId === undefined) {
      return undefined;
    }
    const selectedNode =
      this.getActiveSpatiallyIndexedSkeletonLayer()?.getNode(nodeId);
    const layerSelectionState =
      this.layer.manager.root.selectionState.value?.layers.find(
        (entry) => entry.layer === this.layer,
      )?.state;
    return {
      nodeId,
      segmentId:
        selectedNode?.segmentId ??
        getSpatialSkeletonSegmentIdFromLayerSelectionState(layerSelectionState),
    };
  }

  protected bindPinnedSelectionAction(
    activation: ToolActivation<this>,
    options: {
      showNodeSelectionMessage?: boolean;
    } = {},
  ) {
    const { showNodeSelectionMessage = true } = options;
    activation.bindAction(
      "spatial-skeleton-pin-node",
      (event: ActionEvent<MouseEvent>) => {
        if (
          event.detail.button !== 2 ||
          !event.detail.ctrlKey ||
          event.detail.shiftKey
        ) {
          return;
        }
        event.stopPropagation();
        event.detail.preventDefault();
        const skeletonLayer = this.getActiveSpatiallyIndexedSkeletonLayer();
        if (skeletonLayer === undefined) {
          return;
        }
        const pickedNode = this.resolvePickedNodeSelection(skeletonLayer);
        if (pickedNode === undefined) {
          const pickedSegmentId = this.getPickedSpatialSkeletonSegment();
          if (pickedSegmentId === undefined) {
            return;
          }
          this.layer.clearSpatialSkeletonNodeSelection(false);
          this.pinSegmentByNumber(pickedSegmentId);
          return;
        }
        if (pickedNode.segmentId !== undefined) {
          this.pinSegmentByNumber(pickedNode.segmentId);
        }
        this.layer.selectSpatialSkeletonNode(pickedNode.nodeId, true, {
          segmentId: pickedNode.segmentId,
          position: pickedNode.position,
        });
        if (showNodeSelectionMessage) {
          StatusMessage.showTemporaryMessage(
            `Selected and pinned node ${pickedNode.nodeId}.`,
          );
        }
      },
    );
  }

  protected bindClearSelectionAction(activation: ToolActivation<this>) {
    activation.bindAction(
      "spatial-skeleton-clear-node-selection",
      (event: ActionEvent<MouseEvent>) => {
        if (
          event.detail.button !== 2 ||
          !event.detail.ctrlKey ||
          !event.detail.shiftKey
        ) {
          return;
        }
        event.stopPropagation();
        event.detail.preventDefault();
        const pinnedSelection = this.layer.manager.root.selectionState.value;
        const hasSpatialSkeletonSelection =
          this.layer.selectedSpatialSkeletonNodeId.value !== undefined ||
          (pinnedSelection?.layers.some(
            ({ layer, state }) =>
              layer === this.layer && hasSpatialSkeletonNodeSelection(state),
          ) ??
            false);
        if (hasSpatialSkeletonSelection) {
          this.layer.clearSpatialSkeletonNodeSelection("force-unpin");
          StatusMessage.showTemporaryMessage(
            "Spatial skeleton node selection cleared.",
          );
          return;
        }
        this.layer.manager.root.selectionState.unpin();
      },
    );
  }

  protected activateModeWatchable(
    activation: ToolActivation<this>,
    modeWatchable: { value: boolean },
  ) {
    setSpatialSkeletonModesToLinesAndPoints(this.layer);
    modeWatchable.value = true;
    activation.registerDisposer(() => {
      modeWatchable.value = false;
    });
  }

  protected registerAutoCancelOnDisabled(
    activation: ToolActivation<this>,
    requiredCapabilities:
      | SpatiallyIndexedSkeletonSourceCapability
      | readonly SpatiallyIndexedSkeletonSourceCapability[],
    onReady?: () => void,
  ) {
    const handleStateChanged = () => {
      const disabledReason =
        this.layer.getSpatialSkeletonActionsDisabledReason(
          requiredCapabilities,
        );
      if (disabledReason === undefined) {
        onReady?.();
        return;
      }
      StatusMessage.showTemporaryMessage(disabledReason);
      activation.cancel();
    };
    activation.registerDisposer(
      this.layer.spatialSkeletonActionsAllowed.changed.add(handleStateChanged),
    );
    activation.registerDisposer(
      this.layer.spatialSkeletonSourceCapabilities.changed.add(
        handleStateChanged,
      ),
    );
  }

}

export class SpatialSkeletonEditModeTool extends SpatialSkeletonToolBase {
  toJSON() {
    return SPATIAL_SKELETON_EDIT_MODE_TOOL_ID;
  }

  get description() {
    return "skeleton edit mode";
  }

  private getMousePosition() {
    if (!this.mouseState.updateUnconditionally() || !this.mouseState.active) {
      return undefined;
    }
    const pos = this.mouseState.unsnappedPosition;
    if (pos.length < 3) return undefined;
    const x = Number(pos[0]);
    const y = Number(pos[1]);
    const z = Number(pos[2]);
    if (!Number.isFinite(x) || !Number.isFinite(y) || !Number.isFinite(z)) {
      return undefined;
    }
    return new Float32Array([x, y, z]);
  }

  private getSelectedParentNodeForAdd(
    skeletonLayer: SpatiallyIndexedSkeletonLayer,
    parentNodeId: number | undefined,
  ) {
    if (parentNodeId === undefined) {
      return undefined;
    }
    return (
      this.layer.spatialSkeletonState.getCachedNode(parentNodeId) ??
      skeletonLayer.getNode(parentNodeId)
    );
  }

  private getAddNodeBlockedReason(
    skeletonLayer: SpatiallyIndexedSkeletonLayer,
    parentNodeId: number | undefined,
  ) {
    if (parentNodeId === undefined) {
      return undefined;
    }
    const selectedParentNode = this.getSelectedParentNodeForAdd(
      skeletonLayer,
      parentNodeId,
    );
    if (
      selectedParentNode !== undefined &&
      hasSpatialSkeletonTrueEndLabel(selectedParentNode.labels)
    ) {
      return `Node ${parentNodeId} is marked as a true end. Remove the true end label before appending a child node.`;
    }
    return undefined;
  }

  private getRenderedDataPanelForEvent(
    event: MouseEvent,
  ): RenderedDataPanel | undefined {
    const display = this.layer.manager.root.display;
    const target = event.target;
    if (target instanceof Node) {
      for (const panel of display.panels) {
        if (!(panel instanceof RenderedDataPanel)) continue;
        if (panel.element.contains(target)) {
          return panel;
        }
      }
    }
    const clientX = event.clientX;
    const clientY = event.clientY;
    for (const panel of display.panels) {
      if (!(panel instanceof RenderedDataPanel)) continue;
      const rect = panel.element.getBoundingClientRect();
      if (
        clientX >= rect.left &&
        clientX <= rect.right &&
        clientY >= rect.top &&
        clientY <= rect.bottom
      ) {
        return panel;
      }
    }
    return undefined;
  }

  activate(activation: ToolActivation<this>) {
    const { layer } = this;
    const rawInputEventMapBinder = activation.inputEventMapBinder;
    const { body, header } =
      makeToolActivationStatusMessageWithHeader(activation);
    header.textContent = "Spatial skeleton edit mode";
    let statusOverride: string | undefined;
    const renderStatus = () => {
      const selectedPoint = this.getSelectedSpatialSkeletonNodeSummary();
      renderSpatialSkeletonToolStatus(body, {
        message:
          statusOverride ?? getSpatialSkeletonEditBannerMessage(selectedPoint),
        point: selectedPoint,
      });
    };
    const setStatus = (nextStatus: string | undefined) => {
      statusOverride = nextStatus;
      renderStatus();
    };
    const setReadyStatus = () => {
      setStatus(undefined);
    };

    const disableWithMessage = (message: string) => {
      setStatus(message);
      StatusMessage.showTemporaryMessage(message);
      queueMicrotask(() => activation.cancel());
    };

    const getEditCapabilityDisabledReason = () =>
      layer.getSpatialSkeletonActionsDisabledReason(["addNodes", "moveNodes"], {
        requireVisibleChunks: false,
      });
    const getEditMutationDisabledReason = () =>
      layer.getSpatialSkeletonActionsDisabledReason(["addNodes", "moveNodes"]);
    const updateInteractionStatus = () => {
      const reason = getEditMutationDisabledReason();
      if (reason === undefined) {
        setReadyStatus();
        return undefined;
      }
      const message = `${reason} Node selection is still available.`;
      setStatus(message);
      return reason;
    };

    const disabledReason = getEditCapabilityDisabledReason();
    if (disabledReason !== undefined) {
      disableWithMessage(disabledReason);
      return;
    }
    if (this.getActiveSpatiallyIndexedSkeletonLayer() === undefined) {
      disableWithMessage(
        "No spatially indexed skeleton source is currently loaded.",
      );
      return;
    }

    this.activateModeWatchable(activation, layer.spatialSkeletonEditMode);
    activation.bindInputEventMap(SPATIAL_SKELETON_EDIT_STATUS_INPUT_EVENT_MAP);
    rawInputEventMapBinder(
      SPATIAL_SKELETON_EDIT_AUX_INPUT_EVENT_MAP,
      activation,
    );
    this.bindPinnedSelectionAction(activation, {
      showNodeSelectionMessage: false,
    });
    this.bindClearSelectionAction(activation);
    this.bindVisibilityToggleAction(activation);
    updateInteractionStatus();
    activation.registerDisposer(() => {
      layer.spatialSkeletonState.clearPendingNodePositions();
      layer.clearSpatialSkeletonNodeSelection(false);
    });
    activation.registerDisposer(
      layer.selectedSpatialSkeletonNodeId.changed.add(renderStatus),
    );
    activation.registerDisposer(
      layer.manager.root.selectionState.changed.add(renderStatus),
    );
    activation.registerDisposer(
      layer.spatialSkeletonActionsAllowed.changed.add(() => {
        if (!layer.spatialSkeletonActionsAllowed.value) {
          const capabilityReason = getEditCapabilityDisabledReason();
          if (capabilityReason !== undefined) {
            StatusMessage.showTemporaryMessage(capabilityReason);
            activation.cancel();
            return;
          }
          const reason = updateInteractionStatus();
          if (reason !== undefined) {
            StatusMessage.showTemporaryMessage(reason);
            return;
          }
        }
        setReadyStatus();
      }),
    );
    activation.registerDisposer(
      layer.spatialSkeletonSourceCapabilities.changed.add(() => {
        const capabilityReason = getEditCapabilityDisabledReason();
        if (capabilityReason !== undefined) {
          StatusMessage.showTemporaryMessage(capabilityReason);
          activation.cancel();
          return;
        }
        const reason = updateInteractionStatus();
        if (reason !== undefined) {
          StatusMessage.showTemporaryMessage(reason);
          return;
        }
        setReadyStatus();
      }),
    );

    activation.bindAction(
      "spatial-skeleton-add-node",
      (event: ActionEvent<MouseEvent>) => {
        if (
          event.detail.button !== 0 ||
          !event.detail.ctrlKey ||
          event.detail.shiftKey ||
          event.detail.altKey ||
          event.detail.metaKey
        ) {
          return;
        }
        event.stopPropagation();
        event.detail.preventDefault();
        const disabledReason =
          layer.getSpatialSkeletonActionsDisabledReason("addNodes");
        if (disabledReason !== undefined) {
          StatusMessage.showTemporaryMessage(disabledReason);
          return;
        }
        const skeletonLayer = this.getActiveSpatiallyIndexedSkeletonLayer();
        if (skeletonLayer === undefined) {
          StatusMessage.showTemporaryMessage(
            "No spatially indexed skeleton source is currently loaded.",
          );
          return;
        }
        const selectedParentNodeId = layer.selectedSpatialSkeletonNodeId.value;
        const addNodeBlockedReason = this.getAddNodeBlockedReason(
          skeletonLayer,
          selectedParentNodeId,
        );
        if (addNodeBlockedReason !== undefined) {
          StatusMessage.showTemporaryMessage(addNodeBlockedReason);
          return;
        }
        if (selectedParentNodeId === undefined) {
          const pickedSegmentId = this.getPickedSpatialSkeletonSegment();
          if (pickedSegmentId !== undefined) {
            this.selectSegmentByNumber(pickedSegmentId);
            return;
          }
        }
        const clickStartPosition = this.getMousePosition();
        if (clickStartPosition === undefined) {
          StatusMessage.showTemporaryMessage(
            "Unable to resolve add-node position for this click.",
          );
          return;
        }
        let dragDistanceSquared = 0;
        startRelativeMouseDrag(
          event.detail,
          (_event, deltaX, deltaY) => {
            dragDistanceSquared += deltaX * deltaX + deltaY * deltaY;
          },
          (_finishEvent) => {
            const thresholdSquared =
              DRAG_START_DISTANCE_PX * DRAG_START_DISTANCE_PX;
            if (dragDistanceSquared > thresholdSquared) {
              setReadyStatus();
              return;
            }
            const selectedParentNodeId =
              layer.selectedSpatialSkeletonNodeId.value;
            const addNodeBlockedReason = this.getAddNodeBlockedReason(
              skeletonLayer,
              selectedParentNodeId,
            );
            if (addNodeBlockedReason !== undefined) {
              setReadyStatus();
              StatusMessage.showTemporaryMessage(addNodeBlockedReason);
              return;
            }
            const selectedParentNode = this.getSelectedParentNodeForAdd(
              skeletonLayer,
              selectedParentNodeId,
            );
            const targetSkeletonId =
              selectedParentNode === undefined
                ? 0
                : selectedParentNode.segmentId;
            const clickPosition =
              this.getMousePosition() ?? new Float32Array(clickStartPosition);
            void (async () => {
              try {
                await executeSpatialSkeletonAddNode(layer, {
                  skeletonId: targetSkeletonId,
                  parentNodeId: selectedParentNodeId,
                  position: clickPosition,
                });
              } catch (error) {
                showSpatialSkeletonActionError("create node", error);
                return;
              }
              setReadyStatus();
            })();
          },
        );
      },
    );

    activation.bindAction(
      "spatial-skeleton-move-node",
      (event: ActionEvent<MouseEvent>) => {
        event.stopPropagation();
        event.detail.preventDefault();
        const disabledReason =
          layer.getSpatialSkeletonActionsDisabledReason("moveNodes");
        if (disabledReason !== undefined) {
          StatusMessage.showTemporaryMessage(disabledReason);
          return;
        }
        const skeletonLayer = this.getActiveSpatiallyIndexedSkeletonLayer();
        if (skeletonLayer === undefined) {
          StatusMessage.showTemporaryMessage(
            "No spatially indexed skeleton source is currently loaded.",
          );
          return;
        }
        const actionPanel = this.getRenderedDataPanelForEvent(event.detail);
        const pickedNode = this.getPickedSpatialSkeletonNode();
        if (pickedNode === undefined) {
          const pickedSegmentId = this.getPickedSpatialSkeletonSegment();
          if (pickedSegmentId !== undefined) {
            this.selectSegmentByNumber(pickedSegmentId);
            layer.clearSpatialSkeletonNodeSelection(false);
          }
          return;
        }
        const pickedPosition = this.mouseState.position;
        const hasPickedPosition =
          pickedPosition.length >= 3 &&
          Number.isFinite(pickedPosition[0]) &&
          Number.isFinite(pickedPosition[1]) &&
          Number.isFinite(pickedPosition[2]);
        const nodeInfo = hasPickedPosition
          ? {
              nodeId: pickedNode.nodeId,
              segmentId: pickedNode.segmentId ?? 0,
              position: new Float32Array([
                Number(pickedPosition[0]),
                Number(pickedPosition[1]),
                Number(pickedPosition[2]),
              ]),
            }
          : skeletonLayer.getNode(pickedNode.nodeId);
        if (nodeInfo === undefined) {
          return;
        }
        const dragPanel = actionPanel;
        if (dragPanel === undefined) {
          StatusMessage.showTemporaryMessage(
            "Unable to resolve active panel for node drag.",
          );
          return;
        }
        let moved = false;
        const lastPosition = new Float32Array(
          nodeInfo.position,
        ) as unknown as vec3;
        const dragAnchorPosition = new Float32Array(
          nodeInfo.position,
        ) as unknown as vec3;
        const panelTranslatedPosition = new Float32Array(
          nodeInfo.position,
        ) as unknown as vec3;
        let moveEvents = 0;
        let totalDeltaX = 0;
        let totalDeltaY = 0;
        let dragDistanceSquared = 0;
        let dragActive = false;
        startRelativeMouseDrag(
          event.detail,
          (_event, deltaX, deltaY) => {
            dragDistanceSquared += deltaX * deltaX + deltaY * deltaY;
            const thresholdSquared =
              DRAG_START_DISTANCE_PX * DRAG_START_DISTANCE_PX;
            if (!dragActive && dragDistanceSquared >= thresholdSquared) {
              dragActive = true;
              setStatus("Dragging node");
            }
            if (!dragActive) return;
            ++moveEvents;
            totalDeltaX += deltaX;
            totalDeltaY += deltaY;
            panelTranslatedPosition[0] = dragAnchorPosition[0];
            panelTranslatedPosition[1] = dragAnchorPosition[1];
            panelTranslatedPosition[2] = dragAnchorPosition[2];
            dragPanel.translateDataPointByViewportPixels(
              panelTranslatedPosition,
              dragAnchorPosition,
              totalDeltaX,
              totalDeltaY,
            );
            if (
              !Number.isFinite(panelTranslatedPosition[0]) ||
              !Number.isFinite(panelTranslatedPosition[1]) ||
              !Number.isFinite(panelTranslatedPosition[2])
            ) {
              return;
            }
            const previewChanged =
              layer.spatialSkeletonState.setPendingNodePosition(
                pickedNode.nodeId,
                panelTranslatedPosition,
              );
            if (!previewChanged) return;
            moved = true;
            lastPosition[0] = panelTranslatedPosition[0];
            lastPosition[1] = panelTranslatedPosition[1];
            lastPosition[2] = panelTranslatedPosition[2];
          },
          (_finishEvent) => {
            if (!dragActive) {
              setReadyStatus();
              return;
            }
            setReadyStatus();
            if (moved) {
              void executeSpatialSkeletonMoveNode(layer, {
                node: nodeInfo,
                nextPosition: new Float32Array(lastPosition),
              })
                .then(() => {
                  layer.spatialSkeletonState.clearPendingNodePosition(
                    pickedNode.nodeId,
                  );
                })
                .catch((error) => {
                  layer.spatialSkeletonState.clearPendingNodePosition(
                    pickedNode.nodeId,
                  );
                  showSpatialSkeletonActionError("move node", error);
                });
              return;
            }
            layer.spatialSkeletonState.clearPendingNodePosition(
              pickedNode.nodeId,
            );
          },
        );
      },
    );
  }
}

class SpatialSkeletonMergeModeTool extends SpatialSkeletonToolBase {
  toJSON() {
    return SPATIAL_SKELETON_MERGE_MODE_TOOL_ID;
  }

  get description() {
    return "skeleton merge mode";
  }

  activate(activation: ToolActivation<this>) {
    const rawInputEventMapBinder = activation.inputEventMapBinder;
    const reason =
      this.layer.getSpatialSkeletonActionsDisabledReason("mergeSkeletons");
    if (reason !== undefined) {
      StatusMessage.showTemporaryMessage(reason);
      queueMicrotask(() => activation.cancel());
      return;
    }
    if (this.getActiveSpatiallyIndexedSkeletonLayer() === undefined) {
      StatusMessage.showTemporaryMessage(
        "No spatially indexed skeleton source is currently loaded.",
      );
      queueMicrotask(() => activation.cancel());
      return;
    }

    this.activateModeWatchable(activation, this.layer.spatialSkeletonMergeMode);
    this.layer.clearSpatialSkeletonNodeSelection("force-unpin");
    this.layer.clearSpatialSkeletonMergeAnchor();
    activation.registerDisposer(() => {
      this.layer.clearSpatialSkeletonMergeAnchor();
    });
    const { body, header } =
      makeToolActivationStatusMessageWithHeader(activation);
    header.textContent = "Spatial skeleton merge mode";
    let pending = false;
    let statusOverride: string | undefined;
    const getAnchorNode = () => {
      const nodeId = this.layer.spatialSkeletonState.mergeAnchorNodeId.value;
      if (nodeId === undefined || !Number.isSafeInteger(nodeId)) {
        return undefined;
      }
      const cachedNode =
        this.getActiveSpatiallyIndexedSkeletonLayer()?.getNode(nodeId) ??
        this.layer.spatialSkeletonState.getCachedNode(nodeId);
      return {
        nodeId,
        segmentId: cachedNode?.segmentId,
        position: cachedNode?.position,
      };
    };
    const renderStatus = () => {
      const anchorNode = getAnchorNode();
      renderSpatialSkeletonToolStatus(body, {
        message:
          statusOverride ?? getSpatialSkeletonMergeBannerMessage(anchorNode),
        point: anchorNode,
      });
    };
    const setStatus = (nextStatus: string | undefined) => {
      statusOverride = nextStatus;
      renderStatus();
    };
    const setReadyStatus = () => {
      setStatus(undefined);
    };
    setReadyStatus();
    activation.bindInputEventMap(SPATIAL_SKELETON_PICK_INPUT_EVENT_MAP);
    rawInputEventMapBinder(
      SPATIAL_SKELETON_PICK_AUX_INPUT_EVENT_MAP,
      activation,
    );
    this.bindClearSelectionAction(activation);
    this.bindVisibilityToggleAction(activation);
    this.registerAutoCancelOnDisabled(
      activation,
      "mergeSkeletons",
      setReadyStatus,
    );
    activation.registerDisposer(
      this.layer.spatialSkeletonState.mergeAnchorNodeId.changed.add(
        renderStatus,
      ),
    );
    activation.bindAction(
      "spatial-skeleton-pick-node",
      (event: ActionEvent<MouseEvent>) => {
        if (
          event.detail.button !== 2 ||
          !event.detail.ctrlKey ||
          event.detail.shiftKey ||
          event.detail.altKey ||
          event.detail.metaKey
        ) {
          return;
        }
        if (pending) return;
        const disabledReason =
          this.layer.getSpatialSkeletonActionsDisabledReason("mergeSkeletons");
        if (disabledReason !== undefined) {
          StatusMessage.showTemporaryMessage(disabledReason);
          return;
        }
        const skeletonLayer = this.getActiveSpatiallyIndexedSkeletonLayer();
        if (skeletonLayer === undefined) {
          StatusMessage.showTemporaryMessage(
            "No spatially indexed skeleton source is currently loaded.",
          );
          return;
        }
        const pickedNode = this.resolvePickedNodeSelection(skeletonLayer);
        const anchorNode = getAnchorNode();
        if (pickedNode === undefined) {
          const pickedSegmentId = this.getPickedSpatialSkeletonSegment();
          if (pickedSegmentId !== undefined) {
            this.pinSegmentByNumber(pickedSegmentId);
            if (
              anchorNode === undefined ||
              pickedSegmentId === anchorNode.segmentId
            ) {
              this.layer.clearSpatialSkeletonNodeSelection(false);
            }
            renderStatus();
          }
          return;
        }
        if (pickedNode === undefined || pickedNode.segmentId === undefined) {
          return;
        }
        this.pinSegmentByNumber(pickedNode.segmentId);
        if (
          anchorNode === undefined ||
          anchorNode.nodeId === pickedNode.nodeId
        ) {
          this.layer.setSpatialSkeletonMergeAnchor(pickedNode.nodeId);
          this.layer.selectSpatialSkeletonNode(pickedNode.nodeId, true, {
            segmentId: pickedNode.segmentId,
            position: pickedNode.position,
          });
          renderStatus();
          return;
        }
        if (anchorNode.segmentId === pickedNode.segmentId) {
          this.layer.setSpatialSkeletonMergeAnchor(pickedNode.nodeId);
          this.layer.selectSpatialSkeletonNode(pickedNode.nodeId, true, {
            segmentId: pickedNode.segmentId,
            position: pickedNode.position,
          });
          renderStatus();
          return;
        }
        const firstNode = anchorNode;
        const secondNode = {
          nodeId: pickedNode.nodeId,
          segmentId: pickedNode.segmentId,
          position: pickedNode.position,
        };
        if (
          firstNode.segmentId === undefined ||
          secondNode.segmentId === undefined
        ) {
          StatusMessage.showTemporaryMessage(
            "Load both skeletons in the Skeleton tab before merging them.",
          );
          return;
        }
        pending = true;
        setStatus("Merging selected nodes.");
        void (async () => {
          try {
            await executeSpatialSkeletonMerge(
              this.layer,
              {
                nodeId: firstNode.nodeId,
                segmentId: firstNode.segmentId!,
              },
              {
                nodeId: secondNode.nodeId,
                segmentId: secondNode.segmentId!,
              },
            );
          } catch (error) {
            showSpatialSkeletonActionError("merge skeletons", error);
          } finally {
            pending = false;
            setReadyStatus();
          }
        })();
      },
    );
  }
}

class SpatialSkeletonSplitModeTool extends SpatialSkeletonToolBase {
  toJSON() {
    return SPATIAL_SKELETON_SPLIT_MODE_TOOL_ID;
  }

  get description() {
    return "skeleton split mode";
  }

  activate(activation: ToolActivation<this>) {
    const rawInputEventMapBinder = activation.inputEventMapBinder;
    const reason =
      this.layer.getSpatialSkeletonActionsDisabledReason("splitSkeletons");
    if (reason !== undefined) {
      StatusMessage.showTemporaryMessage(reason);
      queueMicrotask(() => activation.cancel());
      return;
    }
    if (this.getActiveSpatiallyIndexedSkeletonLayer() === undefined) {
      StatusMessage.showTemporaryMessage(
        "No spatially indexed skeleton source is currently loaded.",
      );
      queueMicrotask(() => activation.cancel());
      return;
    }

    this.activateModeWatchable(activation, this.layer.spatialSkeletonSplitMode);
    this.layer.clearSpatialSkeletonNodeSelection("force-unpin");
    const { body, header } =
      makeToolActivationStatusMessageWithHeader(activation);
    header.textContent = "Spatial skeleton split mode";
    let pending = false;
    let statusOverride: string | undefined;
    let pendingPoint: SpatialSkeletonToolPointInfo | undefined;
    const renderStatus = () => {
      renderSpatialSkeletonToolStatus(body, {
        message: statusOverride ?? SPATIAL_SKELETON_SPLIT_BANNER_MESSAGE,
        point: pendingPoint,
      });
    };
    const setStatus = (
      nextStatus: string | undefined,
      nextPoint: SpatialSkeletonToolPointInfo | undefined = pendingPoint,
    ) => {
      statusOverride = nextStatus;
      pendingPoint = nextPoint;
      renderStatus();
    };
    const setReadyStatus = () => {
      setStatus(undefined, undefined);
    };
    setReadyStatus();
    activation.bindInputEventMap(SPATIAL_SKELETON_PICK_INPUT_EVENT_MAP);
    rawInputEventMapBinder(
      SPATIAL_SKELETON_PICK_AUX_INPUT_EVENT_MAP,
      activation,
    );
    this.bindClearSelectionAction(activation);
    this.bindVisibilityToggleAction(activation);
    this.registerAutoCancelOnDisabled(
      activation,
      "splitSkeletons",
      setReadyStatus,
    );
    activation.bindAction(
      "spatial-skeleton-pick-node",
      (event: ActionEvent<MouseEvent>) => {
        if (
          event.detail.button !== 2 ||
          !event.detail.ctrlKey ||
          event.detail.shiftKey ||
          event.detail.altKey ||
          event.detail.metaKey
        ) {
          return;
        }
        if (pending) return;
        const disabledReason =
          this.layer.getSpatialSkeletonActionsDisabledReason("splitSkeletons");
        if (disabledReason !== undefined) {
          StatusMessage.showTemporaryMessage(disabledReason);
          return;
        }
        const skeletonLayer = this.getActiveSpatiallyIndexedSkeletonLayer();
        if (skeletonLayer === undefined) {
          StatusMessage.showTemporaryMessage(
            "No spatially indexed skeleton source is currently loaded.",
          );
          return;
        }
        const pickedNode = this.resolvePickedNodeSelection(skeletonLayer);
        if (pickedNode === undefined) {
          const pickedSegmentId = this.getPickedSpatialSkeletonSegment();
          if (pickedSegmentId !== undefined) {
            this.pinSegmentByNumber(pickedSegmentId);
            this.layer.clearSpatialSkeletonNodeSelection(false);
            renderStatus();
          }
          return;
        }
        if (pickedNode === undefined || pickedNode.segmentId === undefined) {
          return;
        }
        this.pinSegmentByNumber(pickedNode.segmentId);
        this.layer.selectSpatialSkeletonNode(pickedNode.nodeId, true, {
          segmentId: pickedNode.segmentId,
          position: pickedNode.position,
        });
        const point = {
          nodeId: pickedNode.nodeId,
          segmentId: pickedNode.segmentId,
          position: pickedNode.position,
        };
        pending = true;
        setStatus("Splitting selected node.", point);
        void (async () => {
          try {
            await executeSpatialSkeletonSplit(this.layer, {
              nodeId: pickedNode.nodeId,
              segmentId: pickedNode.segmentId!,
            });
          } catch (error) {
            showSpatialSkeletonActionError("split skeleton", error);
          } finally {
            pending = false;
            setReadyStatus();
          }
        })();
      },
    );
  }
}

export function registerSpatialSkeletonEditModeTool(
  contextType: typeof SegmentationUserLayer,
) {
  registerTool(contextType, SPATIAL_SKELETON_EDIT_MODE_TOOL_ID, (layer) => {
    return new SpatialSkeletonEditModeTool(layer);
  });
  registerTool(contextType, SPATIAL_SKELETON_MERGE_MODE_TOOL_ID, (layer) => {
    return new SpatialSkeletonMergeModeTool(layer);
  });
  registerTool(contextType, SPATIAL_SKELETON_SPLIT_MODE_TOOL_ID, (layer) => {
    return new SpatialSkeletonSplitModeTool(layer);
  });
}
