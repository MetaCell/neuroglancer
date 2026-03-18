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

import type { SegmentationUserLayer } from "#src/layer/segmentation/index.js";
import {
  addSegmentToVisibleSets,
  getVisibleSegments,
  removeSegmentFromVisibleSets,
} from "#src/segmentation_display_state/base.js";
import type { SpatiallyIndexedSkeletonAddNodeResult } from "#src/skeleton/api.js";
import {
  PerspectiveViewSpatiallyIndexedSkeletonLayer,
  SpatiallyIndexedSkeletonLayer,
  SliceViewPanelSpatiallyIndexedSkeletonLayer,
  SliceViewSpatiallyIndexedSkeletonLayer,
} from "#src/skeleton/frontend.js";
import { getEditableSpatiallyIndexedSkeletonSource } from "#src/skeleton/state.js";
import { RenderedDataPanel } from "#src/rendered_data_panel.js";
import { StatusMessage } from "#src/status.js";
import type { ToolActivation } from "#src/ui/tool.js";
import {
  LayerTool,
  makeToolActivationStatusMessageWithHeader,
  registerTool,
} from "#src/ui/tool.js";
import type { ActionEvent } from "#src/util/event_action_map.js";
import { EventActionMap } from "#src/util/event_action_map.js";
import type { vec3 } from "#src/util/geom.js";
import { startRelativeMouseDrag } from "#src/util/mouse_drag.js";
import type { SpatiallyIndexedSkeletonSourceCapability } from "#src/skeleton/state.js";

export const SPATIAL_SKELETON_EDIT_MODE_TOOL_ID = "spatialSkeletonEditMode";
export const SPATIAL_SKELETON_MERGE_MODE_TOOL_ID = "spatialSkeletonMergeMode";
export const SPATIAL_SKELETON_SPLIT_MODE_TOOL_ID = "spatialSkeletonSplitMode";

const SPATIAL_SKELETON_EDIT_INPUT_EVENT_MAP = EventActionMap.fromObject({
  // Use ctrl+click for node creation, ctrl+right-click for node selection,
  // and alt+drag for node movement.
  // Keep plain left-drag/click camera controls available from default bindings.
  "at:control+mousedown0": "spatial-skeleton-add-node",
  "at:alt+mousedown0": "spatial-skeleton-move-node",
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
  "at:control+mousedown2": {
    action: "spatial-skeleton-pin-node",
    stopPropagation: true,
    preventDefault: true,
  },
});

const SPATIAL_SKELETON_PICK_INPUT_EVENT_MAP = EventActionMap.fromObject({
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
  "at:control+mousedown2": {
    action: "spatial-skeleton-pick-node",
    stopPropagation: true,
    preventDefault: true,
  },
});

const DRAG_START_DISTANCE_PX = 4;
const SPATIAL_SKELETON_EDIT_DEBUG_LOGS = true;

function logSpatialSkeletonEdit(
  label: string,
  data: Record<string, unknown> = {},
) {
  if (!SPATIAL_SKELETON_EDIT_DEBUG_LOGS) return;
  console.debug(`[SpatialSkeletonEdit] ${label}`, data);
}

function formatVec3(value: ArrayLike<number> | undefined) {
  if (value === undefined || value.length < 3) return "n/a";
  const x = Number(value[0]);
  const y = Number(value[1]);
  const z = Number(value[2]);
  if (!Number.isFinite(x) || !Number.isFinite(y) || !Number.isFinite(z)) {
    return "n/a";
  }
  return `(${x.toFixed(2)}, ${y.toFixed(2)}, ${z.toFixed(2)})`;
}

function formatMouseEvent(event: MouseEvent) {
  const modifiers = [
    event.ctrlKey ? "Ctrl" : "",
    event.altKey ? "Alt" : "",
    event.metaKey ? "Meta" : "",
    event.shiftKey ? "Shift" : "",
  ]
    .filter((x) => x.length > 0)
    .join("+");
  const mods = modifiers.length > 0 ? modifiers : "none";
  return `button=${event.button} buttons=${event.buttons} mods=${mods} prevented=${event.defaultPrevented}`;
}

abstract class SpatialSkeletonToolBase extends LayerTool<SegmentationUserLayer> {
  constructor(layer: SegmentationUserLayer) {
    super(layer, true);
  }

  protected getEditableSource(skeletonLayer: SpatiallyIndexedSkeletonLayer) {
    return getEditableSpatiallyIndexedSkeletonSource(skeletonLayer);
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
          : this.layer.spatialSkeletonState.getCachedNode(selectedNodeId);
      if (selectedNode?.segmentId === pickedSegmentId) {
        this.layer.clearSpatialSkeletonNodeSelection(false);
      }
      if (
        this.layer.spatialSkeletonState.mergeAnchorSegmentId.value ===
        pickedSegmentId
      ) {
        this.layer.clearSpatialSkeletonMergeAnchor();
      }
      const cachedSegmentIds = new Set<number>(
        [...getVisibleSegments(this.layer.displayState.segmentationGroupState.value).keys()]
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
      skeletonLayer.getNode(nodeHit.nodeId) ??
      this.layer.spatialSkeletonState.getCachedNode(nodeHit.nodeId);
    return {
      nodeId: nodeHit.nodeId,
      segmentId: nodeHit.segmentId ?? resolvedNodeInfo?.segmentId,
      position: resolvedNodeInfo?.position,
    };
  }

  protected bindPinnedSelectionAction(activation: ToolActivation<this>) {
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
          StatusMessage.showTemporaryMessage(
            this.isSpatialSkeletonSegmentVisible(pickedSegmentId)
              ? `Pinned skeleton ${pickedSegmentId}.`
              : this.describeVisibleSegmentRequirement(pickedSegmentId),
          );
          return;
        }
        if (pickedNode.segmentId !== undefined) {
          this.pinSegmentByNumber(pickedNode.segmentId);
        }
        this.layer.selectSpatialSkeletonNode(pickedNode.nodeId, true, {
          segmentId: pickedNode.segmentId,
          position: pickedNode.position,
        });
        StatusMessage.showTemporaryMessage(
          `Selected and pinned node ${pickedNode.nodeId}.`,
        );
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
              layer === this.layer &&
              typeof state.value === "object" &&
              state.value !== null &&
              "kind" in state.value &&
              state.value.kind === "spatialSkeletonNode",
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

  protected updateVisibleSkeletonSegments(
    resultSkeletonId: number | undefined,
    deletedSkeletonId?: number,
  ) {
    if (resultSkeletonId === undefined || !Number.isFinite(resultSkeletonId)) {
      return;
    }
    this.ensureSegmentVisibleByNumber(resultSkeletonId);
    if (
      deletedSkeletonId !== undefined &&
      Number.isFinite(deletedSkeletonId) &&
      Math.round(deletedSkeletonId) !== Math.round(resultSkeletonId)
    ) {
      this.removeVisibleSegmentByNumber(deletedSkeletonId, {
        deselect: true,
      });
    }
    this.selectSegmentByNumber(resultSkeletonId);
  }

  protected formatError(error: unknown) {
    return error instanceof Error ? error.message : String(error);
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

  private async commitMoveNode(
    skeletonLayer: SpatiallyIndexedSkeletonLayer,
    nodeId: number,
    position: Float32Array,
  ) {
    const skeletonSource = this.getEditableSource(skeletonLayer);
    if (skeletonSource === undefined) {
      throw new Error(
        "Unable to resolve editable source for the active spatial skeleton layer.",
      );
    }
    const x = Number(position[0]);
    const y = Number(position[1]);
    const z = Number(position[2]);
    const cachedNode =
      this.layer.spatialSkeletonState.getCachedNode(nodeId) ??
      skeletonLayer.getNode(nodeId);
    if (cachedNode?.segmentId === undefined) {
      throw new Error(
        `Moved node ${nodeId} is missing from the source-backed overlay cache.`,
      );
    }
    await skeletonSource.moveNode(nodeId, x, y, z);
    skeletonLayer.retainOverlaySegment(cachedNode.segmentId);
    this.layer.spatialSkeletonState.moveCachedNode(nodeId, position);
    this.layer.markSpatialSkeletonNodeDataChanged({
      invalidateFullSkeletonCache: false,
    });
    StatusMessage.showTemporaryMessage(
      `Moved node ${nodeId} to (${Math.round(x)}, ${Math.round(y)}, ${Math.round(z)}).`,
    );
  }

  private async commitAddNode(
    skeletonLayer: SpatiallyIndexedSkeletonLayer,
    skeletonId: number,
    parentNodeId: number | undefined,
    position: Float32Array,
  ): Promise<SpatiallyIndexedSkeletonAddNodeResult> {
    const skeletonSource = this.getEditableSource(skeletonLayer);
    if (skeletonSource === undefined) {
      logSpatialSkeletonEdit("commit-add-node-no-client", {
        skeletonId,
        parentNodeId,
        layerType: skeletonLayer.constructor.name,
      });
      throw new Error(
        "Unable to resolve editable source for the active spatial skeleton layer.",
      );
    }
    const x = Number(position[0]);
    const y = Number(position[1]);
    const z = Number(position[2]);
    logSpatialSkeletonEdit("commit-add-node-request", {
      skeletonId,
      parentNodeId,
      x,
      y,
      z,
      layerType: skeletonLayer.constructor.name,
    });
    const nodeInfo = await skeletonSource.addNode(
      skeletonId,
      x,
      y,
      z,
      parentNodeId,
    );
    logSpatialSkeletonEdit("commit-add-node-response", {
      requestedSkeletonId: skeletonId,
      parentNodeId,
      treenodeId: nodeInfo.treenodeId,
      skeletonId: nodeInfo.skeletonId,
    });
    StatusMessage.showTemporaryMessage(
      `Added node ${nodeInfo.treenodeId} on segment ${nodeInfo.skeletonId}.`,
    );
    return nodeInfo;
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
    const { body, header } =
      makeToolActivationStatusMessageWithHeader(activation);
    header.textContent = "Spatial skeleton edit mode";
    const statusElement = document.createElement("span");
    body.appendChild(statusElement);
    const setStatus = (message: string) => {
      statusElement.textContent = message;
    };

    const setDebug = (_key: string, _value: string) => {};
    const debugLog = (label: string, data: Record<string, unknown> = {}) => {
      logSpatialSkeletonEdit(label, data);
    };
    const isChunkLoadWaitReason = (reason: string | undefined) => {
      if (reason === undefined) return false;
      return (
        reason === "Waiting for visible skeleton chunks." ||
        reason.startsWith("Wait for visible skeleton chunks to load (")
      );
    };

    const disableWithMessage = (message: string) => {
      setStatus(message);
      StatusMessage.showTemporaryMessage(message);
      debugLog("activation-disabled", { message });
      queueMicrotask(() => activation.cancel());
    };

    const getEditCapabilityDisabledReason = () =>
      layer.getSpatialSkeletonActionsDisabledReason(["addNodes", "moveNodes"], {
        requireMaxLod: false,
        requireVisibleChunks: false,
      });
    const getEditMutationDisabledReason = () =>
      layer.getSpatialSkeletonActionsDisabledReason(["addNodes", "moveNodes"]);
    const setReadyStatus = () => {
      setStatus(
        "Edit mode enabled. You can move nodes, append new nodes to an existing skeleton, and create new skeletons.",
      );
    };
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

    layer.spatialSkeletonEditMode.value = true;
    activation.registerDisposer(() => {
      layer.spatialSkeletonEditMode.value = false;
    });
    setDebug("mode", "on");
    setDebug(
      "editableSource",
      String(
        layer.spatialSkeletonSourceCapabilities.value.addNodes &&
          layer.spatialSkeletonSourceCapabilities.value.moveNodes,
      ),
    );
    setDebug(
      "maxLodAllowed",
      String(layer.spatialSkeletonEditModeAllowed.value),
    );
    setDebug(
      "gridLevel2d",
      String(layer.displayState.spatialSkeletonGridLevel2d.value),
    );
    setDebug(
      "gridLevel3d",
      String(layer.displayState.spatialSkeletonGridLevel3d.value),
    );
    setDebug(
      "pickedLayer",
      this.mouseState.pickedRenderLayer?.constructor?.name ?? "none",
    );
    setDebug("mousePos", formatVec3(this.mouseState.unsnappedPosition));
    debugLog("activation-enabled", {
      gridLevel2d: layer.displayState.spatialSkeletonGridLevel2d.value,
      gridLevel3d: layer.displayState.spatialSkeletonGridLevel3d.value,
      levels: layer.displayState.spatialSkeletonGridLevels.value.length,
    });
    activation.bindInputEventMap(SPATIAL_SKELETON_EDIT_INPUT_EVENT_MAP);
    this.bindPinnedSelectionAction(activation);
    this.bindClearSelectionAction(activation);
    this.bindVisibilityToggleAction(activation);
    updateInteractionStatus();
    activation.registerDisposer(() => {
      layer.spatialSkeletonState.clearPendingNodePositions();
      layer.clearSpatialSkeletonNodeSelection(false);
    });
    activation.registerEventListener(
      window,
      "mousedown",
      (event: MouseEvent) => {
        if (event.button !== 0) return;
        setDebug("windowMouseDown", formatMouseEvent(event));
        debugLog("window-mousedown-capture", {
          mouse: formatMouseEvent(event),
          targetType:
            event.target instanceof Element
              ? event.target.tagName.toLowerCase()
              : typeof event.target,
        });
      },
      { capture: true },
    );
    activation.registerDisposer(
      layer.spatialSkeletonActionsAllowed.changed.add(() => {
        setDebug(
          "actionsAllowed",
          String(layer.spatialSkeletonActionsAllowed.value),
        );
        if (!layer.spatialSkeletonActionsAllowed.value) {
          const capabilityReason = getEditCapabilityDisabledReason();
          if (capabilityReason !== undefined) {
            StatusMessage.showTemporaryMessage(capabilityReason);
            debugLog("auto-cancel-actions-not-supported", {
              reason: capabilityReason,
            });
            activation.cancel();
            return;
          }
          const reason = updateInteractionStatus();
          if (reason !== undefined) {
            StatusMessage.showTemporaryMessage(reason);
            if (isChunkLoadWaitReason(reason)) {
              debugLog("actions-paused-waiting-for-chunks", { reason });
              return;
            }
            debugLog("actions-paused-editing-limited", { reason });
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
        setDebug("action", "spatial-skeleton-add-node");
        setDebug("actionMouseDown", formatMouseEvent(event.detail));
        debugLog("add-node-action-triggered", {
          mouse: formatMouseEvent(event.detail),
          pickedLayer: this.mouseState.pickedRenderLayer?.constructor?.name,
          selectedSegment: String(
            layer.displayState.segmentSelectionState.baseValue ?? "none",
          ),
        });
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
        if (selectedParentNodeId === undefined) {
          const pickedSegmentId = this.getPickedSpatialSkeletonSegment();
          if (pickedSegmentId !== undefined) {
            this.selectSegmentByNumber(pickedSegmentId);
            StatusMessage.showTemporaryMessage(
              this.isSpatialSkeletonSegmentVisible(pickedSegmentId)
                ? `Skeleton ${pickedSegmentId} is visible. Select a node before adding a connected node.`
                : this.describeVisibleSegmentRequirement(pickedSegmentId),
            );
            return;
          }
        }
        const clickStartPosition = this.getMousePosition();
        if (clickStartPosition === undefined) {
          StatusMessage.showTemporaryMessage(
            "Unable to resolve add-node position for this click.",
          );
          debugLog("add-node-position-unresolved", {
            selectedNodeId: layer.selectedSpatialSkeletonNodeId.value,
            clientX: event.detail.clientX,
            clientY: event.detail.clientY,
          });
          return;
        }
        let dragDistanceSquared = 0;
        setDebug("dragState", "armed-ctrl");
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
              setDebug("dragState", "ignored-ctrl-drag");
              debugLog("ctrl-drag-ignored", {
                dragDistanceSquared,
                thresholdSquared,
              });
              return;
            }
            const selectedParentNodeId =
              layer.selectedSpatialSkeletonNodeId.value;
            const selectedParentNode =
              selectedParentNodeId === undefined
                ? undefined
                : skeletonLayer.getNode(selectedParentNodeId);
            const targetSkeletonId =
              selectedParentNode === undefined
                ? 0
                : selectedParentNode.segmentId;
            const clickPosition =
              this.getMousePosition() ?? new Float32Array(clickStartPosition);
            debugLog("ctrl-add-attempt", {
              selectedParentNodeId,
              selectedParentSegmentId: selectedParentNode?.segmentId,
              targetSkeletonId,
              clickPosition: formatVec3(clickPosition),
            });
            void (async () => {
              let committedNode: SpatiallyIndexedSkeletonAddNodeResult;
              try {
                committedNode = await this.commitAddNode(
                  skeletonLayer,
                  targetSkeletonId,
                  selectedParentNodeId,
                  clickPosition,
                );
              } catch (error) {
                StatusMessage.showTemporaryMessage(
                  "Failed to commit node creation to the active skeleton source.",
                );
                const errorInfo: Record<string, unknown> =
                  error instanceof Error
                    ? { message: error.message, stack: error.stack }
                    : { error: String(error) };
                debugLog("add-node-commit-failed", {
                  ...errorInfo,
                  parentNodeId: selectedParentNodeId,
                  clickPosition: formatVec3(clickPosition),
                });
                return;
              }
              const newNode = {
                nodeId: committedNode.treenodeId,
                segmentId: committedNode.skeletonId,
                position: new Float32Array(clickPosition),
                parentNodeId: selectedParentNodeId,
              };
              layer.spatialSkeletonState.upsertCachedNode(newNode, {
                allowUncachedSegment: selectedParentNodeId === undefined,
              });
              this.ensureSegmentVisibleByNumber(newNode.segmentId);
              this.pinSegmentByNumber(newNode.segmentId);
              layer.selectSpatialSkeletonNode(
                newNode.nodeId,
                layer.manager.root.selectionState.pin.value,
                {
                  segmentId: newNode.segmentId,
                  position: newNode.position,
                },
              );
              layer.markSpatialSkeletonNodeDataChanged({
                invalidateFullSkeletonCache: false,
              });
              skeletonLayer.invalidateSourceCaches();
              debugLog("add-node-committed", {
                nodeId: newNode.nodeId,
                segmentId: newNode.segmentId,
                parentNodeId: selectedParentNodeId,
                position: formatVec3(newNode.position),
              });
              setReadyStatus();
            })();
          },
        );
      },
    );

    activation.bindAction(
      "spatial-skeleton-move-node",
      (event: ActionEvent<MouseEvent>) => {
        setDebug("action", "spatial-skeleton-move-node");
        setDebug("actionMouseDown", formatMouseEvent(event.detail));
        debugLog("move-node-action-triggered", {
          mouse: formatMouseEvent(event.detail),
          pickedLayer: this.mouseState.pickedRenderLayer?.constructor?.name,
          selectedSegment: String(
            layer.displayState.segmentSelectionState.baseValue ?? "none",
          ),
        });
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
        debugLog("resolved-action-panel", {
          panel: actionPanel?.constructor?.name ?? "none",
          pickedLayer:
            this.mouseState.pickedRenderLayer?.constructor?.name ?? "none",
        });
        const pickedNode = this.getPickedSpatialSkeletonNode();
        if (pickedNode === undefined) {
          const pickedSegmentId = this.getPickedSpatialSkeletonSegment();
          if (pickedSegmentId !== undefined) {
            this.selectSegmentByNumber(pickedSegmentId);
            layer.clearSpatialSkeletonNodeSelection(false);
            StatusMessage.showTemporaryMessage(
              this.isSpatialSkeletonSegmentVisible(pickedSegmentId)
                ? `Skeleton ${pickedSegmentId} is visible. Click a node in that skeleton, then Alt-drag to move it.`
                : this.describeVisibleSegmentRequirement(pickedSegmentId),
            );
          }
          setDebug("dragState", "ignored-no-node");
          debugLog("drag-ignored-no-picked-node");
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
          debugLog("drag-node-not-found", {
            nodeId: pickedNode.nodeId,
          });
          return;
        }
        const dragPanel = actionPanel;
        if (dragPanel === undefined) {
          StatusMessage.showTemporaryMessage(
            "Unable to resolve active panel for node drag.",
          );
          return;
        }
        setDebug("hit", `node=${pickedNode.nodeId}`);
        setDebug("hitPos", formatVec3(nodeInfo.position));
        setDebug("dragPanel", dragPanel.constructor?.name ?? "none");
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
        setDebug("dragState", "armed");
        startRelativeMouseDrag(
          event.detail,
          (_event, deltaX, deltaY) => {
            dragDistanceSquared += deltaX * deltaX + deltaY * deltaY;
            const thresholdSquared =
              DRAG_START_DISTANCE_PX * DRAG_START_DISTANCE_PX;
            if (!dragActive && dragDistanceSquared >= thresholdSquared) {
              dragActive = true;
              setStatus(`Dragging node ${pickedNode.nodeId}`);
              setDebug("dragState", "active");
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
            setDebug("draggingNode", String(pickedNode.nodeId));
            setDebug("dragMoves", String(moveEvents));
            setDebug("lastAppliedPos", formatVec3(lastPosition));
          },
          (_finishEvent) => {
            if (!dragActive) {
              setReadyStatus();
              setDebug("dragState", "idle");
              return;
            }
            setReadyStatus();
            setDebug("dragState", "idle");
            setDebug("draggingNode", "none");
            debugLog("drag-end", {
              nodeId: pickedNode.nodeId,
              moved,
              moveEvents,
              finalPosition: formatVec3(lastPosition),
            });
            if (moved) {
              void this.commitMoveNode(
                skeletonLayer,
                pickedNode.nodeId,
                lastPosition,
              )
                .then(() => {
                  layer.spatialSkeletonState.clearPendingNodePosition(
                    pickedNode.nodeId,
                  );
                })
                .catch((error) => {
                  layer.spatialSkeletonState.clearPendingNodePosition(
                    pickedNode.nodeId,
                  );
                  StatusMessage.showTemporaryMessage(
                    `Failed to commit move for node ${pickedNode.nodeId}.`,
                  );
                  debugLog("move-node-commit-failed", {
                    nodeId: pickedNode.nodeId,
                    error: String(error),
                    position: formatVec3(lastPosition),
                  });
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
    activation.registerDisposer(() => {
      this.layer.clearSpatialSkeletonMergeAnchor();
      this.layer.clearSecondaryInspectedSpatialSkeletonSegment();
    });
    const { body, header } =
      makeToolActivationStatusMessageWithHeader(activation);
    header.textContent = "Spatial skeleton merge mode";
    let pending = false;
    const getAnchorNode = () => {
      const nodeId = this.layer.spatialSkeletonState.mergeAnchorNodeId.value;
      const segmentId =
        this.layer.spatialSkeletonState.mergeAnchorSegmentId.value;
      if (
        nodeId === undefined ||
        segmentId === undefined ||
        !Number.isSafeInteger(nodeId) ||
        !Number.isSafeInteger(segmentId)
      ) {
        return undefined;
      }
      return { nodeId, segmentId };
    };
    const updateStatus = () => {
      const anchorNode = getAnchorNode();
      if (pending) {
        body.textContent = "Merging selected skeleton nodes.";
        return;
      }
      if (anchorNode === undefined) {
        body.textContent =
          "Choose a merge anchor node, then choose a node in a different skeleton to merge into it.";
        return;
      }
      body.textContent = `Anchor node ${anchorNode.nodeId} on skeleton ${anchorNode.segmentId}. Choose a node in a different skeleton to complete the merge.`;
    };
    updateStatus();
    activation.bindInputEventMap(SPATIAL_SKELETON_PICK_INPUT_EVENT_MAP);
    this.bindClearSelectionAction(activation);
    this.bindVisibilityToggleAction(activation);
    this.registerAutoCancelOnDisabled(
      activation,
      "mergeSkeletons",
      updateStatus,
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
            StatusMessage.showTemporaryMessage(
              anchorNode === undefined ||
                pickedSegmentId === anchorNode.segmentId
                ? this.isSpatialSkeletonSegmentVisible(pickedSegmentId)
                  ? `Skeleton ${pickedSegmentId} is visible. Ctrl+right-click a node to choose the merge anchor.`
                  : this.describeVisibleSegmentRequirement(pickedSegmentId)
                : this.isSpatialSkeletonSegmentVisible(pickedSegmentId)
                  ? `Skeleton ${pickedSegmentId} is visible. Ctrl+right-click a node in that skeleton to merge with anchor node ${anchorNode.nodeId}.`
                  : this.describeVisibleSegmentRequirement(pickedSegmentId),
            );
            updateStatus();
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
          this.layer.setSpatialSkeletonMergeAnchor(
            pickedNode.nodeId,
            pickedNode.segmentId,
          );
          this.layer.selectSpatialSkeletonNode(pickedNode.nodeId, true, {
            segmentId: pickedNode.segmentId,
            position: pickedNode.position,
          });
          StatusMessage.showTemporaryMessage(
            `Selected node ${pickedNode.nodeId} as merge anchor.`,
          );
          updateStatus();
          return;
        }
        if (anchorNode.segmentId === pickedNode.segmentId) {
          this.layer.setSpatialSkeletonMergeAnchor(
            pickedNode.nodeId,
            pickedNode.segmentId,
          );
          this.layer.selectSpatialSkeletonNode(pickedNode.nodeId, true, {
            segmentId: pickedNode.segmentId,
            position: pickedNode.position,
          });
          StatusMessage.showTemporaryMessage(
            `Merge requires nodes from different skeletons. Node ${pickedNode.nodeId} is now the merge anchor.`,
          );
          updateStatus();
          return;
        }
        const skeletonSource = this.getEditableSource(skeletonLayer);
        if (skeletonSource === undefined) {
          StatusMessage.showTemporaryMessage(
            "Unable to resolve editable source for the active spatial skeleton layer.",
          );
          return;
        }
        const firstNode = anchorNode;
        const secondNode = {
          nodeId: pickedNode.nodeId,
          segmentId: pickedNode.segmentId,
        };
        pending = true;
        updateStatus();
        void (async () => {
          try {
            const result = await skeletonSource.mergeSkeletons(
              firstNode.nodeId,
              secondNode.nodeId,
            );
            const winningNode =
              result.resultSkeletonId === secondNode.segmentId
                ? secondNode
                : firstNode;
            const losingNode =
              winningNode.nodeId === firstNode.nodeId ? secondNode : firstNode;
            const resultSkeletonId =
              result.resultSkeletonId ?? winningNode.segmentId;
            const deletedSkeletonId =
              result.deletedSkeletonId ?? losingNode.segmentId;
            this.updateVisibleSkeletonSegments(
              resultSkeletonId,
              deletedSkeletonId,
            );
            this.layer.spatialSkeletonState.mergeCachedSegments({
              resultSegmentId: resultSkeletonId,
              mergedSegmentId: deletedSkeletonId,
              childNodeId: losingNode.nodeId,
              parentNodeId: winningNode.nodeId,
            });
            this.layer.selectSpatialSkeletonNode(
              losingNode.nodeId,
              this.layer.manager.root.selectionState.pin.value,
              {
                segmentId: resultSkeletonId,
              },
            );
            this.layer.markSpatialSkeletonNodeDataChanged({
              invalidateFullSkeletonCache: false,
            });
            skeletonLayer.invalidateSourceCaches();
            this.layer.clearSpatialSkeletonMergeAnchor();
            const swapSuffix = result.stableAnnotationSwap
              ? " Merge direction was adjusted by the active source."
              : "";
            StatusMessage.showTemporaryMessage(
              `Merged skeleton ${deletedSkeletonId} into ${resultSkeletonId}.${swapSuffix}`,
            );
          } catch (error) {
            StatusMessage.showTemporaryMessage(
              `Failed to merge skeletons: ${this.formatError(error)}`,
            );
          } finally {
            pending = false;
            updateStatus();
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
    const { body, header } =
      makeToolActivationStatusMessageWithHeader(activation);
    header.textContent = "Spatial skeleton split mode";
    let pendingNodeId: number | undefined;
    const updateStatus = () => {
      if (pendingNodeId !== undefined) {
        body.textContent = `Splitting skeleton at node ${pendingNodeId}.`;
        return;
      }
      body.textContent =
        "Choose the node where the skeleton should be split. Root nodes cannot be split.";
    };
    updateStatus();
    activation.bindInputEventMap(SPATIAL_SKELETON_PICK_INPUT_EVENT_MAP);
    this.bindClearSelectionAction(activation);
    this.bindVisibilityToggleAction(activation);
    this.registerAutoCancelOnDisabled(
      activation,
      "splitSkeletons",
      updateStatus,
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
        if (pendingNodeId !== undefined) return;
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
            StatusMessage.showTemporaryMessage(
              this.isSpatialSkeletonSegmentVisible(pickedSegmentId)
                ? `Skeleton ${pickedSegmentId} is visible. Ctrl+right-click a node in that skeleton to split it.`
                : this.describeVisibleSegmentRequirement(pickedSegmentId),
            );
            updateStatus();
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
        const skeletonSource = this.getEditableSource(skeletonLayer);
        if (skeletonSource === undefined) {
          StatusMessage.showTemporaryMessage(
            "Unable to resolve editable source for the active spatial skeleton layer.",
          );
          return;
        }
        pendingNodeId = pickedNode.nodeId;
        updateStatus();
        void (async () => {
          try {
            const result = await skeletonSource.splitSkeleton(
              pickedNode.nodeId,
            );
            const newSkeletonId = result.newSkeletonId;
            const existingSkeletonId =
              result.existingSkeletonId ?? pickedNode.segmentId;
            if (newSkeletonId === undefined) {
              throw new Error(
                "The active skeleton source did not return a new skeleton id for the split.",
              );
            }
            if (existingSkeletonId === undefined) {
              throw new Error(
                "The active skeleton source did not return the existing skeleton id for the split.",
              );
            }
            this.ensureSegmentVisibleByNumber(existingSkeletonId);
            this.ensureSegmentVisibleByNumber(newSkeletonId);
            this.pinSegmentByNumber(newSkeletonId);
            this.layer.selectSpatialSkeletonNode(
              pickedNode.nodeId,
              this.layer.manager.root.selectionState.pin.value,
              {
                segmentId: newSkeletonId,
              },
            );
            this.layer.spatialSkeletonState.splitCachedSegmentAtNode({
              existingSegmentId: existingSkeletonId,
              nodeId: pickedNode.nodeId,
              newSegmentId: newSkeletonId,
            });
            this.layer.markSpatialSkeletonNodeDataChanged({
              invalidateFullSkeletonCache: false,
            });
            skeletonLayer.invalidateSourceCaches();
            StatusMessage.showTemporaryMessage(
              `Split skeleton ${existingSkeletonId}. New skeleton: ${newSkeletonId}.`,
            );
          } catch (error) {
            StatusMessage.showTemporaryMessage(
              `Failed to split skeleton: ${this.formatError(error)}`,
            );
          } finally {
            pendingNodeId = undefined;
            updateStatus();
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
