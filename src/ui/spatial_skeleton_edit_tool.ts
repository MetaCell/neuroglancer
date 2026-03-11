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
import { CatmaidClient } from "#src/datasource/catmaid/api.js";
import type { CatmaidAddNodeResult } from "#src/datasource/catmaid/api.js";
import {
  PerspectiveViewSpatiallyIndexedSkeletonLayer,
  SpatiallyIndexedSkeletonLayer,
  SliceViewPanelSpatiallyIndexedSkeletonLayer,
  SliceViewSpatiallyIndexedSkeletonLayer,
} from "#src/skeleton/frontend.js";
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

export const SPATIAL_SKELETON_EDIT_MODE_TOOL_ID = "spatialSkeletonEditMode";
export const SPATIAL_SKELETON_MERGE_MODE_TOOL_ID = "spatialSkeletonMergeMode";
export const SPATIAL_SKELETON_SPLIT_MODE_TOOL_ID = "spatialSkeletonSplitMode";

const SPATIAL_SKELETON_EDIT_INPUT_EVENT_MAP = EventActionMap.fromObject({
  // Use shift+click for node creation, and alt+drag for node movement.
  // Keep plain left-drag/click camera controls available from default bindings.
  "at:shift+mousedown0": "spatial-skeleton-edit-drag-node",
  "at:alt+mousedown0": "spatial-skeleton-edit-drag-node",
  "at:click0": {
    action: "spatial-skeleton-edit-select-node",
    stopPropagation: false,
    preventDefault: false,
  },
});

const SPATIAL_SKELETON_PICK_INPUT_EVENT_MAP = EventActionMap.fromObject({
  "at:click0": {
    action: "spatial-skeleton-pick-node",
    stopPropagation: false,
    preventDefault: false,
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

abstract class SpatialSkeletonCatmaidToolBase extends LayerTool<SegmentationUserLayer> {
  private readonly catmaidClients = new Map<string, CatmaidClient>();

  constructor(layer: SegmentationUserLayer) {
    super(layer, true);
  }

  protected getCatmaidClient(
    skeletonLayer: SpatiallyIndexedSkeletonLayer,
  ): CatmaidClient | undefined {
    const source = skeletonLayer.source as {
      parameters?: {
        catmaidParameters?: {
          url?: string;
          projectId?: number;
          token?: string;
        };
      };
      credentialsProvider?: unknown;
    };
    const catmaidParameters = source.parameters?.catmaidParameters;
    if (
      catmaidParameters === undefined ||
      typeof catmaidParameters.url !== "string" ||
      typeof catmaidParameters.projectId !== "number"
    ) {
      return undefined;
    }
    const cacheKey = `${catmaidParameters.url}|${catmaidParameters.projectId}|${catmaidParameters.token ?? ""}`;
    let client = this.catmaidClients.get(cacheKey);
    if (client === undefined) {
      client = new CatmaidClient(
        catmaidParameters.url,
        catmaidParameters.projectId,
        catmaidParameters.token,
        source.credentialsProvider as any,
      );
      this.catmaidClients.set(cacheKey, client);
    }
    return client;
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

  protected selectSegmentByNumber(value: number) {
    if (!Number.isFinite(value)) return;
    this.layer.selectSegment(BigInt(Math.round(value)), false);
  }

  protected resolvePickedNodeForAction(
    skeletonLayer: SpatiallyIndexedSkeletonLayer,
  ) {
    const nodeHit = this.getPickedSpatialSkeletonNode();
    if (nodeHit === undefined) {
      return undefined;
    }
    const resolvedNodeInfo =
      nodeHit.segmentId === undefined
        ? skeletonLayer.getNode(nodeHit.nodeId)
        : undefined;
    const segmentId = nodeHit.segmentId ?? resolvedNodeInfo?.segmentId;
    if (segmentId !== undefined) {
      this.selectSegmentByNumber(segmentId);
    }
    this.layer.selectedSpatialSkeletonNodeId.value = nodeHit.nodeId;
    return {
      nodeId: nodeHit.nodeId,
      segmentId,
    };
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
    onReady?: () => void,
  ) {
    activation.registerDisposer(
      this.layer.spatialSkeletonActionsAllowed.changed.add(() => {
        if (this.layer.spatialSkeletonActionsAllowed.value) {
          onReady?.();
          return;
        }
        const disabledReason =
          this.layer.getSpatialSkeletonActionsDisabledReason();
        if (disabledReason !== undefined) {
          StatusMessage.showTemporaryMessage(disabledReason);
        }
        activation.cancel();
      }),
    );
  }

  protected updateVisibleSkeletonSegments(
    resultSkeletonId: number | undefined,
    deletedSkeletonId?: number,
  ) {
    if (resultSkeletonId === undefined || !Number.isFinite(resultSkeletonId)) {
      return;
    }
    const segmentationGroupState =
      this.layer.displayState.segmentationGroupState.value;
    const { visibleSegments, selectedSegments } = segmentationGroupState;
    visibleSegments.add(BigInt(resultSkeletonId));
    if (
      deletedSkeletonId !== undefined &&
      Number.isFinite(deletedSkeletonId) &&
      Math.round(deletedSkeletonId) !== Math.round(resultSkeletonId)
    ) {
      visibleSegments.delete(BigInt(deletedSkeletonId));
      selectedSegments.delete(BigInt(deletedSkeletonId));
    }
    this.selectSegmentByNumber(resultSkeletonId);
  }

  protected formatError(error: unknown) {
    return error instanceof Error ? error.message : String(error);
  }
}

export class SpatialSkeletonEditModeTool extends SpatialSkeletonCatmaidToolBase {
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
    const client = this.getCatmaidClient(skeletonLayer);
    if (client === undefined) {
      throw new Error(
        "Unable to resolve CATMAID client for active spatial skeleton source.",
      );
    }
    const x = Number(position[0]);
    const y = Number(position[1]);
    const z = Number(position[2]);
    await client.moveNode(nodeId, x, y, z);
    StatusMessage.showTemporaryMessage(
      `Moved node ${nodeId} to (${Math.round(x)}, ${Math.round(y)}, ${Math.round(z)}).`,
    );
  }

  private async commitAddNode(
    skeletonLayer: SpatiallyIndexedSkeletonLayer,
    skeletonId: number,
    parentNodeId: number | undefined,
    position: Float32Array,
  ): Promise<CatmaidAddNodeResult> {
    const client = this.getCatmaidClient(skeletonLayer);
    if (client === undefined) {
      logSpatialSkeletonEdit("commit-add-node-no-client", {
        skeletonId,
        parentNodeId,
        layerType: skeletonLayer.constructor.name,
      });
      throw new Error(
        "Unable to resolve CATMAID client for active spatial skeleton source.",
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
    const nodeInfo = await client.addNodeWithInfo(
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

    const disabledReason = layer.getSpatialSkeletonActionsDisabledReason();
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
    setDebug("catmaid", String(layer.isCatmaidSource.value));
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
    const setReadyStatus = () => {
      setStatus(
        "Edit mode enabled. Left click toggles node selection. Shift+click adds a node (root if no node is selected). Alt+drag moves nodes.",
      );
    };
    setReadyStatus();
    activation.registerDisposer(() => {
      layer.selectedSpatialSkeletonNodeId.value = undefined;
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
    activation.registerEventListener(
      window,
      "pointermove",
      (_event: PointerEvent) => {
        if (!this.mouseState.updateUnconditionally()) return;
        setDebug(
          "pickedLayer",
          this.mouseState.pickedRenderLayer?.constructor?.name ?? "none",
        );
        setDebug("mousePos", formatVec3(this.mouseState.unsnappedPosition));
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
          const reason = layer.getSpatialSkeletonActionsDisabledReason();
          if (reason !== undefined) {
            StatusMessage.showTemporaryMessage(reason);
            if (isChunkLoadWaitReason(reason)) {
              setStatus(reason);
              debugLog("actions-paused-waiting-for-chunks", { reason });
              return;
            }
            debugLog("auto-cancel-actions-not-allowed", { reason });
          }
          activation.cancel();
          return;
        }
        setReadyStatus();
      }),
    );

    activation.bindAction(
      "spatial-skeleton-edit-select-node",
      (event: ActionEvent<MouseEvent>) => {
        if (event.detail.button !== 0) return;
        if (
          event.detail.shiftKey ||
          event.detail.ctrlKey ||
          event.detail.altKey ||
          event.detail.metaKey
        ) {
          return;
        }
        const clickDisabledReason =
          layer.getSpatialSkeletonActionsDisabledReason();
        if (clickDisabledReason !== undefined) {
          StatusMessage.showTemporaryMessage(clickDisabledReason);
          return;
        }
        const skeletonLayer = this.getActiveSpatiallyIndexedSkeletonLayer();
        if (skeletonLayer === undefined) {
          return;
        }
        const nodeHit = this.getPickedSpatialSkeletonNode();
        debugLog("select-click-hit-test", {
          nodeHitId: nodeHit?.nodeId,
          selectedNodeId: layer.selectedSpatialSkeletonNodeId.value,
          mousePosition: formatVec3(this.mouseState.unsnappedPosition),
        });
        if (nodeHit === undefined) {
          if (layer.selectedSpatialSkeletonNodeId.value !== undefined) {
            layer.selectedSpatialSkeletonNodeId.value = undefined;
            StatusMessage.showTemporaryMessage("Selection cleared.");
            debugLog("selection-cleared", { reason: "empty-space-click" });
            setReadyStatus();
          }
          return;
        }

        if (layer.selectedSpatialSkeletonNodeId.value === nodeHit.nodeId) {
          layer.selectedSpatialSkeletonNodeId.value = undefined;
          StatusMessage.showTemporaryMessage(
            `Deselected node ${nodeHit.nodeId}.`,
          );
          debugLog("node-deselected", { nodeId: nodeHit.nodeId });
          setReadyStatus();
          return;
        }

        const resolvedNodeInfo =
          nodeHit.segmentId === undefined
            ? skeletonLayer.getNode(nodeHit.nodeId)
            : undefined;
        const segmentId = nodeHit.segmentId ?? resolvedNodeInfo?.segmentId;
        if (segmentId !== undefined) {
          this.selectSegmentByNumber(segmentId);
        }
        layer.selectedSpatialSkeletonNodeId.value = nodeHit.nodeId;
        StatusMessage.showTemporaryMessage(
          `Selected node ${nodeHit.nodeId}. Shift+click to add a connected node.`,
        );
        debugLog("node-selected", {
          nodeId: nodeHit.nodeId,
          segmentId: nodeHit.segmentId,
        });
        setReadyStatus();
      },
    );

    activation.bindAction(
      "spatial-skeleton-edit-drag-node",
      (event: ActionEvent<MouseEvent>) => {
        setDebug("action", "spatial-skeleton-edit-drag-node");
        setDebug("actionMouseDown", formatMouseEvent(event.detail));
        debugLog("drag-action-triggered", {
          mouse: formatMouseEvent(event.detail),
          pickedLayer: this.mouseState.pickedRenderLayer?.constructor?.name,
          selectedSegment: String(
            layer.displayState.segmentSelectionState.baseValue ?? "none",
          ),
        });
        event.stopPropagation();
        event.detail.preventDefault();
        const dragDisabledReason =
          layer.getSpatialSkeletonActionsDisabledReason();
        if (dragDisabledReason !== undefined) {
          StatusMessage.showTemporaryMessage(dragDisabledReason);
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
        const addGesture = event.detail.shiftKey;
        if (addGesture) {
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
          setDebug("dragState", "armed-shift");
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
                setDebug("dragState", "ignored-shift-drag");
                debugLog("shift-drag-ignored", {
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
              debugLog("shift-add-attempt", {
                selectedParentNodeId,
                selectedParentSegmentId: selectedParentNode?.segmentId,
                targetSkeletonId,
                clickPosition: formatVec3(clickPosition),
              });
              void (async () => {
                let committedNode: CatmaidAddNodeResult;
                try {
                  committedNode = await this.commitAddNode(
                    skeletonLayer,
                    targetSkeletonId,
                    selectedParentNodeId,
                    clickPosition,
                  );
                } catch (error) {
                  StatusMessage.showTemporaryMessage(
                    "Failed to commit node creation to CATMAID.",
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
                const newNode = skeletonLayer.addNode(clickPosition, {
                  segmentId: committedNode.skeletonId,
                  parentNodeId: selectedParentNodeId,
                  nodeId: committedNode.treenodeId,
                });
                if (newNode === undefined) {
                  StatusMessage.showTemporaryMessage(
                    `Added node ${committedNode.treenodeId} in CATMAID, but local preview update failed.`,
                  );
                  layer.selectedSpatialSkeletonNodeId.value =
                    committedNode.treenodeId;
                  this.selectSegmentByNumber(committedNode.skeletonId);
                  layer.markSpatialSkeletonNodeDataChanged();
                  setReadyStatus();
                  return;
                }
                layer.selectedSpatialSkeletonNodeId.value = newNode.nodeId;
                this.selectSegmentByNumber(newNode.segmentId);
                layer.markSpatialSkeletonNodeDataChanged();
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
          return;
        }

        const pickedNode = this.getPickedSpatialSkeletonNode();
        if (pickedNode === undefined) {
          setDebug("dragState", "ignored-no-node");
          debugLog("drag-ignored-no-picked-node");
          return;
        }
        const nodeInfo = skeletonLayer.getNode(pickedNode.nodeId);
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
            const didMove = skeletonLayer.setNodePosition(
              pickedNode.nodeId,
              panelTranslatedPosition,
            );
            if (!didMove) return;
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
              layer.markSpatialSkeletonNodeDataChanged();
              void this.commitMoveNode(
                skeletonLayer,
                pickedNode.nodeId,
                lastPosition,
              ).catch((error) => {
                StatusMessage.showTemporaryMessage(
                  `Failed to commit move for node ${pickedNode.nodeId}.`,
                );
                debugLog("move-node-commit-failed", {
                  nodeId: pickedNode.nodeId,
                  error: String(error),
                  position: formatVec3(lastPosition),
                });
              });
            }
          },
        );
      },
    );
  }
}

class SpatialSkeletonMergeModeTool extends SpatialSkeletonCatmaidToolBase {
  toJSON() {
    return SPATIAL_SKELETON_MERGE_MODE_TOOL_ID;
  }

  get description() {
    return "skeleton merge mode";
  }

  activate(activation: ToolActivation<this>) {
    const reason = this.layer.getSpatialSkeletonActionsDisabledReason();
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
    const { body, header } =
      makeToolActivationStatusMessageWithHeader(activation);
    header.textContent = "Spatial skeleton merge mode";
    let anchorNode:
      | {
          nodeId: number;
          segmentId: number;
        }
      | undefined;
    let pending = false;
    const updateStatus = () => {
      if (pending) {
        body.textContent = "Merging selected skeleton nodes.";
        return;
      }
      if (anchorNode === undefined) {
        body.textContent =
          "Click a node to choose the merge anchor, then click a node in a different skeleton.";
        return;
      }
      body.textContent = `Anchor node ${anchorNode.nodeId} on skeleton ${anchorNode.segmentId}. Click a node in a different skeleton to merge into it.`;
    };
    updateStatus();
    activation.bindInputEventMap(SPATIAL_SKELETON_PICK_INPUT_EVENT_MAP);
    this.registerAutoCancelOnDisabled(activation, updateStatus);
    activation.bindAction(
      "spatial-skeleton-pick-node",
      (event: ActionEvent<MouseEvent>) => {
        if (event.detail.button !== 0) return;
        if (
          event.detail.shiftKey ||
          event.detail.ctrlKey ||
          event.detail.altKey ||
          event.detail.metaKey
        ) {
          return;
        }
        if (pending) return;
        const disabledReason =
          this.layer.getSpatialSkeletonActionsDisabledReason();
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
        const pickedNode = this.resolvePickedNodeForAction(skeletonLayer);
        if (pickedNode === undefined || pickedNode.segmentId === undefined) {
          return;
        }
        if (
          anchorNode === undefined ||
          anchorNode.nodeId === pickedNode.nodeId
        ) {
          anchorNode = {
            nodeId: pickedNode.nodeId,
            segmentId: pickedNode.segmentId,
          };
          StatusMessage.showTemporaryMessage(
            `Selected node ${pickedNode.nodeId} as merge anchor.`,
          );
          updateStatus();
          return;
        }
        if (anchorNode.segmentId === pickedNode.segmentId) {
          anchorNode = {
            nodeId: pickedNode.nodeId,
            segmentId: pickedNode.segmentId,
          };
          StatusMessage.showTemporaryMessage(
            `Merge requires nodes from different skeletons. Node ${pickedNode.nodeId} is now the merge anchor.`,
          );
          updateStatus();
          return;
        }
        const client = this.getCatmaidClient(skeletonLayer);
        if (client === undefined) {
          StatusMessage.showTemporaryMessage(
            "Unable to resolve CATMAID client for the active source.",
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
            const result = await client.mergeSkeletonsWithInfo(
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
            skeletonLayer.mergeSkeletonNodes({
              parentNodeId: winningNode.nodeId,
              childNodeId: losingNode.nodeId,
              resultSegmentId: resultSkeletonId,
              mergedSegmentId: deletedSkeletonId,
            });
            this.updateVisibleSkeletonSegments(
              resultSkeletonId,
              deletedSkeletonId,
            );
            this.layer.selectedSpatialSkeletonNodeId.value = losingNode.nodeId;
            this.layer.markSpatialSkeletonNodeDataChanged();
            anchorNode = undefined;
            const swapSuffix = result.stableAnnotationSwap
              ? " Merge direction was adjusted by CATMAID."
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

class SpatialSkeletonSplitModeTool extends SpatialSkeletonCatmaidToolBase {
  toJSON() {
    return SPATIAL_SKELETON_SPLIT_MODE_TOOL_ID;
  }

  get description() {
    return "skeleton split mode";
  }

  activate(activation: ToolActivation<this>) {
    const reason = this.layer.getSpatialSkeletonActionsDisabledReason();
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
        "Click the node where the skeleton should be split. Root nodes cannot be split.";
    };
    updateStatus();
    activation.bindInputEventMap(SPATIAL_SKELETON_PICK_INPUT_EVENT_MAP);
    this.registerAutoCancelOnDisabled(activation, updateStatus);
    activation.bindAction(
      "spatial-skeleton-pick-node",
      (event: ActionEvent<MouseEvent>) => {
        if (event.detail.button !== 0) return;
        if (
          event.detail.shiftKey ||
          event.detail.ctrlKey ||
          event.detail.altKey ||
          event.detail.metaKey
        ) {
          return;
        }
        if (pendingNodeId !== undefined) return;
        const disabledReason =
          this.layer.getSpatialSkeletonActionsDisabledReason();
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
        const pickedNode = this.resolvePickedNodeForAction(skeletonLayer);
        if (pickedNode === undefined || pickedNode.segmentId === undefined) {
          return;
        }
        const client = this.getCatmaidClient(skeletonLayer);
        if (client === undefined) {
          StatusMessage.showTemporaryMessage(
            "Unable to resolve CATMAID client for the active source.",
          );
          return;
        }
        pendingNodeId = pickedNode.nodeId;
        updateStatus();
        void (async () => {
          try {
            const result = await client.splitSkeletonWithInfo(
              pickedNode.nodeId,
            );
            const newSkeletonId = result.newSkeletonId;
            const existingSkeletonId =
              result.existingSkeletonId ?? pickedNode.segmentId;
            if (newSkeletonId === undefined) {
              throw new Error(
                "CATMAID did not return a new skeleton id for the split.",
              );
            }
            if (existingSkeletonId === undefined) {
              throw new Error(
                "CATMAID did not return the existing skeleton id for the split.",
              );
            }
            skeletonLayer.splitSkeletonAtNode(pickedNode.nodeId, newSkeletonId);
            const segmentationGroupState =
              this.layer.displayState.segmentationGroupState.value;
            segmentationGroupState.visibleSegments.add(
              BigInt(existingSkeletonId),
            );
            segmentationGroupState.visibleSegments.add(BigInt(newSkeletonId));
            this.selectSegmentByNumber(newSkeletonId);
            this.layer.selectedSpatialSkeletonNodeId.value = pickedNode.nodeId;
            this.layer.markSpatialSkeletonNodeDataChanged();
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
