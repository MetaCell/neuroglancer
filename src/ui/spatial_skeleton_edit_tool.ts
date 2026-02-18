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
  // Capture left-drag regardless of modifier keys so panel camera bindings
  // never steal the drag gesture while edit mode is active.
  "at:control?+alt?+meta?+shift?+mousedown0":
    "spatial-skeleton-edit-drag-node",
});

const DRAG_START_DISTANCE_PX = 4;

interface PendingNodeAddSelection {
  segmentId?: number;
  parentNodeId?: number;
  source: "node" | "empty";
  anchorPosition: Float32Array;
  anchorClientX: number;
  anchorClientY: number;
  anchorPanel?: RenderedDataPanel;
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

function formatAxis3(value: Float32Array) {
  return `(${value[0].toFixed(4)}, ${value[1].toFixed(4)}, ${value[2].toFixed(4)})`;
}

export class SpatialSkeletonEditModeTool extends LayerTool<SegmentationUserLayer> {
  private readonly catmaidClients = new Map<string, CatmaidClient>();

  constructor(layer: SegmentationUserLayer) {
    super(layer, true);
  }

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

  private getNodeSelectionRadius() {
    const { layer } = this;
    const levels = layer.displayState.spatialSkeletonGridLevels.value;
    if (levels.length === 0) return 1;
    const gridLevel = Math.min(
      Math.max(layer.displayState.spatialSkeletonGridLevel3d.value, 0),
      levels.length - 1,
    );
    const size = levels[gridLevel].size;
    const spacing = Math.max(Math.min(size.x, size.y, size.z), 1);
    return Math.max(spacing * 0.75, 1);
  }

  private getCatmaidClient(
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
    parentNodeId: number | undefined,
    position: Float32Array,
  ): Promise<CatmaidAddNodeResult> {
    const client = this.getCatmaidClient(skeletonLayer);
    if (client === undefined) {
      throw new Error(
        "Unable to resolve CATMAID client for active spatial skeleton source.",
      );
    }
    const x = Number(position[0]);
    const y = Number(position[1]);
    const z = Number(position[2]);
    const nodeInfo = await client.addNodeWithInfo(0, x, y, z, parentNodeId);
    StatusMessage.showTemporaryMessage(
      `Added node ${nodeInfo.treenodeId} on segment ${nodeInfo.skeletonId}.`,
    );
    return nodeInfo;
  }

  private getActivePixelSize() {
    const pickedLayerName =
      this.mouseState.pickedRenderLayer?.constructor?.name ?? "";
    const use2d = pickedLayerName.includes("SliceView");
    const pixelSize2d = Math.max(
      this.layer.displayState.spatialSkeletonGridPixelSize2d.value,
      1e-6,
    );
    const pixelSize3d = Math.max(
      this.layer.displayState.spatialSkeletonGridPixelSize3d.value,
      1e-6,
    );
    if (use2d) return pixelSize2d;
    return pixelSize3d;
  }

  private getActiveSpatiallyIndexedSkeletonLayer() {
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

  private getInitialFallbackDragAxes(pixelSize: number) {
    const axisX = new Float32Array([pixelSize, 0, 0]);
    const axisY = new Float32Array([0, -pixelSize, 0]);
    const displayDimensions = this.mouseState.displayDimensions;
    const indices = displayDimensions?.displayDimensionIndices;
    if (indices === undefined || indices.length < 2) {
      return { axisX, axisY };
    }
    axisX.fill(0);
    axisY.fill(0);
    const dimX = indices[0];
    const dimY = indices[1];
    if (dimX >= 0 && dimX < 3) {
      axisX[dimX] = pixelSize;
    } else {
      axisX[0] = pixelSize;
    }
    if (dimY >= 0 && dimY < 3) {
      axisY[dimY] = -pixelSize;
    } else {
      axisY[1] = -pixelSize;
    }
    return { axisX, axisY };
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
    const debugLog = (_label: string, _data: Record<string, unknown> = {}) => {};

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
      disableWithMessage("No spatially indexed skeleton source is currently loaded.");
      return;
    }

    layer.spatialSkeletonEditMode.value = true;
    activation.registerDisposer(() => {
      layer.spatialSkeletonEditMode.value = false;
    });
    setDebug("mode", "on");
    setDebug("catmaid", String(layer.isCatmaidSource.value));
    setDebug("maxLodAllowed", String(layer.spatialSkeletonEditModeAllowed.value));
    setDebug("gridLevel2d", String(layer.displayState.spatialSkeletonGridLevel2d.value));
    setDebug("gridLevel3d", String(layer.displayState.spatialSkeletonGridLevel3d.value));
    setDebug(
      "pickedLayer",
      this.mouseState.pickedRenderLayer?.constructor?.name ?? "none",
    );
    setDebug(
      "mousePos",
      formatVec3(this.mouseState.unsnappedPosition),
    );
    debugLog("activation-enabled", {
      gridLevel2d: layer.displayState.spatialSkeletonGridLevel2d.value,
      gridLevel3d: layer.displayState.spatialSkeletonGridLevel3d.value,
      levels: layer.displayState.spatialSkeletonGridLevels.value.length,
    });
    activation.bindInputEventMap(SPATIAL_SKELETON_EDIT_INPUT_EVENT_MAP);
    let pendingAddSelection: PendingNodeAddSelection | undefined;
    const updatePendingDebug = () => {
      if (pendingAddSelection === undefined) {
        setDebug("pendingAdd", "none");
        return;
      }
      const parent =
        pendingAddSelection.parentNodeId === undefined
          ? "none"
          : String(pendingAddSelection.parentNodeId);
      const segment =
        pendingAddSelection.segmentId === undefined
          ? "new"
          : String(pendingAddSelection.segmentId);
      setDebug(
        "pendingAdd",
        `${pendingAddSelection.source} seg=${segment} parent=${parent}`,
      );
    };
    const clearPendingAddSelection = () => {
      pendingAddSelection = undefined;
      updatePendingDebug();
    };
    const setPendingAddSelection = (selection: PendingNodeAddSelection) => {
      pendingAddSelection = selection;
      updatePendingDebug();
    };
    const setReadyStatus = () => {
      if (pendingAddSelection === undefined) {
        setStatus(
          "Edit mode enabled. Left click selects a node or empty space. Second click adds a node. Drag nodes to move.",
        );
      } else {
        const parent =
          pendingAddSelection.parentNodeId === undefined
            ? "none"
            : String(pendingAddSelection.parentNodeId);
        const segment =
          pendingAddSelection.segmentId === undefined
            ? "new"
            : String(pendingAddSelection.segmentId);
        if (pendingAddSelection.parentNodeId === undefined) {
          setStatus(
            "Empty-space selection armed. Click to add a root node.",
          );
        } else {
          setStatus(
            `Selection set (segment ${segment}, parent ${parent}). Click to add node.`,
          );
        }
      }
    };
    setReadyStatus();
    activation.registerEventListener(
      window,
      "mousedown",
      (event: MouseEvent) => {
        if (event.button !== 0) return;
        setDebug("windowMouseDown", formatMouseEvent(event));
      },
      { capture: true },
    );
    activation.registerEventListener(
      window,
      "pointermove",
      () => {
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
            debugLog("auto-cancel-actions-not-allowed", { reason });
          }
          activation.cancel();
        }
      }),
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
        const dragDisabledReason = layer.getSpatialSkeletonActionsDisabledReason();
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
        const getPanelFallbackPosition = (
          panel: RenderedDataPanel,
          targetEvent: MouseEvent,
        ): Float32Array | undefined => {
          const localPosition = layer.localPosition.value;
          if (localPosition.length < 3) return undefined;
          const anchor = new Float32Array([
            Number(localPosition[0]),
            Number(localPosition[1]),
            Number(localPosition[2]),
          ]);
          if (
            !Number.isFinite(anchor[0]) ||
            !Number.isFinite(anchor[1]) ||
            !Number.isFinite(anchor[2])
          ) {
            return undefined;
          }
          const rect = panel.element.getBoundingClientRect();
          const deltaX = targetEvent.clientX - (rect.left + rect.width / 2);
          const deltaY = targetEvent.clientY - (rect.top + rect.height / 2);
          const projected = new Float32Array(anchor);
          panel.translateDataPointByViewportPixels(
            projected as unknown as vec3,
            anchor as unknown as vec3,
            deltaX,
            deltaY,
          );
          if (
            !Number.isFinite(projected[0]) ||
            !Number.isFinite(projected[1]) ||
            !Number.isFinite(projected[2])
          ) {
            return undefined;
          }
          return projected;
        };
        const resolveClickPosition = (
          targetEvent: MouseEvent,
          selection: PendingNodeAddSelection | undefined,
        ) => {
          const directMousePosition = this.getMousePosition();
          if (directMousePosition !== undefined) {
            return directMousePosition;
          }
          if (selection?.anchorPanel !== undefined) {
            const translated = new Float32Array(selection.anchorPosition);
            selection.anchorPanel.translateDataPointByViewportPixels(
              translated as unknown as vec3,
              selection.anchorPosition as unknown as vec3,
              targetEvent.clientX - selection.anchorClientX,
              targetEvent.clientY - selection.anchorClientY,
            );
            if (
              Number.isFinite(translated[0]) &&
              Number.isFinite(translated[1]) &&
              Number.isFinite(translated[2])
            ) {
              return translated;
            }
          }
          if (selection !== undefined) {
            return new Float32Array(selection.anchorPosition);
          }
          if (actionPanel !== undefined) {
            return getPanelFallbackPosition(actionPanel, targetEvent);
          }
          return undefined;
        };
        const mousePosition = this.getMousePosition();
        const clickStartPosition =
          mousePosition ??
          (actionPanel === undefined
            ? undefined
            : getPanelFallbackPosition(actionPanel, event.detail));
        if (
          mousePosition === undefined &&
          clickStartPosition === undefined &&
          pendingAddSelection === undefined
        ) {
          debugLog("click-aborted-no-position");
          return;
        }
        const segmentId = layer.displayState.segmentSelectionState.baseValue;
        const nodeSelectionRadius = this.getNodeSelectionRadius();
        const nodeHit =
          mousePosition === undefined
            ? undefined
            : skeletonLayer.findClosestNode(mousePosition, {
                segmentId,
                maxDistance: nodeSelectionRadius,
              });
        setDebug("dragStartMousePos", formatVec3(clickStartPosition));
        setDebug(
          "nodeHit",
          nodeHit === undefined
            ? "none"
            : `${nodeHit.nodeId} dist2=${nodeHit.distanceSquared.toFixed(3)}`,
        );
        debugLog("primary-action-hit-test", {
          nodeHit: nodeHit?.nodeId,
          segmentId: segmentId?.toString(),
          nodeSelectionRadius,
          clickStartPosition: formatVec3(clickStartPosition),
          pendingAddSelection,
        });

        const selectSegmentByNumber = (value: number) => {
          if (!Number.isFinite(value)) return;
          const rounded = Math.round(value);
          layer.selectSegment(BigInt(rounded), false);
        };

        const commitClickAction = async (commitEvent: MouseEvent) => {
          if (pendingAddSelection !== undefined) {
            const selection = pendingAddSelection;
            if (
              selection.parentNodeId !== undefined &&
              nodeHit?.nodeId === selection.parentNodeId
            ) {
              clearPendingAddSelection();
              layer.selectedSpatialSkeletonNodeId.value = undefined;
              setReadyStatus();
              StatusMessage.showTemporaryMessage(
                `Selection for node ${selection.parentNodeId} canceled.`,
              );
              debugLog("add-selection-canceled", {
                nodeId: selection.parentNodeId,
                reason: "clicked-selected-node-again",
              });
              return;
            }
            const clickPosition = resolveClickPosition(commitEvent, selection);
            if (clickPosition === undefined) {
              StatusMessage.showTemporaryMessage(
                "Unable to resolve add-node position for this click.",
              );
              debugLog("add-node-position-unresolved", {
                selection,
                clientX: commitEvent.clientX,
                clientY: commitEvent.clientY,
              });
              return;
            }
            let committedNode: CatmaidAddNodeResult;
            try {
              committedNode = await this.commitAddNode(
                skeletonLayer,
                selection.parentNodeId,
                clickPosition,
              );
            } catch (error) {
              StatusMessage.showTemporaryMessage(
                "Failed to commit node creation to CATMAID.",
              );
              debugLog("add-node-commit-failed", {
                error: String(error),
                selection,
                clickPosition: formatVec3(clickPosition),
              });
              return;
            }
            const committedSegmentId =
              selection.parentNodeId === undefined
                ? committedNode.skeletonId
                : selection.segmentId ?? committedNode.skeletonId;
            const newNode = skeletonLayer.addNode(clickPosition, {
              segmentId: committedSegmentId,
              parentNodeId: selection.parentNodeId,
              nodeId: committedNode.treenodeId,
            });
            if (newNode === undefined) {
              StatusMessage.showTemporaryMessage(
                `Added node ${committedNode.treenodeId} in CATMAID, but local preview update failed.`,
              );
              debugLog("add-node-local-update-failed", {
                clickPosition: formatVec3(clickPosition),
                selection,
                committedNode,
              });
              layer.selectedSpatialSkeletonNodeId.value = committedNode.treenodeId;
              selectSegmentByNumber(committedSegmentId);
              layer.markSpatialSkeletonNodeDataChanged();
              clearPendingAddSelection();
              setReadyStatus();
              return;
            }
            layer.selectedSpatialSkeletonNodeId.value = newNode.nodeId;
            selectSegmentByNumber(newNode.segmentId);
            layer.markSpatialSkeletonNodeDataChanged();
            debugLog("add-node-committed", {
              nodeId: newNode.nodeId,
              segmentId: newNode.segmentId,
              parentNodeId: selection.parentNodeId,
              position: formatVec3(newNode.position),
            });
            clearPendingAddSelection();
            setReadyStatus();
            return;
          }

          if (nodeHit !== undefined) {
            selectSegmentByNumber(nodeHit.segmentId);
            layer.selectedSpatialSkeletonNodeId.value = nodeHit.nodeId;
            const selectionPanel =
              this.getRenderedDataPanelForEvent(commitEvent) ?? actionPanel;
            setPendingAddSelection({
              segmentId: nodeHit.segmentId,
              parentNodeId: nodeHit.nodeId,
              source: "node",
              anchorPosition: new Float32Array(nodeHit.position),
              anchorClientX: commitEvent.clientX,
              anchorClientY: commitEvent.clientY,
              anchorPanel: selectionPanel,
            });
            StatusMessage.showTemporaryMessage(
              `Selected node ${nodeHit.nodeId}. Click empty space to add a connected node, or click the same node again to cancel.`,
            );
            setReadyStatus();
            return;
          }
          const emptyAnchorPosition =
            clickStartPosition === undefined
              ? resolveClickPosition(commitEvent, undefined)
              : new Float32Array(clickStartPosition);
          if (emptyAnchorPosition === undefined) {
            StatusMessage.showTemporaryMessage(
              "Unable to resolve a world position for empty-space selection.",
            );
            debugLog("empty-selection-position-unresolved", {
              clientX: commitEvent.clientX,
              clientY: commitEvent.clientY,
            });
            return;
          }
          const selectionPanel =
            this.getRenderedDataPanelForEvent(commitEvent) ?? actionPanel;
          setPendingAddSelection({
            segmentId: undefined,
            parentNodeId: undefined,
            source: "empty",
            anchorPosition: emptyAnchorPosition,
            anchorClientX: commitEvent.clientX,
            anchorClientY: commitEvent.clientY,
            anchorPanel: selectionPanel,
          });
          setReadyStatus();
          StatusMessage.showTemporaryMessage(
            "No node selected. Click again to add a root node in empty space.",
          );
          debugLog("click-selection-empty-space", {
            mousePosition: formatVec3(emptyAnchorPosition),
            selectedSegment: "ignored",
          });
        };

        if (nodeHit === undefined) {
          let movementSquared = 0;
          startRelativeMouseDrag(
            event.detail,
            (_event, deltaX, deltaY) => {
              movementSquared += deltaX * deltaX + deltaY * deltaY;
            },
            (finishEvent) => {
              const thresholdSquared = DRAG_START_DISTANCE_PX * DRAG_START_DISTANCE_PX;
              if (movementSquared > thresholdSquared) {
                setReadyStatus();
                setDebug("dragState", "ignored-no-node");
                debugLog("drag-ignored-no-node-hit", { movementSquared });
                return;
              }
              void commitClickAction(finishEvent);
            },
          );
          return;
        }

        setDebug("hit", `node=${nodeHit.nodeId} dist2=${nodeHit.distanceSquared.toFixed(3)}`);
        setDebug("hitPos", formatVec3(nodeHit.position));
        const dragPanel = this.getRenderedDataPanelForEvent(event.detail);
        setDebug("dragPanel", dragPanel?.constructor?.name ?? "none");
        debugLog("drag-hit-node", {
          nodeId: nodeHit.nodeId,
          distanceSquared: nodeHit.distanceSquared,
          hitPosition: formatVec3(nodeHit.position),
          dragPanel: dragPanel?.constructor?.name ?? "none",
        });
        let moved = false;
        let lastPosition = nodeHit.position;
        const dragAnchorPosition =
          new Float32Array(nodeHit.position) as unknown as vec3;
        const panelTranslatedPosition =
          new Float32Array(nodeHit.position) as unknown as vec3;
        let fallbackPosition = new Float32Array(nodeHit.position);
        let previousMousePosition = this.getMousePosition();
        const initialPixelSize = this.getActivePixelSize();
        const initialAxes = this.getInitialFallbackDragAxes(initialPixelSize);
        const fallbackAxisX = initialAxes.axisX;
        const fallbackAxisY = initialAxes.axisY;
        let moveEvents = 0;
        let totalDeltaX = 0;
        let totalDeltaY = 0;
        let fallbackEvents = 0;
        let fallbackReason = "none";
        let dragDistanceSquared = 0;
        let dragActive = false;
        const applyFallbackDragDelta = (deltaX: number, deltaY: number) => {
          const applied = new Float32Array(fallbackPosition);
          applied[0] += fallbackAxisX[0] * deltaX + fallbackAxisY[0] * deltaY;
          applied[1] += fallbackAxisX[1] * deltaX + fallbackAxisY[1] * deltaY;
          applied[2] += fallbackAxisX[2] * deltaX + fallbackAxisY[2] * deltaY;
          fallbackPosition = applied;
          ++fallbackEvents;
          setDebug("fallbackEvents", String(fallbackEvents));
          setDebug("fallbackReason", fallbackReason);
          setDebug("fallbackAxisX", formatAxis3(fallbackAxisX));
          setDebug("fallbackAxisY", formatAxis3(fallbackAxisY));
          return applied;
        };
        const updateFallbackAxesFromSample = (
          from: Float32Array,
          to: Float32Array,
          deltaX: number,
          deltaY: number,
        ) => {
          const denom = deltaX * deltaX + deltaY * deltaY;
          if (denom <= 1e-6) return;
          const worldDx = to[0] - from[0];
          const worldDy = to[1] - from[1];
          const worldDz = to[2] - from[2];
          const scaleX = deltaX / denom;
          const scaleY = deltaY / denom;
          // Exponential smoothing keeps drag stable while adapting to camera orientation.
          const alpha = 0.5;
          const oneMinusAlpha = 1 - alpha;
          const candidateX0 = worldDx * scaleX;
          const candidateX1 = worldDy * scaleX;
          const candidateX2 = worldDz * scaleX;
          const candidateY0 = worldDx * scaleY;
          const candidateY1 = worldDy * scaleY;
          const candidateY2 = worldDz * scaleY;
          fallbackAxisX[0] = fallbackAxisX[0] * oneMinusAlpha + candidateX0 * alpha;
          fallbackAxisX[1] = fallbackAxisX[1] * oneMinusAlpha + candidateX1 * alpha;
          fallbackAxisX[2] = fallbackAxisX[2] * oneMinusAlpha + candidateX2 * alpha;
          fallbackAxisY[0] = fallbackAxisY[0] * oneMinusAlpha + candidateY0 * alpha;
          fallbackAxisY[1] = fallbackAxisY[1] * oneMinusAlpha + candidateY1 * alpha;
          fallbackAxisY[2] = fallbackAxisY[2] * oneMinusAlpha + candidateY2 * alpha;
        };
        const applyMousePosition = (deltaX: number, deltaY: number) => {
          ++moveEvents;
          totalDeltaX += deltaX;
          totalDeltaY += deltaY;
          let appliedPosition: Float32Array | undefined;
          let source = "panel-translation";
          if (dragPanel !== undefined) {
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
              Number.isFinite(panelTranslatedPosition[0]) &&
              Number.isFinite(panelTranslatedPosition[1]) &&
              Number.isFinite(panelTranslatedPosition[2])
            ) {
              appliedPosition = new Float32Array(panelTranslatedPosition);
              fallbackPosition = new Float32Array(appliedPosition);
            }
          }
          if (appliedPosition === undefined) {
            const nextPosition = this.getMousePosition();
            source = "mouse-position";
            if (nextPosition === undefined) {
              fallbackReason = "no-mouse-position";
              appliedPosition = applyFallbackDragDelta(deltaX, deltaY);
              source = "fallback-no-mouse-position";
            } else if (
              previousMousePosition !== undefined &&
              nextPosition[0] === previousMousePosition[0] &&
              nextPosition[1] === previousMousePosition[1] &&
              nextPosition[2] === previousMousePosition[2]
            ) {
              fallbackReason = "stale-mouse-position";
              appliedPosition = applyFallbackDragDelta(deltaX, deltaY);
              source = "fallback-stale-mouse-position";
            } else {
              if (previousMousePosition !== undefined) {
                updateFallbackAxesFromSample(
                  previousMousePosition,
                  nextPosition,
                  deltaX,
                  deltaY,
                );
              }
              fallbackPosition = new Float32Array(nextPosition);
              appliedPosition = nextPosition;
            }
            previousMousePosition = nextPosition ?? previousMousePosition;
          }
          const didMove = skeletonLayer.setNodePosition(
            nodeHit.nodeId,
            appliedPosition,
          );
          if (!didMove) return;
          moved = true;
          lastPosition = appliedPosition;
          setDebug("draggingNode", String(nodeHit.nodeId));
          setDebug("dragMoves", String(moveEvents));
          setDebug("lastAppliedPos", formatVec3(appliedPosition));
          if (moveEvents <= 5 || moveEvents % 20 === 0) {
            debugLog("drag-move", {
              nodeId: nodeHit.nodeId,
              moveEvents,
              deltaX,
              deltaY,
              source,
              appliedPosition: formatVec3(appliedPosition),
            });
          }
        };
        setDebug("dragState", "armed");
        startRelativeMouseDrag(
          event.detail,
          (_event, deltaX, deltaY) => {
            dragDistanceSquared += deltaX * deltaX + deltaY * deltaY;
            const thresholdSquared = DRAG_START_DISTANCE_PX * DRAG_START_DISTANCE_PX;
            if (!dragActive && dragDistanceSquared >= thresholdSquared) {
              dragActive = true;
              setStatus(`Dragging node ${nodeHit.nodeId}`);
              setDebug("dragState", "active");
            }
            if (!dragActive) return;
            applyMousePosition(deltaX, deltaY);
          },
          (finishEvent) => {
            if (!dragActive) {
              setDebug("dragState", "click");
              void commitClickAction(finishEvent);
              return;
            }
            setReadyStatus();
            setDebug("dragState", "idle");
            setDebug("draggingNode", "none");
            debugLog("drag-end", {
              nodeId: nodeHit.nodeId,
              moved,
              moveEvents,
              finalPosition: formatVec3(lastPosition),
            });
            if (moved) {
              layer.markSpatialSkeletonNodeDataChanged();
              void this.commitMoveNode(
                skeletonLayer,
                nodeHit.nodeId,
                lastPosition,
              ).catch((error) => {
                StatusMessage.showTemporaryMessage(
                  `Failed to commit move for node ${nodeHit.nodeId}.`,
                );
                debugLog("move-node-commit-failed", {
                  nodeId: nodeHit.nodeId,
                  error: String(error),
                  position: formatVec3(lastPosition),
                });
              });
            }
            clearPendingAddSelection();
          },
        );
      },
    );
  }
}

class SpatialSkeletonPlaceholderTool extends LayerTool<SegmentationUserLayer> {
  constructor(
    layer: SegmentationUserLayer,
    private readonly toolId: string,
    private readonly modeLabel: string,
    private readonly modeWatchable: { value: boolean },
  ) {
    super(layer, true);
  }

  toJSON() {
    return this.toolId;
  }

  get description() {
    return `skeleton ${this.modeLabel.toLowerCase()} mode`;
  }

  activate(activation: ToolActivation<this>) {
    const reason = this.layer.getSpatialSkeletonActionsDisabledReason();
    if (reason !== undefined) {
      StatusMessage.showTemporaryMessage(reason);
      queueMicrotask(() => activation.cancel());
      return;
    }
    this.modeWatchable.value = true;
    activation.registerDisposer(() => {
      this.modeWatchable.value = false;
    });
    const { body, header } =
      makeToolActivationStatusMessageWithHeader(activation);
    header.textContent = `Spatial skeleton ${this.modeLabel} mode`;
    body.textContent = `${this.modeLabel} mode is not implemented yet.`;
    StatusMessage.showTemporaryMessage(
      `${this.modeLabel} mode is not implemented yet.`,
    );
    activation.registerDisposer(
      this.layer.spatialSkeletonActionsAllowed.changed.add(() => {
        if (this.layer.spatialSkeletonActionsAllowed.value) return;
        const disabledReason = this.layer.getSpatialSkeletonActionsDisabledReason();
        if (disabledReason !== undefined) {
          StatusMessage.showTemporaryMessage(disabledReason);
        }
        activation.cancel();
      }),
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
    return new SpatialSkeletonPlaceholderTool(
      layer,
      SPATIAL_SKELETON_MERGE_MODE_TOOL_ID,
      "Merge",
      layer.spatialSkeletonMergeMode,
    );
  });
  registerTool(contextType, SPATIAL_SKELETON_SPLIT_MODE_TOOL_ID, (layer) => {
    return new SpatialSkeletonPlaceholderTool(
      layer,
      SPATIAL_SKELETON_SPLIT_MODE_TOOL_ID,
      "Split",
      layer.spatialSkeletonSplitMode,
    );
  });
}
