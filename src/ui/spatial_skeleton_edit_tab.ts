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

import svg_arrow_left from "ikonate/icons/arrow-left.svg?raw";
import svg_arrow_right from "ikonate/icons/arrow-right.svg?raw";
import svg_bin from "ikonate/icons/bin.svg?raw";
import svg_chevron_down from "ikonate/icons/chevron-down.svg?raw";
import svg_chevron_right from "ikonate/icons/chevron-right.svg?raw";
import svg_chevron_up from "ikonate/icons/chevron-up.svg?raw";
import svg_chevrons_left from "ikonate/icons/chevrons-left.svg?raw";
import svg_chevrons_right from "ikonate/icons/chevrons-right.svg?raw";
import svg_circle from "ikonate/icons/circle.svg?raw";
import svg_flag from "ikonate/icons/flag.svg?raw";
import svg_minus from "ikonate/icons/minus.svg?raw";
import svg_origin from "ikonate/icons/origin.svg?raw";
import svg_redo from "ikonate/icons/redo.svg?raw";
import svg_retweet from "ikonate/icons/retweet.svg?raw";
import svg_share_android from "ikonate/icons/share-android.svg?raw";
import svg_undo from "ikonate/icons/undo.svg?raw";
import type { SegmentationUserLayer } from "#src/layer/segmentation/index.js";
import { getSpatialSkeletonNodeIdFromViewerHover } from "#src/layer/segmentation/selection.js";
import {
  executeSpatialSkeletonDeleteNode,
  executeSpatialSkeletonNodeTrueEndUpdate,
  redoSpatialSkeletonCommand,
  undoSpatialSkeletonCommand,
} from "#src/layer/segmentation/spatial_skeleton_commands.js";
import { showSpatialSkeletonActionError } from "#src/layer/segmentation/spatial_skeleton_errors.js";
import {
  getSegmentEquivalences,
  getVisibleSegments,
} from "#src/segmentation_display_state/base.js";
import { getBaseObjectColor } from "#src/segmentation_display_state/frontend.js";
import type {
  SpatiallyIndexedSkeletonNavigationTarget,
  SpatiallyIndexedSkeletonNode,
  SpatiallyIndexedSkeletonOpenLeaf,
} from "#src/skeleton/api.js";
import {
  SpatialSkeletonActions,
  type SpatialSkeletonAction,
} from "#src/skeleton/actions.js";
import {
  buildSpatiallyIndexedSkeletonNavigationGraph,
  getBranchEnd as getBranchEndFromGraph,
  getBranchStart as getBranchStartFromGraph,
  getRandomChildNode as getRandomChildNodeFromGraph,
  getNextCollapsedLevelNode as getNextCollapsedLevelNodeFromGraph,
  getOpenLeaves as getOpenLeavesFromGraph,
  getParentNode as getParentNodeFromGraph,
  getSkeletonRootNode as getSkeletonRootNodeFromGraph,
  type SpatiallyIndexedSkeletonNavigationGraph,
} from "#src/skeleton/navigation.js";
import {
  getSpatialSkeletonNodeFilterLabel,
  getSpatialSkeletonNodeIconFilterType,
  SpatialSkeletonNodeFilterType,
  type SpatialSkeletonDisplayNodeType as SkeletonNodeType,
} from "#src/skeleton/node_types.js";
import { StatusMessage } from "#src/status.js";
import { observeWatchable, registerNested } from "#src/trackable_value.js";
import {
  buildSpatialSkeletonSegmentRenderState,
  type SpatialSkeletonSegmentRenderState,
} from "#src/ui/spatial_skeleton_edit_tab_render_state.js";
import {
  SPATIAL_SKELETON_EDIT_MODE_TOOL_ID,
  SPATIAL_SKELETON_MERGE_MODE_TOOL_ID,
  SPATIAL_SKELETON_SPLIT_MODE_TOOL_ID,
} from "#src/ui/spatial_skeleton_edit_tool.js";
import { makeToolButton } from "#src/ui/tool.js";
import { isAbortError } from "#src/util/abort.js";
import { TrackableEnum } from "#src/util/trackable_enum.js";
import { EnumSelectWidget } from "#src/widget/enum_widget.js";
import { makeIcon } from "#src/widget/icon.js";
import { Tab } from "#src/widget/tab_view.js";

const MAX_LISTED_NODES = 10000;

export function syncShownSpatialSkeletonSegmentIds(
  shownSegmentIds: Set<number>,
  nextSegmentIds: readonly number[],
  previousSegmentIds: readonly number[],
) {
  const nextVisibleIds = new Set(nextSegmentIds);
  for (const segmentId of shownSegmentIds) {
    if (!nextVisibleIds.has(segmentId)) {
      shownSegmentIds.delete(segmentId);
    }
  }

  const previousVisibleIds = new Set(previousSegmentIds);
  for (const segmentId of nextSegmentIds) {
    if (!previousVisibleIds.has(segmentId)) {
      shownSegmentIds.add(segmentId);
    }
  }
}

interface SpatiallyIndexedSkeletonNavigationApi {
  getSkeletonRootNode(
    skeletonId: number,
  ): Promise<SpatiallyIndexedSkeletonNavigationTarget>;
  getBranchStart(
    nodeId: number,
  ): Promise<SpatiallyIndexedSkeletonNavigationTarget>;
  getBranchEnd(
    nodeId: number,
  ): Promise<SpatiallyIndexedSkeletonNavigationTarget>;
  getNextCollapsedLevelNode(
    nodeId: number,
  ): Promise<SpatiallyIndexedSkeletonNavigationTarget>;
  getOpenLeaves(
    skeletonId: number,
    nodeId: number,
  ): Promise<SpatiallyIndexedSkeletonOpenLeaf[]>;
  getParentNode(
    nodeId: number,
  ): Promise<SpatiallyIndexedSkeletonNavigationTarget | undefined>;
  getChildNode(
    nodeId: number,
  ): Promise<SpatiallyIndexedSkeletonNavigationTarget | undefined>;
}

const NODE_TYPE_ICONS: Record<SkeletonNodeType, string> = {
  root: svg_origin,
  branchStart: svg_share_android,
  regular: svg_minus,
  virtualEnd: svg_circle,
};

const NODE_TYPE_LABELS: Record<SkeletonNodeType, string> = {
  root: "root",
  branchStart: "branch start",
  regular: "regular",
  virtualEnd: "virtual end",
};

function formatNodeCoordinates(position: ArrayLike<number>) {
  const x = Number(position[0]);
  const y = Number(position[1]);
  const z = Number(position[2]);
  return `${Math.round(x)} ${Math.round(y)} ${Math.round(z)}`;
}

export class SpatialSkeletonEditTab extends Tab {
  constructor(public layer: SegmentationUserLayer) {
    super();
    const { element } = this;
    element.classList.add("neuroglancer-spatial-skeleton-tab");

    const toolbox = document.createElement("div");
    toolbox.className =
      "neuroglancer-segmentation-toolbox neuroglancer-spatial-skeleton-toolbar";
    toolbox.appendChild(
      makeToolButton(this, layer.toolBinder, {
        toolJson: SPATIAL_SKELETON_EDIT_MODE_TOOL_ID,
        label: "Edit",
        title: "Toggle skeleton node edit mode",
      }),
    );
    toolbox.appendChild(
      makeToolButton(this, layer.toolBinder, {
        toolJson: SPATIAL_SKELETON_MERGE_MODE_TOOL_ID,
        label: "Merge",
        title: "Toggle skeleton merge mode",
      }),
    );
    toolbox.appendChild(
      makeToolButton(this, layer.toolBinder, {
        toolJson: SPATIAL_SKELETON_SPLIT_MODE_TOOL_ID,
        label: "Split",
        title: "Toggle skeleton split mode",
      }),
    );
    const toolbarActions = document.createElement("div");
    toolbarActions.className = "neuroglancer-spatial-skeleton-toolbar-actions";

    const makeIconButton = (
      parent: HTMLElement,
      svg: string,
      title: string,
      onClick: () => void,
    ) => {
      const button = document.createElement("button");
      button.className = "neuroglancer-spatial-skeleton-icon-button";
      button.type = "button";
      button.title = title;
      button.setAttribute("aria-label", title);
      button.appendChild(makeIcon({ svg, title, clickable: false }));
      button.addEventListener("click", () => onClick());
      parent.appendChild(button);
      return button;
    };
    const undoButton = makeIconButton(toolbarActions, svg_undo, "Undo", () => {
      if (undoButton.disabled) return;
      void (async () => {
        try {
          await undoSpatialSkeletonCommand(layer);
        } catch (error) {
          showSpatialSkeletonActionError("undo", error);
        }
      })();
    });
    const redoButton = makeIconButton(toolbarActions, svg_redo, "Redo", () => {
      if (redoButton.disabled) return;
      void (async () => {
        try {
          await redoSpatialSkeletonCommand(layer);
        } catch (error) {
          showSpatialSkeletonActionError("redo", error);
        }
      })();
    });
    toolbox.appendChild(toolbarActions);

    const navTools = document.createElement("div");
    navTools.className = "neuroglancer-spatial-skeleton-nav-tools";

    const nodesSection = document.createElement("div");
    nodesSection.className = "neuroglancer-spatial-skeleton-section";
    const collapseButton = document.createElement("button");
    collapseButton.className = "neuroglancer-spatial-skeleton-section-toggle";
    collapseButton.type = "button";
    const filterInput = document.createElement("input");
    filterInput.type = "text";
    filterInput.placeholder = "Enter node ID, coordinates or description";
    filterInput.className = "neuroglancer-spatial-skeleton-filter";
    const nodeFilterTypeModel = new TrackableEnum(
      SpatialSkeletonNodeFilterType,
      SpatialSkeletonNodeFilterType.NONE,
    );
    const nodeFilterTypeWidget = this.registerDisposer(
      new EnumSelectWidget(nodeFilterTypeModel),
    );
    nodeFilterTypeWidget.element.classList.add(
      "neuroglancer-layer-control-control",
      "neuroglancer-spatial-skeleton-filter-select",
    );
    nodeFilterTypeWidget.element.title = "Filter loaded nodes by node type";
    nodeFilterTypeWidget.element.setAttribute(
      "aria-label",
      nodeFilterTypeWidget.element.title,
    );
    for (const option of nodeFilterTypeWidget.element.options) {
      option.textContent = getSpatialSkeletonNodeFilterLabel(
        nodeFilterTypeModel.enumType[
          option.value.toUpperCase()
        ] as SpatialSkeletonNodeFilterType,
      );
    }
    const nodeFilterTypeRow = document.createElement("label");
    nodeFilterTypeRow.className = "neuroglancer-spatial-skeleton-filter-row";
    const nodeFilterTypeLabel = document.createElement("span");
    nodeFilterTypeLabel.className =
      "neuroglancer-spatial-skeleton-filter-label";
    nodeFilterTypeLabel.textContent = "Filter";
    nodeFilterTypeRow.appendChild(nodeFilterTypeLabel);
    nodeFilterTypeRow.appendChild(nodeFilterTypeWidget.element);
    const showFilterSection = document.createElement("div");
    showFilterSection.className = "neuroglancer-spatial-skeleton-show-section";
    const showFilterLabel = document.createElement("div");
    showFilterLabel.className = "neuroglancer-spatial-skeleton-filter-label";
    showFilterLabel.textContent = "Show";
    const showFilterList = document.createElement("div");
    showFilterList.className = "neuroglancer-spatial-skeleton-show-list";
    showFilterSection.appendChild(showFilterLabel);
    showFilterSection.appendChild(showFilterList);
    const nodesNavigationBar = document.createElement("div");
    nodesNavigationBar.className =
      "neuroglancer-spatial-skeleton-navigation-bar";
    const nodesSummaryBar = document.createElement("div");
    nodesSummaryBar.className = "neuroglancer-spatial-skeleton-summary-bar";
    const nodesSummary = document.createElement("div");
    nodesSummary.className = "neuroglancer-spatial-skeleton-summary";
    const nodesList = document.createElement("div");
    nodesList.className = "neuroglancer-spatial-skeleton-tree";
    nodesSection.appendChild(filterInput);
    nodesSection.appendChild(nodeFilterTypeRow);
    nodesSection.appendChild(showFilterSection);
    nodesNavigationBar.appendChild(navTools);
    nodesSection.appendChild(nodesNavigationBar);
    nodesSummaryBar.appendChild(nodesSummary);
    nodesSummaryBar.appendChild(collapseButton);
    nodesSection.appendChild(nodesSummaryBar);
    nodesSection.appendChild(nodesList);
    element.appendChild(nodesSection);

    let allNodes: SpatiallyIndexedSkeletonNode[] = [];
    let activeSegmentIds: number[] = [];
    let nodesBySegment = new Map<number, SpatiallyIndexedSkeletonNode[]>();
    const shownSegmentIds = new Set<number>();
    let filterText = "";
    let nodeFilterType = SpatialSkeletonNodeFilterType.NONE;
    let inspectionAllowed = false;
    let navigationAllowed = false;
    let trueEndEditingAllowed = false;
    let nodeDeletionAllowed = false;
    let nodeRerootAllowed = false;
    let listCollapsed = true;
    let pendingScrollToSelectedNode = false;
    let refreshRequestId = 0;
    let loadedNodeSummarySuffix = "";
    let hoveredViewerNodeId: number | undefined;
    const pendingDeleteNodes = new Set<number>();
    const pendingRerootNodes = new Set<number>();
    const pendingTrueEndNodes = new Set<number>();
    const renderedRowsByNodeId = new Map<number, HTMLDivElement>();
    const renderedEntriesByNodeId = new Map<number, HTMLDivElement>();
    const skeletonState = layer.spatialSkeletonState;
    const mouseState = layer.manager.root.layerSelectedValues.mouseState;
    const navigationGraphCache = new Map<
      number,
      {
        nodes: readonly SpatiallyIndexedSkeletonNode[];
        graph: SpatiallyIndexedSkeletonNavigationGraph;
      }
    >();
    const segmentColorScratch = new Float32Array(4);

    const getSelectedNode = () => {
      const selectedId = layer.selectedSpatialSkeletonNodeId.value;
      if (selectedId === undefined) return undefined;
      return allNodes.find((node) => node.nodeId === selectedId);
    };

    const updateCollapseButton = () => {
      collapseButton.textContent = "";
      collapseButton.title = listCollapsed
        ? "Expand regular nodes"
        : "Collapse regular nodes";
      collapseButton.setAttribute("aria-label", collapseButton.title);
      collapseButton.appendChild(
        makeIcon({
          svg: listCollapsed ? svg_chevron_down : svg_chevron_up,
          title: collapseButton.title,
          clickable: false,
        }),
      );
    };

    const ensureActionsAllowed = (
      requiredActions:
        | SpatialSkeletonAction
        | readonly SpatialSkeletonAction[],
      options: {
        requireVisibleChunks?: boolean;
      } = {},
    ) => {
      const reason = layer.getSpatialSkeletonActionsDisabledReason(
        requiredActions,
        options,
      );
      if (reason !== undefined) {
        StatusMessage.showTemporaryMessage(reason);
        return false;
      }
      return true;
    };

    const selectNode = (
      node: SpatiallyIndexedSkeletonNode | undefined,
      options: {
        moveView?: boolean;
        pin?: boolean;
      } = {},
    ) => {
      if (node === undefined) return;
      const moveView = options.moveView ?? false;
      const pin = options.pin ?? false;
      pendingScrollToSelectedNode = true;
      layer.selectSpatialSkeletonNode(node.nodeId, pin, {
        segmentId: node.segmentId,
        position: node.position,
      });
      if (moveView) {
        moveViewToNodePosition(node.position);
      }
      applyRowInteractionState({ scrollSelectedIntoView: true });
    };

    const moveViewToNodePosition = (position: ArrayLike<number>) => {
      layer.moveViewToSpatialSkeletonNodePosition(position);
    };

    const getNavigationNode = (nodeId: number) => {
      return skeletonState.getCachedNode(nodeId);
    };

    const getSegmentNavigationNodes = (segmentId: number) => {
      return (
        nodesBySegment.get(segmentId) ??
        skeletonState.getCachedSegmentNodes(segmentId)
      );
    };

    const getSegmentNavigationGraph = (segmentId: number) => {
      const segmentNodes = getSegmentNavigationNodes(segmentId);
      if (segmentNodes === undefined || segmentNodes.length === 0) {
        throw new Error(
          `Skeleton graph for segment ${segmentId} is not loaded yet.`,
        );
      }
      const cached = navigationGraphCache.get(segmentId);
      if (cached !== undefined && cached.nodes === segmentNodes) {
        return cached.graph;
      }
      const graph = buildSpatiallyIndexedSkeletonNavigationGraph(segmentNodes);
      navigationGraphCache.set(segmentId, {
        nodes: segmentNodes,
        graph,
      });
      return graph;
    };

    const getSegmentChipColors = (segmentId: number) => {
      const color = getBaseObjectColor(
        layer.displayState,
        BigInt(segmentId),
        segmentColorScratch,
      );
      const r = Math.round(color[0] * 255);
      const g = Math.round(color[1] * 255);
      const b = Math.round(color[2] * 255);
      const luminance = (0.299 * r + 0.587 * g + 0.114 * b) / 255;
      return {
        background: `rgb(${r}, ${g}, ${b})`,
        foreground: luminance > 0.6 ? "#101010" : "#f5f5f5",
      };
    };

    const bindSegmentSelectionControls = (
      element: HTMLElement,
      segmentId: number,
    ) => {
      const id = BigInt(segmentId);
      const hasSegmentSelectionModifiers = (event: MouseEvent) =>
        event.ctrlKey && !event.altKey && !event.metaKey;
      element.addEventListener("mousedown", (event: MouseEvent) => {
        if (event.button !== 2 || !hasSegmentSelectionModifiers(event)) {
          return;
        }
        layer.selectSegment(id, event.shiftKey ? "force-unpin" : true);
        event.preventDefault();
        event.stopPropagation();
      });
      element.addEventListener("contextmenu", (event: MouseEvent) => {
        if (!hasSegmentSelectionModifiers(event)) return;
        if (event.button !== 2) {
          layer.selectSegment(id, event.shiftKey ? "force-unpin" : true);
        }
        event.preventDefault();
        event.stopPropagation();
      });
    };

    const getSegmentSelectionTitle = (segmentId: number) =>
      `segment ${segmentId}\n` +
      "Ctrl+right-click to pin selection\n" +
      "Ctrl+shift+right-click to unpin";

    const getNodeDescriptionText = (node: SpatiallyIndexedSkeletonNode) =>
      layer.getSpatialSkeletonNodeDisplayDescription(node);

    const getHoveredNodeIdFromViewer = () => {
      return getSpatialSkeletonNodeIdFromViewerHover(mouseState, layer);
    };

    const applyRowInteractionState = (
      options: { scrollSelectedIntoView?: boolean } = {},
    ) => {
      const selectedNodeId = layer.selectedSpatialSkeletonNodeId.value;
      let selectedRow: HTMLDivElement | undefined;
      for (const [nodeId, entry] of renderedEntriesByNodeId) {
        const isSelected = nodeId === selectedNodeId;
        const isHovered = nodeId === hoveredViewerNodeId;
        entry.dataset.selected = String(isSelected);
        entry.dataset.viewerHovered = String(isHovered);
        if (isSelected) {
          selectedRow = renderedRowsByNodeId.get(nodeId);
        }
      }
      if (options.scrollSelectedIntoView) {
        pendingScrollToSelectedNode = false;
        selectedRow?.scrollIntoView({ block: "nearest" });
      }
    };

    const updateHoveredViewerNode = () => {
      const nextHoveredNodeId = getHoveredNodeIdFromViewer();
      if (hoveredViewerNodeId === nextHoveredNodeId) return;
      hoveredViewerNodeId = nextHoveredNodeId;
      applyRowInteractionState();
    };

    const skeletonNavigationApi: SpatiallyIndexedSkeletonNavigationApi = {
      async getSkeletonRootNode(skeletonId: number) {
        return getSkeletonRootNodeFromGraph(
          getSegmentNavigationGraph(skeletonId),
        );
      },
      async getBranchStart(nodeId: number) {
        const node = getNavigationNode(nodeId);
        if (node === undefined) {
          throw new Error(
            `Node ${nodeId} is not available in the loaded skeleton cache.`,
          );
        }
        return getBranchStartFromGraph(
          getSegmentNavigationGraph(node.segmentId),
          nodeId,
        );
      },
      async getBranchEnd(nodeId: number) {
        const node = getNavigationNode(nodeId);
        if (node === undefined) {
          throw new Error(
            `Node ${nodeId} is not available in the loaded skeleton cache.`,
          );
        }
        return getBranchEndFromGraph(
          getSegmentNavigationGraph(node.segmentId),
          nodeId,
        );
      },
      async getNextCollapsedLevelNode(nodeId: number) {
        const node = getNavigationNode(nodeId);
        if (node === undefined) {
          throw new Error(
            `Node ${nodeId} is not available in the loaded skeleton cache.`,
          );
        }
        return getNextCollapsedLevelNodeFromGraph(
          getSegmentNavigationGraph(node.segmentId),
          nodeId,
        );
      },
      async getOpenLeaves(skeletonId: number, nodeId: number) {
        return getOpenLeavesFromGraph(
          getSegmentNavigationGraph(skeletonId),
          nodeId,
        );
      },
      async getParentNode(nodeId: number) {
        const node = getNavigationNode(nodeId);
        if (node === undefined) {
          throw new Error(
            `Node ${nodeId} is not available in the loaded skeleton cache.`,
          );
        }
        return getParentNodeFromGraph(
          getSegmentNavigationGraph(node.segmentId),
          nodeId,
        );
      },
      async getChildNode(nodeId: number) {
        const node = getNavigationNode(nodeId);
        if (node === undefined) {
          throw new Error(
            `Node ${nodeId} is not available in the loaded skeleton cache.`,
          );
        }
        return getRandomChildNodeFromGraph(
          getSegmentNavigationGraph(node.segmentId),
          nodeId,
        );
      },
    };

    const navigateToNodeTarget = (target: {
      nodeId: number;
      x: number;
      y: number;
      z: number;
    }) => {
      const existingNode = allNodes.find(
        (node) => node.nodeId === target.nodeId,
      );
      if (existingNode !== undefined) {
        selectNode(existingNode, { moveView: true, pin: true });
        return;
      }
      pendingScrollToSelectedNode = true;
      const position = [target.x, target.y, target.z];
      layer.selectSpatialSkeletonNode(target.nodeId, true, { position });
      moveViewToNodePosition(position);
      updateDisplay();
    };

    const getSelectedNavigationContext = () => {
      if (
        !ensureActionsAllowed(SpatialSkeletonActions.inspect, {
          requireVisibleChunks: false,
        })
      ) {
        return undefined;
      }
      const selectedNode = getSelectedNode();
      if (selectedNode === undefined) {
        StatusMessage.showTemporaryMessage("No skeleton node is selected.");
        return undefined;
      }
      try {
        getSegmentNavigationGraph(selectedNode.segmentId);
        return { selectedNode, skeletonApi: skeletonNavigationApi };
      } catch (error) {
        const message = error instanceof Error ? error.message : String(error);
        StatusMessage.showTemporaryMessage(
          `Unable to resolve the local skeleton graph for navigation: ${message}`,
        );
        return undefined;
      }
    };

    const updateTrueEndLabel = (
      node: SpatiallyIndexedSkeletonNode,
      present: boolean,
    ) => {
      if (!ensureActionsAllowed(SpatialSkeletonActions.editNodeTrueEnd)) return;
      if (pendingTrueEndNodes.has(node.nodeId)) return;
      pendingTrueEndNodes.add(node.nodeId);
      updateDisplay();
      void (async () => {
        try {
          const currentNode = skeletonState.getCachedNode(node.nodeId);
          if (currentNode === undefined) {
            throw new Error(
              `Node ${node.nodeId} is missing from the inspected skeleton cache.`,
            );
          }
          await executeSpatialSkeletonNodeTrueEndUpdate(layer, {
            node: currentNode,
            nextIsTrueEnd: present,
          });
        } catch (error) {
          const message =
            error instanceof Error ? error.message : String(error);
          StatusMessage.showTemporaryMessage(
            `Failed to update true end state: ${message}`,
          );
        } finally {
          pendingTrueEndNodes.delete(node.nodeId);
          updateDisplay();
        }
      })();
    };

    const goToClosestUnfinishedBranch = () => {
      const context = getSelectedNavigationContext();
      if (context === undefined) return;
      const { selectedNode, skeletonApi } = context;
      void (async () => {
        try {
          const openLeaves = await skeletonApi.getOpenLeaves(
            selectedNode.segmentId,
            selectedNode.nodeId,
          );
          if (openLeaves.length === 0) {
            StatusMessage.showTemporaryMessage(
              "No unfinished branch was found in the current skeleton.",
            );
            return;
          }
          openLeaves.sort((a, b) =>
            a.distance === b.distance
              ? a.nodeId - b.nodeId
              : a.distance - b.distance,
          );
          navigateToNodeTarget(openLeaves[0]);
        } catch (error) {
          const message =
            error instanceof Error ? error.message : String(error);
          StatusMessage.showTemporaryMessage(
            `Failed to locate unfinished branch: ${message}`,
          );
        }
      })();
    };

    const deleteNode = (node: SpatiallyIndexedSkeletonNode) => {
      if (!ensureActionsAllowed(SpatialSkeletonActions.deleteNodes)) return;
      if (pendingDeleteNodes.has(node.nodeId)) {
        return;
      }
      const segmentNodes = nodesBySegment.get(node.segmentId) ?? [];
      const hasChildren = segmentNodes.some(
        (candidate) => candidate.parentNodeId === node.nodeId,
      );
      if (node.parentNodeId === undefined && hasChildren) {
        StatusMessage.showTemporaryMessage(
          "Reroot the skeleton manually before deleting the current root node.",
        );
        return;
      }
      pendingDeleteNodes.add(node.nodeId);
      updateDisplay();
      void (async () => {
        try {
          await executeSpatialSkeletonDeleteNode(layer, node);
          refreshNodes();
        } catch (error) {
          showSpatialSkeletonActionError("delete node", error);
          updateDisplay();
        } finally {
          pendingDeleteNodes.delete(node.nodeId);
          updateDisplay();
        }
      })();
    };

    const rerootNode = (node: SpatiallyIndexedSkeletonNode) => {
      if (
        !ensureActionsAllowed(SpatialSkeletonActions.reroot, {
          requireVisibleChunks: false,
        })
      ) {
        return;
      }
      if (node.parentNodeId === undefined) {
        StatusMessage.showTemporaryMessage("Selected node is already root.");
        return;
      }
      if (pendingRerootNodes.has(node.nodeId)) {
        return;
      }
      pendingRerootNodes.add(node.nodeId);
      updateDisplay();
      void (async () => {
        try {
          await layer.rerootSpatialSkeletonNode(node);
        } catch (error) {
          showSpatialSkeletonActionError("set node as root", error);
        } finally {
          pendingRerootNodes.delete(node.nodeId);
          updateDisplay();
        }
      })();
    };

    const goRootButton = makeIconButton(
      navTools,
      svg_origin,
      "Go to root",
      () => {
        const context = getSelectedNavigationContext();
        if (context === undefined) return;
        const { selectedNode, skeletonApi } = context;
        void (async () => {
          try {
            navigateToNodeTarget(
              await skeletonApi.getSkeletonRootNode(selectedNode.segmentId),
            );
          } catch (error) {
            const message =
              error instanceof Error ? error.message : String(error);
            StatusMessage.showTemporaryMessage(
              `Failed to locate skeleton root: ${message}`,
            );
          }
        })();
      },
    );
    const goBranchStartButton = makeIconButton(
      navTools,
      svg_chevrons_left,
      "Go to start of the branch",
      () => {
        const context = getSelectedNavigationContext();
        if (context === undefined) return;
        const { selectedNode, skeletonApi } = context;
        void (async () => {
          try {
            navigateToNodeTarget(
              await skeletonApi.getBranchStart(selectedNode.nodeId),
            );
          } catch (error) {
            const message =
              error instanceof Error ? error.message : String(error);
            StatusMessage.showTemporaryMessage(
              `Failed to locate branch start: ${message}`,
            );
          }
        })();
      },
    );
    const goTreeEndButton = makeIconButton(
      navTools,
      svg_chevrons_right,
      "Go to end of the branch",
      () => {
        const context = getSelectedNavigationContext();
        if (context === undefined) return;
        const { selectedNode, skeletonApi } = context;
        void (async () => {
          try {
            navigateToNodeTarget(
              await skeletonApi.getBranchEnd(selectedNode.nodeId),
            );
          } catch (error) {
            const message =
              error instanceof Error ? error.message : String(error);
            StatusMessage.showTemporaryMessage(
              `Failed to locate branch end: ${message}`,
            );
          }
        })();
      },
    );
    const cycleBranchesButton = makeIconButton(
      navTools,
      svg_retweet,
      "Cycle through level nodes",
      () => {
        const context = getSelectedNavigationContext();
        if (context === undefined) return;
        const { selectedNode, skeletonApi } = context;
        void (async () => {
          try {
            navigateToNodeTarget(
              await skeletonApi.getNextCollapsedLevelNode(selectedNode.nodeId),
            );
          } catch (error) {
            const message =
              error instanceof Error ? error.message : String(error);
            StatusMessage.showTemporaryMessage(
              `Failed to cycle through level nodes: ${message}`,
            );
          }
        })();
      },
    );
    const goParentButton = makeIconButton(
      navTools,
      svg_arrow_left,
      "Go to parent",
      () => {
        const context = getSelectedNavigationContext();
        if (context === undefined) return;
        const { selectedNode, skeletonApi } = context;
        void (async () => {
          try {
            const target = await skeletonApi.getParentNode(selectedNode.nodeId);
            if (target === undefined) {
              StatusMessage.showTemporaryMessage(
                "Selected node has no parent.",
              );
              return;
            }
            navigateToNodeTarget(target);
          } catch (error) {
            const message =
              error instanceof Error ? error.message : String(error);
            StatusMessage.showTemporaryMessage(
              `Failed to locate parent node: ${message}`,
            );
          }
        })();
      },
    );
    const goChildButton = makeIconButton(
      navTools,
      svg_arrow_right,
      "Go to child",
      () => {
        const context = getSelectedNavigationContext();
        if (context === undefined) return;
        const { selectedNode, skeletonApi } = context;
        void (async () => {
          try {
            const target = await skeletonApi.getChildNode(selectedNode.nodeId);
            if (target === undefined) {
              StatusMessage.showTemporaryMessage("Selected node has no child.");
              return;
            }
            navigateToNodeTarget(target);
          } catch (error) {
            const message =
              error instanceof Error ? error.message : String(error);
            StatusMessage.showTemporaryMessage(
              `Failed to locate child node: ${message}`,
            );
          }
        })();
      },
    );
    const goUnfinishedBranchButton = makeIconButton(
      navTools,
      svg_chevron_right,
      "Go to unfinished node",
      () => {
        goToClosestUnfinishedBranch();
      },
    );
    element.insertBefore(toolbox, nodesSection);

    const gatedControls = [
      goRootButton,
      goBranchStartButton,
      goTreeEndButton,
      cycleBranchesButton,
      goParentButton,
      goChildButton,
      goUnfinishedBranchButton,
    ];

    const makeRowActionButton = (
      svg: string,
      title: string,
      onClick: () => void,
      disabled: boolean,
    ) => {
      const button = document.createElement("button");
      button.className = "neuroglancer-spatial-skeleton-node-action";
      button.type = "button";
      button.title = title;
      button.disabled = disabled;
      button.appendChild(makeIcon({ svg, title, clickable: false }));
      button.addEventListener("click", (event: MouseEvent) => {
        event.stopPropagation();
        onClick();
      });
      return button;
    };

    type SegmentDisplayState = SpatialSkeletonSegmentRenderState & {
      segmentLabel: string | undefined;
      shown: boolean;
    };

    const getSegmentDisplayLabel = (segmentId: number) => {
      const segmentationGroupState =
        layer.displayState.segmentationGroupState.value;
      const segmentPropertyMap =
        segmentationGroupState.segmentPropertyMap.value;
      if (segmentPropertyMap === undefined) {
        return undefined;
      }
      const mappedSegmentId = getSegmentEquivalences(
        segmentationGroupState,
      ).get(BigInt(segmentId));
      return segmentPropertyMap.getSegmentLabel(mappedSegmentId);
    };

    const buildSegmentDisplayStates = (): SegmentDisplayState[] => {
      const states: SegmentDisplayState[] = [];
      for (const segmentId of activeSegmentIds) {
        const segmentNodes = nodesBySegment.get(segmentId) ?? [];
        const renderState =
          segmentNodes.length === 0
            ? {
                segmentId,
                totalNodeCount: 0,
                matchedNodeCount: 0,
                displayedNodeCount: 0,
                branchCount: 0,
                rows: [],
              }
            : buildSpatialSkeletonSegmentRenderState(
                segmentId,
                getSegmentNavigationGraph(segmentId),
                {
                  filterText,
                  nodeFilterType,
                  collapseRegularNodes: listCollapsed,
                  getNodeDescription: getNodeDescriptionText,
                },
              );
        states.push({
          ...renderState,
          segmentLabel: getSegmentDisplayLabel(segmentId),
          shown: shownSegmentIds.has(segmentId),
        });
      }
      return states;
    };

    const getVisibleSegmentStates = (
      segmentStates: readonly SegmentDisplayState[],
    ) =>
      segmentStates.filter(
        (segmentState) =>
          segmentState.shown && segmentState.displayedNodeCount > 0,
      );

    const updateShowFilter = (
      segmentStates: readonly SegmentDisplayState[],
    ) => {
      showFilterList.textContent = "";
      showFilterSection.hidden = segmentStates.length === 0;
      for (const segmentState of segmentStates) {
        const item = document.createElement("label");
        item.className = "neuroglancer-spatial-skeleton-show-item";

        const checkbox = document.createElement("input");
        checkbox.className = "neuroglancer-spatial-skeleton-show-checkbox";
        checkbox.type = "checkbox";
        checkbox.checked = segmentState.shown;
        checkbox.disabled = segmentState.totalNodeCount === 0;
        checkbox.addEventListener("change", () => {
          if (checkbox.checked) {
            shownSegmentIds.add(segmentState.segmentId);
          } else {
            shownSegmentIds.delete(segmentState.segmentId);
          }
          updateDisplay();
        });

        const content = document.createElement("div");
        content.className = "neuroglancer-spatial-skeleton-show-item-content";
        const segmentChip = document.createElement("span");
        segmentChip.className =
          "neuroglancer-spatial-skeleton-node-segment-chip";
        const segmentChipColors = getSegmentChipColors(segmentState.segmentId);
        segmentChip.textContent = String(segmentState.segmentId);
        segmentChip.style.backgroundColor = segmentChipColors.background;
        segmentChip.style.color = segmentChipColors.foreground;
        segmentChip.title = getSegmentSelectionTitle(segmentState.segmentId);
        bindSegmentSelectionControls(segmentChip, segmentState.segmentId);
        const segmentName = document.createElement("span");
        segmentName.className = "neuroglancer-spatial-skeleton-show-item-name";
        segmentName.textContent = segmentState.segmentLabel ?? "";
        const segmentRatio = document.createElement("span");
        segmentRatio.className =
          "neuroglancer-spatial-skeleton-show-item-ratio";
        segmentRatio.textContent = `${segmentState.displayedNodeCount}/${segmentState.totalNodeCount}`;

        content.appendChild(segmentChip);
        content.appendChild(segmentName);
        content.appendChild(segmentRatio);
        item.appendChild(checkbox);
        item.appendChild(content);
        showFilterList.appendChild(item);
      }
    };

    const updateList = (segmentStates: readonly SegmentDisplayState[]) => {
      nodesList.textContent = "";
      renderedRowsByNodeId.clear();
      renderedEntriesByNodeId.clear();
      if (activeSegmentIds.length === 0) {
        return;
      }
      const visibleSegmentStates = getVisibleSegmentStates(segmentStates);
      if (visibleSegmentStates.length > 0) {
        const listHeader = document.createElement("div");
        listHeader.className = "neuroglancer-spatial-skeleton-list-header";
        const headerActionsSpacer = document.createElement("span");
        headerActionsSpacer.className =
          "neuroglancer-spatial-skeleton-list-header-spacer neuroglancer-spatial-skeleton-list-header-actions";
        const headerTypeSpacer = document.createElement("span");
        headerTypeSpacer.className =
          "neuroglancer-spatial-skeleton-list-header-spacer neuroglancer-spatial-skeleton-list-header-type";
        const headerId = document.createElement("span");
        headerId.className = "neuroglancer-spatial-skeleton-list-header-cell";
        headerId.textContent = "id";
        const headerCoordinates = document.createElement("span");
        headerCoordinates.className =
          "neuroglancer-spatial-skeleton-list-header-cell";
        headerCoordinates.textContent = "coordinates";
        listHeader.appendChild(headerActionsSpacer);
        listHeader.appendChild(headerTypeSpacer);
        listHeader.appendChild(headerId);
        listHeader.appendChild(headerCoordinates);
        nodesList.appendChild(listHeader);
      }

      let renderedNodeCount = 0;
      let overflowNodeCount = 0;
      for (const segmentState of visibleSegmentStates) {
        const segmentEntry = document.createElement("div");
        segmentEntry.className =
          "neuroglancer-spatial-skeleton-tree-entry neuroglancer-spatial-skeleton-segment-entry";
        const segmentRow = document.createElement("div");
        segmentRow.className =
          "neuroglancer-spatial-skeleton-tree-row neuroglancer-spatial-skeleton-segment-row";
        const segmentActionsSpacer = document.createElement("span");
        segmentActionsSpacer.className =
          "neuroglancer-spatial-skeleton-list-header-spacer neuroglancer-spatial-skeleton-list-header-actions";
        const segmentTypeSpacer = document.createElement("span");
        segmentTypeSpacer.className =
          "neuroglancer-spatial-skeleton-list-header-spacer neuroglancer-spatial-skeleton-list-header-type";
        const segmentIdCell = document.createElement("span");
        segmentIdCell.className = "neuroglancer-spatial-skeleton-node-id";
        const segmentChip = document.createElement("span");
        segmentChip.className =
          "neuroglancer-spatial-skeleton-node-segment-chip";
        const segmentChipColors = getSegmentChipColors(segmentState.segmentId);
        segmentChip.textContent = String(segmentState.segmentId);
        segmentChip.style.backgroundColor = segmentChipColors.background;
        segmentChip.style.color = segmentChipColors.foreground;
        segmentChip.title = getSegmentSelectionTitle(segmentState.segmentId);
        bindSegmentSelectionControls(segmentChip, segmentState.segmentId);
        segmentIdCell.appendChild(segmentChip);
        const segmentMeta = document.createElement("div");
        segmentMeta.className =
          "neuroglancer-spatial-skeleton-node-coordinate-cell neuroglancer-spatial-skeleton-segment-meta";
        const segmentMetaLine = document.createElement("div");
        segmentMetaLine.className =
          "neuroglancer-spatial-skeleton-segment-meta-line";
        const segmentName = document.createElement("span");
        segmentName.className = "neuroglancer-spatial-skeleton-segment-name";
        segmentName.textContent = segmentState.segmentLabel ?? "";
        const segmentRatio = document.createElement("span");
        segmentRatio.className = "neuroglancer-spatial-skeleton-segment-ratio";
        segmentRatio.textContent = `${segmentState.displayedNodeCount}/${segmentState.totalNodeCount}`;
        segmentMetaLine.appendChild(segmentName);
        segmentMetaLine.appendChild(segmentRatio);
        segmentMeta.appendChild(segmentMetaLine);
        segmentRow.appendChild(segmentActionsSpacer);
        segmentRow.appendChild(segmentTypeSpacer);
        segmentRow.appendChild(segmentIdCell);
        segmentRow.appendChild(segmentMeta);
        segmentEntry.appendChild(segmentRow);
        nodesList.appendChild(segmentEntry);

        for (const { node, type, isLeaf } of segmentState.rows) {
          if (renderedNodeCount >= MAX_LISTED_NODES) {
            overflowNodeCount++;
            continue;
          }
          renderedNodeCount++;
          const entry = document.createElement("div");
          entry.className = "neuroglancer-spatial-skeleton-tree-entry";
          renderedEntriesByNodeId.set(node.nodeId, entry);

          const row = document.createElement("div");
          row.className = "neuroglancer-spatial-skeleton-tree-row";
          row.dataset.nodeType = type;
          renderedRowsByNodeId.set(node.nodeId, row);
          if (inspectionAllowed) {
            row.tabIndex = 0;
            row.setAttribute("role", "button");
            row.title =
              "Click to move to node and pin selection. Right-click to move to node. Ctrl+right-click to pin selection without moving.";
            row.addEventListener("click", (event: MouseEvent) => {
              const target = event.target;
              if (
                target instanceof HTMLElement &&
                target.closest(
                  ".neuroglancer-spatial-skeleton-node-actions, .neuroglancer-spatial-skeleton-node-type-toggle",
                ) !== null
              ) {
                return;
              }
              if (
                !ensureActionsAllowed(SpatialSkeletonActions.inspect, {
                  requireVisibleChunks: false,
                })
              ) {
                return;
              }
              selectNode(node, { moveView: true, pin: true });
            });
            row.addEventListener("contextmenu", (event: MouseEvent) => {
              const target = event.target;
              if (
                target instanceof HTMLElement &&
                target.closest(
                  ".neuroglancer-spatial-skeleton-node-actions, .neuroglancer-spatial-skeleton-node-type-toggle",
                ) !== null
              ) {
                return;
              }
              event.preventDefault();
              if (
                !ensureActionsAllowed(SpatialSkeletonActions.inspect, {
                  requireVisibleChunks: false,
                })
              ) {
                return;
              }
              if (event.ctrlKey || event.metaKey) {
                selectNode(node, { moveView: false, pin: true });
                return;
              }
              moveViewToNodePosition(node.position);
            });
            row.addEventListener("keydown", (event: KeyboardEvent) => {
              if (event.key !== "Enter" && event.key !== " ") return;
              event.preventDefault();
              if (
                !ensureActionsAllowed(SpatialSkeletonActions.inspect, {
                  requireVisibleChunks: false,
                })
              ) {
                return;
              }
              selectNode(node, { moveView: true, pin: true });
            });
          } else {
            row.setAttribute("aria-disabled", "true");
          }

          const nodeIsTrueEnd = node.isTrueEnd;
          const iconFilterType = getSpatialSkeletonNodeIconFilterType({
            nodeIsTrueEnd,
            nodeType: type,
          });
          const typeIconSvg =
            iconFilterType === SpatialSkeletonNodeFilterType.TRUE_END
              ? svg_flag
              : iconFilterType === SpatialSkeletonNodeFilterType.VIRTUAL_END
                ? svg_circle
                : NODE_TYPE_ICONS[type];
          const typeIconTitle =
            iconFilterType !== undefined
              ? getSpatialSkeletonNodeFilterLabel(iconFilterType).toLowerCase()
              : NODE_TYPE_LABELS[type];
          const typeButtonPending = pendingTrueEndNodes.has(node.nodeId);
          const typeButtonTitle = typeButtonPending
            ? nodeIsTrueEnd
              ? "removing true end"
              : "setting true end"
            : typeIconTitle;
          const typeIcon =
            isLeaf || nodeIsTrueEnd
              ? document.createElement("button")
              : document.createElement("span");
          typeIcon.className =
            isLeaf || nodeIsTrueEnd
              ? "neuroglancer-spatial-skeleton-node-type-toggle"
              : "neuroglancer-spatial-skeleton-node-type";
          typeIcon.title = typeButtonTitle;
          if (typeIcon instanceof HTMLButtonElement) {
            typeIcon.type = "button";
            typeIcon.disabled = !trueEndEditingAllowed || typeButtonPending;
            typeIcon.setAttribute("aria-pressed", String(nodeIsTrueEnd));
            typeIcon.addEventListener("click", (event: MouseEvent) => {
              event.stopPropagation();
              updateTrueEndLabel(node, !nodeIsTrueEnd);
            });
          }
          typeIcon.appendChild(
            makeIcon({
              svg: typeIconSvg,
              title: typeButtonTitle,
              clickable: false,
            }),
          );

          const idCell = document.createElement("span");
          idCell.className = "neuroglancer-spatial-skeleton-node-id";
          idCell.textContent = String(node.nodeId);

          const coordinatesCell = document.createElement("div");
          coordinatesCell.className =
            "neuroglancer-spatial-skeleton-node-coordinate-cell";
          const coordinatesLine = document.createElement("div");
          coordinatesLine.className =
            "neuroglancer-spatial-skeleton-node-coordinates";
          coordinatesLine.textContent = formatNodeCoordinates(node.position);
          coordinatesCell.appendChild(coordinatesLine);
          const description = getNodeDescriptionText(node);
          if (description !== undefined) {
            const descriptionLine = document.createElement("div");
            descriptionLine.className =
              "neuroglancer-spatial-skeleton-node-description";
            descriptionLine.textContent = description;
            coordinatesCell.appendChild(descriptionLine);
          }

          const actions = document.createElement("div");
          actions.className = "neuroglancer-spatial-skeleton-node-actions";
          let rerootActionTitle =
            node.parentNodeId === undefined ? "already root" : "set as root";
          if (pendingRerootNodes.has(node.nodeId)) {
            rerootActionTitle = "setting root";
          }
          actions.appendChild(
            makeRowActionButton(
              svg_origin,
              rerootActionTitle,
              () => rerootNode(node),
              !nodeRerootAllowed ||
                pendingRerootNodes.has(node.nodeId) ||
                node.parentNodeId === undefined,
            ),
          );
          let deleteActionTitle = "delete node";
          if (pendingDeleteNodes.has(node.nodeId)) {
            deleteActionTitle = "deleting node";
          }
          actions.appendChild(
            makeRowActionButton(
              svg_bin,
              deleteActionTitle,
              () => deleteNode(node),
              !nodeDeletionAllowed || pendingDeleteNodes.has(node.nodeId),
            ),
          );

          row.appendChild(actions);
          row.appendChild(typeIcon);
          row.appendChild(idCell);
          row.appendChild(coordinatesCell);
          entry.appendChild(row);

          nodesList.appendChild(entry);
        }
      }

      if (visibleSegmentStates.length === 0) {
        const empty = document.createElement("div");
        empty.className = "neuroglancer-spatial-skeleton-summary";
        empty.textContent =
          shownSegmentIds.size === 0
            ? "Select one or more skeletons to show."
            : filterText.length === 0 &&
                nodeFilterType === SpatialSkeletonNodeFilterType.NONE
              ? "No loaded nodes."
              : "No matching nodes.";
        nodesList.appendChild(empty);
      }
      if (overflowNodeCount > 0) {
        const more = document.createElement("div");
        more.className = "neuroglancer-spatial-skeleton-summary";
        more.textContent = `Showing first ${MAX_LISTED_NODES} nodes`;
        nodesList.appendChild(more);
      }
      if (pendingScrollToSelectedNode) {
        applyRowInteractionState({ scrollSelectedIntoView: true });
      } else {
        applyRowInteractionState();
      }
    };

    const summarizeNodeState = (
      segmentStates: readonly SegmentDisplayState[],
      summarySuffix = "",
    ) => {
      const visibleSegmentStates = getVisibleSegmentStates(segmentStates);
      const skeletonCount = visibleSegmentStates.length;
      let branchCount = 0;
      let nodeCount = 0;
      for (const segmentState of visibleSegmentStates) {
        branchCount += segmentState.branchCount;
        nodeCount += segmentState.displayedNodeCount;
      }
      nodesSummary.textContent = `${skeletonCount} skeleton${
        skeletonCount === 1 ? "" : "s"
      }, ${branchCount} branch${branchCount === 1 ? "" : "es"}, ${nodeCount} node${
        nodeCount === 1 ? "" : "s"
      }`;
      if (summarySuffix.trim().length > 0) {
        nodesSummary.title = summarySuffix.trim();
      } else {
        nodesSummary.removeAttribute("title");
      }
    };

    const updateDisplay = (summarySuffix = loadedNodeSummarySuffix) => {
      if (activeSegmentIds.length === 0) {
        showFilterList.textContent = "";
        showFilterSection.hidden = true;
        nodesList.textContent = "";
        renderedRowsByNodeId.clear();
        renderedEntriesByNodeId.clear();
        return;
      }
      const segmentStates = buildSegmentDisplayStates();
      updateShowFilter(segmentStates);
      summarizeNodeState(segmentStates, summarySuffix);
      updateList(segmentStates);
    };

    const applyNodesBySegment = (
      nextNodesBySegment: Map<number, SpatiallyIndexedSkeletonNode[]>,
      summarySuffix = "",
    ) => {
      loadedNodeSummarySuffix = summarySuffix;
      navigationGraphCache.clear();
      nodesBySegment = nextNodesBySegment;
      const allNodesById = new Map<number, SpatiallyIndexedSkeletonNode>();
      for (const segmentNodes of nextNodesBySegment.values()) {
        for (const node of segmentNodes) {
          if (!allNodesById.has(node.nodeId)) {
            allNodesById.set(node.nodeId, node);
          }
        }
      }
      allNodes = [...allNodesById.values()].sort((a, b) =>
        a.segmentId === b.segmentId
          ? a.nodeId - b.nodeId
          : a.segmentId - b.segmentId,
      );
      const selectedId = layer.selectedSpatialSkeletonNodeId.value;
      if (
        selectedId === undefined ||
        !allNodes.some((node) => node.nodeId === selectedId)
      ) {
        if (allNodes.length > 0) {
          layer.selectSpatialSkeletonNode(allNodes[0].nodeId, false, {
            segmentId: allNodes[0].segmentId,
          });
        } else {
          layer.clearSpatialSkeletonNodeSelection(false);
        }
      }
      updateDisplay(summarySuffix);
    };

    const refreshNodes = () => {
      const requestId = ++refreshRequestId;
      const skeletonLayer = layer.getSpatiallyIndexedSkeletonLayer();
      const nextActiveSegmentIds = [
        ...getVisibleSegments(
          layer.displayState.segmentationGroupState.value,
        ).keys(),
      ]
        .map((segmentId) => Number(segmentId))
        .filter((segmentId) => Number.isFinite(segmentId))
        .sort((a, b) => a - b);
      syncShownSpatialSkeletonSegmentIds(
        shownSegmentIds,
        nextActiveSegmentIds,
        activeSegmentIds,
      );
      activeSegmentIds = nextActiveSegmentIds;
      if (skeletonLayer === undefined || activeSegmentIds.length === 0) {
        allNodes = [];
        nodesBySegment = new Map();
        shownSegmentIds.clear();
        layer.clearSpatialSkeletonNodeSelection(false);
        nodesSummary.removeAttribute("title");
        nodesSummary.textContent =
          "Make one or more segments visible in Seg tab to load editable skeleton nodes.";
        updateDisplay();
        return;
      }

      const cachedSegmentIds = new Set<number>(activeSegmentIds);
      for (const retainedSegmentId of skeletonLayer.getRetainedOverlaySegmentIds()) {
        cachedSegmentIds.add(retainedSegmentId);
      }
      skeletonState.evictInactiveSegmentNodes(cachedSegmentIds);

      void (async () => {
        try {
          const fetchedSegments = await Promise.all(
            activeSegmentIds.map(
              async (segmentId) =>
                [
                  segmentId,
                  await skeletonState.getFullSegmentNodes(
                    skeletonLayer,
                    segmentId,
                  ),
                ] as const,
            ),
          );
          if (requestId !== refreshRequestId) {
            return;
          }
          const nextNodesBySegment = new Map<
            number,
            SpatiallyIndexedSkeletonNode[]
          >(fetchedSegments);
          applyNodesBySegment(
            nextNodesBySegment,
            " Using source-backed full skeleton data.",
          );
        } catch (error) {
          if (requestId !== refreshRequestId) {
            return;
          }
          if (isAbortError(error)) {
            refreshNodes();
            return;
          }
          const message =
            error instanceof Error ? error.message : String(error);
          StatusMessage.showTemporaryMessage(
            `Failed to load full skeleton data from the active source: ${message}`,
          );
          allNodes = [];
          nodesBySegment = new Map();
          layer.clearSpatialSkeletonNodeSelection(false);
          nodesSummary.removeAttribute("title");
          nodesSummary.textContent =
            "Failed to load full skeleton data from the active source.";
          updateDisplay();
        }
      })();
    };

    const updateGateStatus = () => {
      const nextInspectionAllowed =
        layer.getSpatialSkeletonActionsDisabledReason(
          SpatialSkeletonActions.inspect,
          {
            requireVisibleChunks: false,
          },
        ) === undefined;
      const nextNavigationAllowed = nextInspectionAllowed;
      const nextTrueEndEditingAllowed =
        layer.getSpatialSkeletonActionsDisabledReason(
          SpatialSkeletonActions.editNodeTrueEnd,
        ) ===
        undefined;
      const nextNodeDeletionAllowed =
        layer.getSpatialSkeletonActionsDisabledReason(
          SpatialSkeletonActions.deleteNodes,
        ) ===
        undefined;
      const nextNodeRerootAllowed =
        layer.getSpatialSkeletonActionsDisabledReason(
          SpatialSkeletonActions.reroot,
          {
            requireVisibleChunks: false,
          },
        ) === undefined;
      const gateStateChanged =
        inspectionAllowed !== nextInspectionAllowed ||
        navigationAllowed !== nextNavigationAllowed ||
        trueEndEditingAllowed !== nextTrueEndEditingAllowed ||
        nodeDeletionAllowed !== nextNodeDeletionAllowed ||
        nodeRerootAllowed !== nextNodeRerootAllowed;

      inspectionAllowed = nextInspectionAllowed;
      navigationAllowed = nextNavigationAllowed;
      trueEndEditingAllowed = nextTrueEndEditingAllowed;
      nodeDeletionAllowed = nextNodeDeletionAllowed;
      nodeRerootAllowed = nextNodeRerootAllowed;

      filterInput.disabled = !inspectionAllowed;
      nodeFilterTypeWidget.element.disabled = !inspectionAllowed;
      for (const control of gatedControls) {
        control.disabled = !navigationAllowed;
      }
      if (gateStateChanged) {
        updateDisplay();
      }
    };

    const updateHistoryButtons = () => {
      const { commandHistory } = layer.spatialSkeletonState;
      const undoLabel = commandHistory.undoLabel.value;
      const redoLabel = commandHistory.redoLabel.value;
      const busy = commandHistory.isBusy.value;
      undoButton.disabled = busy || !commandHistory.canUndo.value;
      redoButton.disabled = busy || !commandHistory.canRedo.value;
      undoButton.title = busy
        ? "Wait for the current skeleton edit to finish."
        : undoLabel === undefined
          ? "Nothing to undo."
          : `Undo ${undoLabel}`;
      redoButton.title = busy
        ? "Wait for the current skeleton edit to finish."
        : redoLabel === undefined
          ? "Nothing to redo."
          : `Redo ${redoLabel}`;
      undoButton.setAttribute("aria-label", undoButton.title);
      redoButton.setAttribute("aria-label", redoButton.title);
    };

    filterInput.addEventListener("input", () => {
      filterText = filterInput.value.trim().toLowerCase();
      updateDisplay();
    });
    this.registerDisposer(
      nodeFilterTypeModel.changed.add(() => {
        nodeFilterType = nodeFilterTypeModel.value;
        updateDisplay();
      }),
    );
    collapseButton.addEventListener("click", () => {
      listCollapsed = !listCollapsed;
      updateCollapseButton();
      updateDisplay();
    });

    this.registerDisposer(
      observeWatchable(() => updateGateStatus(), layer.spatialSkeletonEditMode),
    );
    this.registerDisposer(
      observeWatchable(
        () => updateGateStatus(),
        layer.spatialSkeletonMergeMode,
      ),
    );
    this.registerDisposer(
      observeWatchable(
        () => updateGateStatus(),
        layer.spatialSkeletonSplitMode,
      ),
    );
    this.registerDisposer(
      layer.spatialSkeletonVisibleChunksAvailable.changed.add(() => {
        updateGateStatus();
      }),
    );
    this.registerDisposer(
      layer.spatialSkeletonVisibleChunksNeeded.changed.add(() => {
        updateGateStatus();
      }),
    );
    this.registerDisposer(
      layer.layersChanged.add(() => {
        updateGateStatus();
      }),
    );
    this.registerDisposer(
      layer.spatialSkeletonState.commandHistory.canUndo.changed.add(() => {
        updateHistoryButtons();
      }),
    );
    this.registerDisposer(
      layer.spatialSkeletonState.commandHistory.canRedo.changed.add(() => {
        updateHistoryButtons();
      }),
    );
    this.registerDisposer(
      layer.spatialSkeletonState.commandHistory.isBusy.changed.add(() => {
        updateHistoryButtons();
      }),
    );
    this.registerDisposer(
      layer.spatialSkeletonState.commandHistory.undoLabel.changed.add(() => {
        updateHistoryButtons();
      }),
    );
    this.registerDisposer(
      layer.spatialSkeletonState.commandHistory.redoLabel.changed.add(() => {
        updateHistoryButtons();
      }),
    );
    this.registerDisposer(
      registerNested((context, segmentationGroupState) => {
        context.registerDisposer(
          segmentationGroupState.visibleSegments.changed.add(() => {
            refreshNodes();
          }),
        );
        context.registerDisposer(
          segmentationGroupState.temporaryVisibleSegments.changed.add(() => {
            refreshNodes();
          }),
        );
        context.registerDisposer(
          segmentationGroupState.useTemporaryVisibleSegments.changed.add(() => {
            refreshNodes();
          }),
        );
      }, layer.displayState.segmentationGroupState),
    );
    this.registerDisposer(
      layer.selectedSpatialSkeletonNodeId.changed.add(() => {
        pendingScrollToSelectedNode = true;
        applyRowInteractionState({ scrollSelectedIntoView: true });
      }),
    );
    this.registerDisposer(
      mouseState.changed.add(() => {
        updateHoveredViewerNode();
      }),
    );
    this.registerDisposer(
      layer.layersChanged.add(() => {
        refreshNodes();
      }),
    );
    this.registerDisposer(
      layer.manager.chunkManager.layerChunkStatisticsUpdated.add(() => {
        updateGateStatus();
      }),
    );
    this.registerDisposer(
      layer.spatialSkeletonNodeDataVersion.changed.add(() => {
        refreshNodes();
      }),
    );
    updateCollapseButton();
    updateGateStatus();
    updateHistoryButtons();
    updateHoveredViewerNode();
    refreshNodes();
  }
}
