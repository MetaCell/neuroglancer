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

import { CATMAID_TRUE_END_LABEL } from "#src/datasource/catmaid/api.js";
import type { SegmentationUserLayer } from "#src/layer/segmentation/index.js";
import {
  getVisibleSegments,
  removeSegmentFromVisibleSets,
} from "#src/segmentation_display_state/base.js";
import type {
  SpatiallyIndexedSkeletonBranchNavigationTarget,
  SpatiallyIndexedSkeletonNavigationTarget,
  SpatiallyIndexedSkeletonOpenLeaf,
} from "#src/skeleton/api.js";
import type { SpatiallyIndexedSkeletonNodeInfo } from "#src/skeleton/frontend.js";
import { getEditableSpatiallyIndexedSkeletonSource } from "#src/skeleton/state.js";
import {
  buildSpatiallyIndexedSkeletonNavigationGraph,
  getCurrentBranchContext,
  getNextBranchOrEnd as getNextBranchOrEndFromGraph,
  getOpenLeaves as getOpenLeavesFromGraph,
  getPreviousBranchOrRoot as getPreviousBranchOrRootFromGraph,
  getSkeletonRootNode as getSkeletonRootNodeFromGraph,
  type SpatiallyIndexedSkeletonNavigationGraph,
} from "#src/skeleton/navigation.js";
import { StatusMessage } from "#src/status.js";
import { observeWatchable, registerNested } from "#src/trackable_value.js";
import {
  SPATIAL_SKELETON_EDIT_MODE_TOOL_ID,
  SPATIAL_SKELETON_MERGE_MODE_TOOL_ID,
  SPATIAL_SKELETON_SPLIT_MODE_TOOL_ID,
} from "#src/ui/spatial_skeleton_edit_tool.js";
import { makeToolButton } from "#src/ui/tool.js";
import { makeIcon } from "#src/widget/icon.js";
import { Tab } from "#src/widget/tab_view.js";
import svg_arrow_right from "ikonate/icons/arrow-right.svg?raw";
import svg_bin from "ikonate/icons/bin.svg?raw";
import svg_chevron_left from "ikonate/icons/chevron-left.svg?raw";
import svg_chevron_right from "ikonate/icons/chevron-right.svg?raw";
import svg_chevrons_left from "ikonate/icons/chevrons-left.svg?raw";
import svg_chevrons_right from "ikonate/icons/chevrons-right.svg?raw";
import svg_circle from "ikonate/icons/circle.svg?raw";
import svg_flag from "ikonate/icons/flag.svg?raw";
import svg_minus from "ikonate/icons/minus.svg?raw";
import svg_origin from "ikonate/icons/origin.svg?raw";
import svg_share_android from "ikonate/icons/share-android.svg?raw";

const MAX_LISTED_NODES = 300;

type SkeletonNodeType = "root" | "branchStart" | "regular" | "virtualEnd";

interface SpatiallyIndexedSkeletonNavigationApi {
  getSkeletonRootNode(
    skeletonId: number,
  ): Promise<SpatiallyIndexedSkeletonNavigationTarget>;
  getPreviousBranchOrRoot(
    nodeId: number,
    options?: { alt?: boolean },
  ): Promise<SpatiallyIndexedSkeletonNavigationTarget>;
  getNextBranchOrEnd(
    nodeId: number,
  ): Promise<SpatiallyIndexedSkeletonBranchNavigationTarget[]>;
  getOpenLeaves(
    skeletonId: number,
    nodeId: number,
  ): Promise<SpatiallyIndexedSkeletonOpenLeaf[]>;
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

function hasTrueEndLabel(node: SpatiallyIndexedSkeletonNodeInfo) {
  return (
    node.labels?.some(
      (label) => label.trim().toLowerCase() === CATMAID_TRUE_END_LABEL,
    ) ?? false
  );
}

function formatNodePosition(position: ArrayLike<number>) {
  const x = Number(position[0]);
  const y = Number(position[1]);
  const z = Number(position[2]);
  return `x ${Math.round(x)} y ${Math.round(y)} z ${Math.round(z)}`;
}

function classifyNodeType(
  node: SpatiallyIndexedSkeletonNodeInfo,
  childCount: number,
  parentInTree: boolean,
): SkeletonNodeType {
  if (!parentInTree || node.parentNodeId === undefined) {
    return "root";
  }
  if (childCount > 1) {
    return "branchStart";
  }
  if (childCount === 1) {
    return "regular";
  }
  return "virtualEnd";
}

function nodeMatchesFilter(
  node: SpatiallyIndexedSkeletonNodeInfo,
  filterText: string,
) {
  if (filterText.length === 0) return true;
  if (String(node.nodeId).includes(filterText)) return true;
  if (String(node.segmentId).includes(filterText)) return true;
  if (
    node.labels?.some((label) => label.toLowerCase().includes(filterText)) ??
    false
  ) {
    return true;
  }
  return formatNodePosition(node.position).toLowerCase().includes(filterText);
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

    const navTools = document.createElement("div");
    navTools.className = "neuroglancer-spatial-skeleton-nav-tools";
    const makeNavIconButton = (
      svg: string,
      title: string,
      onClick: () => void,
    ) => {
      const button = document.createElement("button");
      button.className = "neuroglancer-spatial-skeleton-nav-tool";
      button.type = "button";
      button.title = title;
      button.appendChild(makeIcon({ svg, title, clickable: false }));
      button.addEventListener("click", () => onClick());
      navTools.appendChild(button);
      return button;
    };

    const nodesSection = document.createElement("div");
    nodesSection.className = "neuroglancer-spatial-skeleton-section";
    const nodesTitle = document.createElement("div");
    nodesTitle.className = "neuroglancer-spatial-skeleton-section-title";
    nodesTitle.textContent = "Active Skeleton Nodes";
    const filterInput = document.createElement("input");
    filterInput.type = "text";
    filterInput.placeholder = "Search node or skeleton id";
    filterInput.className = "neuroglancer-spatial-skeleton-filter";
    const nodesSummary = document.createElement("div");
    nodesSummary.className = "neuroglancer-spatial-skeleton-summary";
    const nodesList = document.createElement("div");
    nodesList.className = "neuroglancer-spatial-skeleton-tree";
    nodesSection.appendChild(nodesTitle);
    nodesSection.appendChild(filterInput);
    nodesSection.appendChild(nodesSummary);
    nodesSection.appendChild(nodesList);
    element.appendChild(nodesSection);

    let allNodes: SpatiallyIndexedSkeletonNodeInfo[] = [];
    let activeSegmentIds: number[] = [];
    let nodesBySegment = new Map<number, SpatiallyIndexedSkeletonNodeInfo[]>();
    let filterText = "";
    let inspectionAllowed = false;
    let navigationAllowed = false;
    let labelEditingAllowed = false;
    let nodeDeletionAllowed = false;
    let pendingScrollToSelectedNode = false;
    let refreshRequestId = 0;
    let loadedNodeSummarySuffix = "";
    const pendingDeleteNodes = new Set<number>();
    const pendingTrueEndNodes = new Set<number>();
    const skeletonState = layer.spatialSkeletonState;
    const navigationGraphCache = new Map<
      number,
      {
        nodes: readonly SpatiallyIndexedSkeletonNodeInfo[];
        graph: SpatiallyIndexedSkeletonNavigationGraph;
      }
    >();

    const getSelectedNode = () => {
      const selectedId = layer.selectedSpatialSkeletonNodeId.value;
      if (selectedId === undefined) return undefined;
      return allNodes.find((node) => node.nodeId === selectedId);
    };

    const updateTrueEndLabels = (
      labels: readonly string[] | undefined,
      present: boolean,
    ) => {
      const nextLabels = (labels ?? []).filter(
        (label) => label.trim().toLowerCase() !== CATMAID_TRUE_END_LABEL,
      );
      if (present) {
        nextLabels.push(CATMAID_TRUE_END_LABEL);
      }
      return nextLabels.length > 0 ? nextLabels : undefined;
    };

    const labelsEqual = (
      a: readonly string[] | undefined,
      b: readonly string[] | undefined,
    ) => {
      if (a === b) return true;
      if (a === undefined || b === undefined) {
        return a === undefined && b === undefined;
      }
      if (a.length !== b.length) return false;
      for (let i = 0; i < a.length; i++) {
        if (a[i] !== b[i]) return false;
      }
      return true;
    };

    const ensureActionsAllowed = (
      requiredCapabilities:
        | "inspectSkeletons"
        | "editNodeLabels"
        | "deleteNodes"
        | readonly ("inspectSkeletons" | "editNodeLabels" | "deleteNodes")[],
      options: {
        requireMaxLod?: boolean;
        requireVisibleChunks?: boolean;
      } = {},
    ) => {
      const reason = layer.getSpatialSkeletonActionsDisabledReason(
        requiredCapabilities,
        options,
      );
      if (reason !== undefined) {
        StatusMessage.showTemporaryMessage(reason);
        return false;
      }
      return true;
    };

    const selectNode = (
      node: SpatiallyIndexedSkeletonNodeInfo | undefined,
      options: {
        moveView?: boolean;
      } = {},
    ) => {
      if (node === undefined) return;
      const moveView = options.moveView ?? true;
      pendingScrollToSelectedNode = true;
      layer.selectSpatialSkeletonNode(node.nodeId, false, {
        segmentId: node.segmentId,
      });
      if (moveView) {
        moveViewToNodePosition(node.position);
      }
      updateList();
    };

    const moveViewToNodePosition = (position: ArrayLike<number>) => {
      const globalPosition = layer.manager.root.globalPosition;
      const nextGlobal = globalPosition.value.slice();
      const globalRank = Math.min(nextGlobal.length, 3);
      for (let i = 0; i < globalRank; ++i) {
        const value = Number(position[i]);
        if (Number.isFinite(value)) {
          nextGlobal[i] = value;
        }
      }
      globalPosition.value = nextGlobal;

      const localPosition = layer.localPosition;
      const nextLocal = localPosition.value.slice();
      const localRank = Math.min(nextLocal.length, 3);
      for (let i = 0; i < localRank; ++i) {
        const value = Number(position[i]);
        if (Number.isFinite(value)) {
          nextLocal[i] = value;
        }
      }
      localPosition.value = nextLocal;
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

    const skeletonNavigationApi: SpatiallyIndexedSkeletonNavigationApi = {
      async getSkeletonRootNode(skeletonId: number) {
        return getSkeletonRootNodeFromGraph(
          getSegmentNavigationGraph(skeletonId),
        );
      },
      async getPreviousBranchOrRoot(
        nodeId: number,
        options: { alt?: boolean } = {},
      ) {
        const node = getNavigationNode(nodeId);
        if (node === undefined) {
          throw new Error(
            `Node ${nodeId} is not available in the loaded skeleton cache.`,
          );
        }
        return getPreviousBranchOrRootFromGraph(
          getSegmentNavigationGraph(node.segmentId),
          nodeId,
          options,
        );
      },
      async getNextBranchOrEnd(nodeId: number) {
        const node = getNavigationNode(nodeId);
        if (node === undefined) {
          throw new Error(
            `Node ${nodeId} is not available in the loaded skeleton cache.`,
          );
        }
        return getNextBranchOrEndFromGraph(
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
        selectNode(existingNode);
        return;
      }
      pendingScrollToSelectedNode = true;
      layer.selectSpatialSkeletonNode(target.nodeId, false);
      moveViewToNodePosition([target.x, target.y, target.z]);
      updateList();
    };

    const getSelectedNavigationContext = () => {
      if (
        !ensureActionsAllowed("inspectSkeletons", {
          requireMaxLod: false,
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
        return {
          selectedNode,
          skeletonApi: skeletonNavigationApi,
          navigationGraph: getSegmentNavigationGraph(selectedNode.segmentId),
        };
      } catch (error) {
        const message = error instanceof Error ? error.message : String(error);
        StatusMessage.showTemporaryMessage(
          `Unable to resolve the local skeleton graph for navigation: ${message}`,
        );
        return undefined;
      }
    };

    const getSelectedBranchNavigationContext = () => {
      const context = getSelectedNavigationContext();
      if (context === undefined) return undefined;
      const branchContext = getCurrentBranchContext(
        context.navigationGraph,
        context.selectedNode.nodeId,
        {
          anchorNodeId: layer.spatialSkeletonTreeEndNodeId.value,
        },
      );
      return {
        ...context,
        branchContext,
      };
    };

    const navigateToBranchTarget = (
      target: SpatiallyIndexedSkeletonNavigationTarget,
      options: {
        branchEndNodeId?: number;
      } = {},
    ) => {
      layer.spatialSkeletonTreeEndNodeId.value = options.branchEndNodeId;
      navigateToNodeTarget(target);
    };

    const updateTrueEndLabel = (
      node: SpatiallyIndexedSkeletonNodeInfo,
      present: boolean,
    ) => {
      if (!ensureActionsAllowed("editNodeLabels")) return;
      if (pendingTrueEndNodes.has(node.nodeId)) return;
      const skeletonLayer = layer.getSpatiallyIndexedSkeletonLayer();
      if (skeletonLayer === undefined) {
        StatusMessage.showTemporaryMessage(
          "No active spatial skeleton layer found for label update.",
        );
        return;
      }
      const skeletonSource =
        getEditableSpatiallyIndexedSkeletonSource(skeletonLayer);
      if (skeletonSource === undefined) {
        StatusMessage.showTemporaryMessage(
          "Unable to resolve editable skeleton source for the active layer.",
        );
        return;
      }
      pendingTrueEndNodes.add(node.nodeId);
      updateList();
      void (async () => {
        try {
          if (present) {
            await skeletonSource.addNodeLabel(
              node.nodeId,
              CATMAID_TRUE_END_LABEL,
            );
            StatusMessage.showTemporaryMessage(
              `Set node ${node.nodeId} as true end.`,
            );
          } else {
            await skeletonSource.removeNodeLabel(
              node.nodeId,
              CATMAID_TRUE_END_LABEL,
            );
            StatusMessage.showTemporaryMessage(
              `Removed true end from node ${node.nodeId}.`,
            );
          }
          applyTrueEndLabelLocally(node.nodeId, present);
        } catch (error) {
          const message =
            error instanceof Error ? error.message : String(error);
          StatusMessage.showTemporaryMessage(
            `Failed to update true end label: ${message}`,
          );
        } finally {
          pendingTrueEndNodes.delete(node.nodeId);
          updateList();
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

    const getDirectChildNodeIds = (node: SpatiallyIndexedSkeletonNodeInfo) => {
      const segmentNodes = nodesBySegment.get(node.segmentId) ?? [];
      const childNodeIds: number[] = [];
      for (const candidate of segmentNodes) {
        if (candidate.parentNodeId !== node.nodeId) continue;
        childNodeIds.push(candidate.nodeId);
      }
      childNodeIds.sort((a, b) => a - b);
      return childNodeIds;
    };

    const deleteNode = (node: SpatiallyIndexedSkeletonNodeInfo) => {
      if (!ensureActionsAllowed("deleteNodes")) return;
      if (pendingDeleteNodes.has(node.nodeId)) return;
      const skeletonLayer = layer.getSpatiallyIndexedSkeletonLayer();
      if (skeletonLayer === undefined) {
        StatusMessage.showTemporaryMessage(
          "No active spatial skeleton layer found for delete action.",
        );
        return;
      }
      const skeletonSource =
        getEditableSpatiallyIndexedSkeletonSource(skeletonLayer);
      if (skeletonSource === undefined) {
        StatusMessage.showTemporaryMessage(
          "Unable to resolve editable skeleton source for the active layer.",
        );
        return;
      }
      pendingDeleteNodes.add(node.nodeId);
      updateList();
      void (async () => {
        try {
          const directChildNodeIds = getDirectChildNodeIds(node);
          const deletingIsolatedRoot =
            node.parentNodeId === undefined && directChildNodeIds.length === 0;
          await skeletonSource.deleteNode(node.nodeId, {
            parentNodeId: node.parentNodeId,
            childNodeIds: directChildNodeIds,
          });
          if (deletingIsolatedRoot) {
            const segmentationGroupState =
              layer.displayState.segmentationGroupState.value;
            removeSegmentFromVisibleSets(
              segmentationGroupState,
              BigInt(node.segmentId),
              { deselect: true },
            );
          }
          if (layer.selectedSpatialSkeletonNodeId.value === node.nodeId) {
            layer.clearSpatialSkeletonNodeSelection(false);
          }
          if (layer.spatialSkeletonTreeEndNodeId.value === node.nodeId) {
            layer.spatialSkeletonTreeEndNodeId.value = undefined;
          }
          skeletonState.removeCachedNode(node.nodeId, {
            parentNodeId: node.parentNodeId,
            childNodeIds: directChildNodeIds,
          });
          layer.markSpatialSkeletonNodeDataChanged({
            invalidateFullSkeletonCache: false,
          });
          skeletonLayer.invalidateSourceCaches();
          StatusMessage.showTemporaryMessage(`Deleted node ${node.nodeId}.`);
          refreshNodes();
        } catch (error) {
          const message =
            error instanceof Error ? error.message : String(error);
          StatusMessage.showTemporaryMessage(
            `Failed to delete node: ${message}`,
          );
          updateList();
        } finally {
          pendingDeleteNodes.delete(node.nodeId);
          updateList();
        }
      })();
    };

    const goRootButton = makeNavIconButton(svg_origin, "Go to root", () => {
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
    });
    const goBranchStartButton = makeNavIconButton(
      svg_chevrons_left,
      "Start of branch",
      () => {
        const context = getSelectedNavigationContext();
        if (context === undefined) return;
        const { selectedNode, skeletonApi } = context;
        void (async () => {
          try {
            navigateToNodeTarget(
              await skeletonApi.getPreviousBranchOrRoot(selectedNode.nodeId),
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
    const goPreviousBranchButton = makeNavIconButton(
      svg_chevron_left,
      "Previous branch",
      () => {
        const context = getSelectedBranchNavigationContext();
        if (context === undefined) return;
        const { branchContext } = context;
        const { branches, currentBranchIndex } = branchContext;
        if (branches.length <= 1 || currentBranchIndex === undefined) {
          StatusMessage.showTemporaryMessage(
            "No previous branch was found from the selected node.",
          );
          return;
        }
        if (currentBranchIndex <= 0) {
          StatusMessage.showTemporaryMessage(
            "The selected branch is already the first branch in this branch set.",
          );
          return;
        }
        const targetBranch = branches[currentBranchIndex - 1];
        navigateToBranchTarget(targetBranch.branchStartOrEnd, {
          branchEndNodeId: targetBranch.branchEnd.nodeId,
        });
      },
    );
    const goNextBranchButton = makeNavIconButton(
      svg_chevron_right,
      "Next branch",
      () => {
        const context = getSelectedBranchNavigationContext();
        if (context === undefined) return;
        const { branchContext } = context;
        const { branches, currentBranchIndex } = branchContext;
        if (branches.length <= 1 || currentBranchIndex === undefined) {
          StatusMessage.showTemporaryMessage(
            "No next branch was found from the selected node.",
          );
          return;
        }
        if (currentBranchIndex >= branches.length - 1) {
          StatusMessage.showTemporaryMessage(
            "The selected branch is already the last branch in this branch set.",
          );
          return;
        }
        const targetBranch = branches[currentBranchIndex + 1];
        navigateToBranchTarget(targetBranch.branchStartOrEnd, {
          branchEndNodeId: targetBranch.branchEnd.nodeId,
        });
      },
    );
    const goTreeEndButton = makeNavIconButton(
      svg_chevrons_right,
      "End of branch",
      () => {
        const context = getSelectedBranchNavigationContext();
        if (context === undefined) return;
        const { branchContext, selectedNode } = context;
        const targetBranch =
          branchContext.currentBranchIndex === undefined
            ? undefined
            : branchContext.branches[branchContext.currentBranchIndex];
        if (targetBranch === undefined) {
          navigateToBranchTarget(
            {
              nodeId: selectedNode.nodeId,
              x: Number(selectedNode.position[0]),
              y: Number(selectedNode.position[1]),
              z: Number(selectedNode.position[2]),
            },
            {
              branchEndNodeId: selectedNode.nodeId,
            },
          );
          return;
        }
        navigateToBranchTarget(targetBranch.branchEnd, {
          branchEndNodeId: targetBranch.branchEnd.nodeId,
        });
      },
    );
    const goUnfinishedBranchButton = makeNavIconButton(
      svg_arrow_right,
      "Nearest unfinished branch",
      () => {
        goToClosestUnfinishedBranch();
      },
    );
    toolbox.appendChild(navTools);
    element.insertBefore(toolbox, nodesSection);

    const gatedControls = [
      goRootButton,
      goBranchStartButton,
      goPreviousBranchButton,
      goNextBranchButton,
      goTreeEndButton,
      goUnfinishedBranchButton,
    ];

    const buildSegmentTreeRows = (
      segmentId: number,
      segmentNodes: SpatiallyIndexedSkeletonNodeInfo[],
    ) => {
      if (segmentNodes.length === 0) {
        return [];
      }
      const { nodeById, childrenByParent, rootNodeIds } =
        getSegmentNavigationGraph(segmentId);

      const visibleMemo = new Map<number, boolean>();
      const isNodeVisible = (nodeId: number): boolean => {
        if (filterText.length === 0) return true;
        const cached = visibleMemo.get(nodeId);
        if (cached !== undefined) {
          return cached;
        }
        const node = nodeById.get(nodeId);
        if (node === undefined) {
          visibleMemo.set(nodeId, false);
          return false;
        }
        let visible = nodeMatchesFilter(node, filterText);
        if (!visible) {
          const children = childrenByParent.get(nodeId) ?? [];
          for (const childNodeId of children) {
            if (isNodeVisible(childNodeId)) {
              visible = true;
              break;
            }
          }
        }
        visibleMemo.set(nodeId, visible);
        return visible;
      };

      const rows: Array<{
        node: SpatiallyIndexedSkeletonNodeInfo;
        depth: number;
        type: SkeletonNodeType;
        isLeaf: boolean;
      }> = [];
      const visited = new Set<number>();
      const walk = (nodeId: number, depth: number) => {
        if (visited.has(nodeId)) return;
        visited.add(nodeId);
        if (!isNodeVisible(nodeId)) return;
        const node = nodeById.get(nodeId);
        if (node === undefined) return;
        const children = childrenByParent.get(nodeId) ?? [];
        const parentInTree =
          node.parentNodeId !== undefined && nodeById.has(node.parentNodeId);
        const type = classifyNodeType(node, children.length, parentInTree);
        rows.push({ node, depth, type, isLeaf: children.length === 0 });
        const nextDepth = depth + (children.length > 1 ? 1 : 0);
        for (const childNodeId of children) {
          walk(childNodeId, nextDepth);
        }
      };

      for (const rootNodeId of rootNodeIds) {
        walk(rootNodeId, 0);
      }
      for (const nodeId of nodeById.keys()) {
        if (!visited.has(nodeId)) {
          walk(nodeId, 0);
        }
      }
      return rows;
    };

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

    const updateList = () => {
      nodesList.textContent = "";
      let renderedNodeCount = 0;
      let overflowNodeCount = 0;
      let selectedRowButton: HTMLButtonElement | undefined;
      for (const segmentId of activeSegmentIds) {
        const segmentNodes = nodesBySegment.get(segmentId) ?? [];
        const section = document.createElement("div");
        section.className = "neuroglancer-spatial-skeleton-tree-segment";
        const header = document.createElement("div");
        header.className = "neuroglancer-spatial-skeleton-tree-segment-header";
        header.textContent = `s${segmentId} (${segmentNodes.length} node${
          segmentNodes.length === 1 ? "" : "s"
        })`;
        section.appendChild(header);

        const rows = buildSegmentTreeRows(segmentId, segmentNodes);
        if (rows.length === 0) {
          const empty = document.createElement("div");
          empty.className = "neuroglancer-spatial-skeleton-summary";
          empty.textContent =
            filterText.length === 0 ? "No loaded nodes." : "No matching nodes.";
          section.appendChild(empty);
          nodesList.appendChild(section);
          continue;
        }

        for (const { node, depth, type, isLeaf } of rows) {
          if (renderedNodeCount >= MAX_LISTED_NODES) {
            overflowNodeCount++;
            continue;
          }
          renderedNodeCount++;
          const row = document.createElement("div");
          row.className = "neuroglancer-spatial-skeleton-tree-row";

          const selectButton = document.createElement("button");
          selectButton.className = "neuroglancer-spatial-skeleton-node-main";
          selectButton.type = "button";
          selectButton.disabled = !inspectionAllowed;
          selectButton.dataset.selected = String(
            node.nodeId === layer.selectedSpatialSkeletonNodeId.value,
          );
          selectButton.dataset.nodeType = type;
          selectButton.style.paddingLeft = `${0.4 + depth * 1.0}em`;
          selectButton.addEventListener("click", () => {
            if (
              !ensureActionsAllowed("inspectSkeletons", {
                requireMaxLod: false,
                requireVisibleChunks: false,
              })
            ) {
              return;
            }
            selectNode(node);
          });
          if (node.nodeId === layer.selectedSpatialSkeletonNodeId.value) {
            selectedRowButton = selectButton;
          }

          const nodeIsTrueEnd = hasTrueEndLabel(node);
          const typeIconSvg = nodeIsTrueEnd ? svg_flag : NODE_TYPE_ICONS[type];
          const typeIconTitle = nodeIsTrueEnd
            ? "true end"
            : NODE_TYPE_LABELS[type];
          const typeIcon = document.createElement("span");
          typeIcon.className = "neuroglancer-spatial-skeleton-node-type";
          typeIcon.title = typeIconTitle;
          typeIcon.appendChild(
            makeIcon({
              svg: typeIconSvg,
              title: typeIconTitle,
              clickable: false,
            }),
          );
          const text = document.createElement("span");
          text.className = "neuroglancer-spatial-skeleton-node-text";
          text.textContent = `n${node.nodeId} ${formatNodePosition(node.position)}`;
          selectButton.appendChild(typeIcon);
          selectButton.appendChild(text);
          if (type === "root") {
            const rootTag = document.createElement("span");
            rootTag.className = "neuroglancer-spatial-skeleton-node-root-tag";
            rootTag.textContent = "root";
            selectButton.appendChild(rootTag);
          }

          const actions = document.createElement("div");
          actions.className = "neuroglancer-spatial-skeleton-node-actions";
          let deleteActionTitle = "delete node";
          if (pendingDeleteNodes.has(node.nodeId)) {
            deleteActionTitle = "deleting node";
          }
          const trueEndActionPending = pendingTrueEndNodes.has(node.nodeId);
          const trueEndActionTitle = trueEndActionPending
            ? nodeIsTrueEnd
              ? "removing true end"
              : "setting true end"
            : nodeIsTrueEnd
              ? "remove true end"
              : "set as true end";
          if (isLeaf || nodeIsTrueEnd) {
            actions.appendChild(
              makeRowActionButton(
                svg_flag,
                trueEndActionTitle,
                () => updateTrueEndLabel(node, !nodeIsTrueEnd),
                !labelEditingAllowed || trueEndActionPending,
              ),
            );
          }
          actions.appendChild(
            makeRowActionButton(
              svg_bin,
              deleteActionTitle,
              () => deleteNode(node),
              !nodeDeletionAllowed || pendingDeleteNodes.has(node.nodeId),
            ),
          );

          row.appendChild(selectButton);
          row.appendChild(actions);
          section.appendChild(row);
        }

        nodesList.appendChild(section);
      }

      if (overflowNodeCount > 0) {
        const more = document.createElement("div");
        more.className = "neuroglancer-spatial-skeleton-summary";
        more.textContent = `Showing first ${MAX_LISTED_NODES} nodes`;
        nodesList.appendChild(more);
      }
      if (pendingScrollToSelectedNode) {
        pendingScrollToSelectedNode = false;
        selectedRowButton?.scrollIntoView({
          block: "nearest",
        });
      }
    };

    const summarizeNodeState = (summarySuffix = "") => {
      const segmentPreview = activeSegmentIds
        .slice(0, 5)
        .map(String)
        .join(", ");
      const segmentSuffix = activeSegmentIds.length > 5 ? ", ..." : "";
      nodesSummary.textContent =
        `${allNodes.length} loaded nodes across ${activeSegmentIds.length} active skeleton(s)` +
        (segmentPreview.length > 0
          ? ` (${segmentPreview}${segmentSuffix})`
          : "") +
        `.${summarySuffix}`;
    };

    const applyNodesBySegment = (
      nextNodesBySegment: Map<number, SpatiallyIndexedSkeletonNodeInfo[]>,
      summarySuffix = "",
    ) => {
      loadedNodeSummarySuffix = summarySuffix;
      navigationGraphCache.clear();
      nodesBySegment = nextNodesBySegment;
      const allNodesById = new Map<number, SpatiallyIndexedSkeletonNodeInfo>();
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
      summarizeNodeState(summarySuffix);
      updateList();
    };

    const applyTrueEndLabelLocally = (nodeId: number, present: boolean) => {
      const updateNode = (node: SpatiallyIndexedSkeletonNodeInfo) => {
        const nextLabels = updateTrueEndLabels(node.labels, present);
        if (labelsEqual(node.labels, nextLabels)) {
          return node;
        }
        return { ...node, labels: nextLabels };
      };
      skeletonState.updateCachedNode(nodeId, updateNode);
      let changed = false;
      const nextNodesBySegment = new Map<
        number,
        SpatiallyIndexedSkeletonNodeInfo[]
      >();
      for (const [segmentId, segmentNodes] of nodesBySegment) {
        let segmentChanged = false;
        const nextSegmentNodes = segmentNodes.map((candidate) => {
          if (candidate.nodeId !== nodeId) return candidate;
          const updatedNode = updateNode(candidate);
          segmentChanged ||= updatedNode !== candidate;
          return updatedNode;
        });
        nextNodesBySegment.set(
          segmentId,
          segmentChanged ? nextSegmentNodes : segmentNodes,
        );
        changed ||= segmentChanged;
      }
      if (changed) {
        applyNodesBySegment(nextNodesBySegment, loadedNodeSummarySuffix);
      }
    };

    const refreshNodes = () => {
      const requestId = ++refreshRequestId;
      const skeletonLayer = layer.getSpatiallyIndexedSkeletonLayer();
      const activeSegmentBigints = [
        ...getVisibleSegments(
          layer.displayState.segmentationGroupState.value,
        ).keys(),
      ];
      activeSegmentIds = activeSegmentBigints
        .map((segmentId) => Number(segmentId))
        .filter((segmentId) => Number.isFinite(segmentId))
        .sort((a, b) => a - b);
      if (skeletonLayer === undefined || activeSegmentIds.length === 0) {
        allNodes = [];
        nodesBySegment = new Map();
        layer.clearSpatialSkeletonNodeSelection(false);
        nodesSummary.textContent =
          "Set one or more segments active in Seg tab to inspect skeleton nodes.";
        updateList();
        return;
      }

      skeletonState.evictInactiveSegmentNodes(activeSegmentIds);

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
            SpatiallyIndexedSkeletonNodeInfo[]
          >(fetchedSegments);
          applyNodesBySegment(
            nextNodesBySegment,
            " Using source-backed full skeleton data.",
          );
        } catch (error) {
          if (requestId !== refreshRequestId) {
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
          nodesSummary.textContent =
            "Failed to load full skeleton data from the active source.";
          updateList();
        }
      })();
    };

    const updateGateStatus = () => {
      inspectionAllowed =
        layer.getSpatialSkeletonActionsDisabledReason("inspectSkeletons", {
          requireMaxLod: false,
          requireVisibleChunks: false,
        }) === undefined;
      navigationAllowed = inspectionAllowed;
      labelEditingAllowed =
        layer.getSpatialSkeletonActionsDisabledReason("editNodeLabels") ===
        undefined;
      nodeDeletionAllowed =
        layer.getSpatialSkeletonActionsDisabledReason("deleteNodes") ===
        undefined;
      filterInput.disabled = !inspectionAllowed;
      for (const control of gatedControls) {
        control.disabled = !navigationAllowed;
      }
      updateList();
    };

    filterInput.addEventListener("input", () => {
      filterText = filterInput.value.trim().toLowerCase();
      updateList();
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
      layer.spatialSkeletonActionsAllowed.changed.add(() => {
        updateGateStatus();
      }),
    );
    this.registerDisposer(
      layer.spatialSkeletonSourceCapabilities.changed.add(() => {
        updateGateStatus();
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
        updateList();
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

    updateGateStatus();
    refreshNodes();
  }
}
