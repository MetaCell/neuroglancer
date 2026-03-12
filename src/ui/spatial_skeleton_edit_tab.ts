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
  CATMAID_TRUE_END_LABEL,
  CatmaidClient,
} from "#src/datasource/catmaid/api.js";
import type { SegmentationUserLayer } from "#src/layer/segmentation/index.js";
import { getVisibleSegments } from "#src/segmentation_display_state/base.js";
import type {
  SpatiallyIndexedSkeletonBranchNavigationTarget,
  SpatiallyIndexedSkeletonNavigationTarget,
  SpatiallyIndexedSkeletonNode,
  SpatiallyIndexedSkeletonSource as SpatiallyIndexedSkeletonApi,
} from "#src/skeleton/api.js";
import type { SpatiallyIndexedSkeletonLayer } from "#src/skeleton/frontend.js";
import type { SpatiallyIndexedSkeletonNodeInfo } from "#src/skeleton/frontend.js";
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
import svg_bin from "ikonate/icons/bin.svg?raw";
import svg_chevrons_down from "ikonate/icons/chevrons-down.svg?raw";
import svg_chevrons_up from "ikonate/icons/chevrons-up.svg?raw";
import svg_circle from "ikonate/icons/circle.svg?raw";
import svg_flag from "ikonate/icons/flag.svg?raw";
import svg_minus from "ikonate/icons/minus.svg?raw";
import svg_origin from "ikonate/icons/origin.svg?raw";
import svg_share_android from "ikonate/icons/share-android.svg?raw";

const MAX_LISTED_NODES = 300;

type SkeletonNodeType = "root" | "branchStart" | "regular" | "virtualEnd";

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
    let actionsAllowed = false;
    let pendingScrollToSelectedNode = false;
    let refreshRequestId = 0;
    let fullSkeletonCacheVersion = layer.spatialSkeletonNodeDataVersion.value;
    let loadedNodeSummarySuffix = "";
    const pendingDeleteNodes = new Set<number>();
    const pendingTrueEndNodes = new Set<number>();
    const catmaidClients = new Map<string, CatmaidClient>();
    const fullSegmentNodeCache = new Map<
      number,
      SpatiallyIndexedSkeletonNodeInfo[]
    >();
    const pendingFullSegmentNodeFetches = new Map<
      number,
      Promise<SpatiallyIndexedSkeletonNodeInfo[]>
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

    const ensureActionsAllowed = () => {
      const reason = layer.getSpatialSkeletonActionsDisabledReason();
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
      layer.selectedSpatialSkeletonNodeId.value = node.nodeId;
      if (moveView) {
        moveViewToNodePosition(node.position);
      }
      updateList();
    };

    const mapCatmaidNodeToNodeInfo = (
      node: SpatiallyIndexedSkeletonNode,
      fallbackSegmentId: number,
    ): SpatiallyIndexedSkeletonNodeInfo | undefined => {
      const nodeId = Number(node.id);
      const segmentIdValue = Number(node.skeleton_id);
      const x = Number(node.x);
      const y = Number(node.y);
      const z = Number(node.z);
      if (
        !Number.isFinite(nodeId) ||
        !Number.isFinite(segmentIdValue) ||
        !Number.isFinite(x) ||
        !Number.isFinite(y) ||
        !Number.isFinite(z)
      ) {
        return undefined;
      }
      const parentNodeId =
        node.parent_id === undefined ||
        node.parent_id === null ||
        !Number.isFinite(Number(node.parent_id))
          ? undefined
          : Math.round(Number(node.parent_id));
      return {
        nodeId: Math.round(nodeId),
        segmentId: Math.round(
          Number.isFinite(segmentIdValue) ? segmentIdValue : fallbackSegmentId,
        ),
        position: new Float32Array([x, y, z]),
        parentNodeId,
        labels: node.labels,
      };
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

    const getCatmaidClient = (skeletonLayer: SpatiallyIndexedSkeletonLayer) => {
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
      let client = catmaidClients.get(cacheKey);
      if (client === undefined) {
        client = new CatmaidClient(
          catmaidParameters.url,
          catmaidParameters.projectId,
          catmaidParameters.token,
          source.credentialsProvider as any,
        );
        catmaidClients.set(cacheKey, client);
      }
      return client;
    };

    const getSkeletonSourceApi = (
      skeletonLayer: SpatiallyIndexedSkeletonLayer,
    ): SpatiallyIndexedSkeletonApi | undefined => {
      return getCatmaidClient(skeletonLayer);
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
      layer.selectedSpatialSkeletonNodeId.value = target.nodeId;
      moveViewToNodePosition([target.x, target.y, target.z]);
      updateList();
    };

    const getSelectedNavigationContext = () => {
      if (!ensureActionsAllowed()) return undefined;
      const selectedNode = getSelectedNode();
      if (selectedNode === undefined) {
        StatusMessage.showTemporaryMessage("No skeleton node is selected.");
        return undefined;
      }
      const skeletonLayer = layer.getSpatiallyIndexedSkeletonLayer();
      if (skeletonLayer === undefined) {
        StatusMessage.showTemporaryMessage(
          "No active spatial skeleton layer found for navigation.",
        );
        return undefined;
      }
      const skeletonApi = getSkeletonSourceApi(skeletonLayer);
      if (skeletonApi === undefined) {
        StatusMessage.showTemporaryMessage(
          "Unable to resolve a skeleton API client for the active source.",
        );
        return undefined;
      }
      return { selectedNode, skeletonApi };
    };

    const getCurrentBranchEndTarget = (
      branches: readonly SpatiallyIndexedSkeletonBranchNavigationTarget[],
    ): SpatiallyIndexedSkeletonNavigationTarget | undefined => {
      return branches[0]?.branchEnd;
    };

    const updateTrueEndLabel = (
      node: SpatiallyIndexedSkeletonNodeInfo,
      present: boolean,
    ) => {
      if (!ensureActionsAllowed()) return;
      if (pendingTrueEndNodes.has(node.nodeId)) return;
      const skeletonLayer = layer.getSpatiallyIndexedSkeletonLayer();
      if (skeletonLayer === undefined) {
        StatusMessage.showTemporaryMessage(
          "No active spatial skeleton layer found for label update.",
        );
        return;
      }
      const client = getCatmaidClient(skeletonLayer);
      if (client === undefined) {
        StatusMessage.showTemporaryMessage(
          "Unable to resolve CATMAID client for the active source.",
        );
        return;
      }
      pendingTrueEndNodes.add(node.nodeId);
      updateList();
      void (async () => {
        try {
          if (present) {
            await client.addNodeLabel(node.nodeId, CATMAID_TRUE_END_LABEL);
            StatusMessage.showTemporaryMessage(
              `Set node ${node.nodeId} as true end.`,
            );
          } else {
            await client.removeNodeLabel(node.nodeId, CATMAID_TRUE_END_LABEL);
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
      if (!ensureActionsAllowed()) return;
      if (pendingDeleteNodes.has(node.nodeId)) return;
      const skeletonLayer = layer.getSpatiallyIndexedSkeletonLayer();
      if (skeletonLayer === undefined) {
        StatusMessage.showTemporaryMessage(
          "No active spatial skeleton layer found for delete action.",
        );
        return;
      }
      const client = getCatmaidClient(skeletonLayer);
      if (client === undefined) {
        StatusMessage.showTemporaryMessage(
          "Unable to resolve CATMAID client for the active source.",
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
          await client.deleteNode(node.nodeId, {
            parentNodeId: node.parentNodeId,
            childNodeIds: directChildNodeIds,
          });
          if (deletingIsolatedRoot) {
            const segmentationGroupState =
              layer.displayState.segmentationGroupState.value;
            const { visibleSegments, selectedSegments } =
              segmentationGroupState;
            visibleSegments.delete(BigInt(node.segmentId));
            selectedSegments.delete(BigInt(node.segmentId));
          }
          if (layer.selectedSpatialSkeletonNodeId.value === node.nodeId) {
            layer.selectedSpatialSkeletonNodeId.value = undefined;
          }
          if (layer.spatialSkeletonTreeEndNodeId.value === node.nodeId) {
            layer.spatialSkeletonTreeEndNodeId.value = undefined;
          }
          layer.markSpatialSkeletonNodeDataChanged();
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

    const goRootButton = makeNavIconButton(
      svg_chevrons_up,
      "go to root of skeleton",
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
    const goBranchStartButton = makeNavIconButton(
      svg_origin,
      "go to start of current branch",
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
    const goUnfinishedBranchButton = makeNavIconButton(
      svg_flag,
      "go to closest unfinished branch",
      () => {
        goToClosestUnfinishedBranch();
      },
    );
    const goTreeEndButton = makeNavIconButton(
      svg_chevrons_down,
      "go to end of current branch",
      () => {
        const context = getSelectedNavigationContext();
        if (context === undefined) return;
        const { selectedNode, skeletonApi } = context;
        void (async () => {
          try {
            const branches = await skeletonApi.getNextBranchOrEnd(
              selectedNode.nodeId,
            );
            const target = getCurrentBranchEndTarget(branches);
            if (target === undefined) {
              StatusMessage.showTemporaryMessage(
                "No branch end was found from the selected node.",
              );
              return;
            }
            navigateToNodeTarget(target);
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
    toolbox.appendChild(navTools);
    element.insertBefore(toolbox, nodesSection);

    const gatedControls = [
      filterInput,
      goRootButton,
      goBranchStartButton,
      goUnfinishedBranchButton,
      goTreeEndButton,
    ];

    const buildSegmentTreeRows = (
      segmentNodes: SpatiallyIndexedSkeletonNodeInfo[],
    ) => {
      const nodeById = new Map<number, SpatiallyIndexedSkeletonNodeInfo>();
      for (const node of segmentNodes) {
        nodeById.set(node.nodeId, node);
      }
      const childrenByParent = new Map<number, number[]>();
      for (const node of segmentNodes) {
        const parentNodeId = node.parentNodeId;
        if (parentNodeId === undefined || !nodeById.has(parentNodeId)) {
          continue;
        }
        let children = childrenByParent.get(parentNodeId);
        if (children === undefined) {
          children = [];
          childrenByParent.set(parentNodeId, children);
        }
        children.push(node.nodeId);
      }
      for (const children of childrenByParent.values()) {
        children.sort((a, b) => a - b);
      }

      const roots: number[] = [];
      for (const node of segmentNodes) {
        const parentNodeId = node.parentNodeId;
        if (parentNodeId === undefined || !nodeById.has(parentNodeId)) {
          roots.push(node.nodeId);
        }
      }
      roots.sort((a, b) => a - b);
      if (roots.length === 0 && segmentNodes.length > 0) {
        roots.push(segmentNodes[0].nodeId);
      }

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

      for (const rootNodeId of roots) {
        walk(rootNodeId, 0);
      }
      for (const node of segmentNodes) {
        if (!visited.has(node.nodeId)) {
          walk(node.nodeId, 0);
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

        const rows = buildSegmentTreeRows(segmentNodes);
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
          selectButton.disabled = !actionsAllowed;
          selectButton.dataset.selected = String(
            node.nodeId === layer.selectedSpatialSkeletonNodeId.value,
          );
          selectButton.dataset.nodeType = type;
          selectButton.style.paddingLeft = `${0.4 + depth * 1.0}em`;
          selectButton.addEventListener("click", () => {
            if (!ensureActionsAllowed()) return;
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
                !actionsAllowed || trueEndActionPending,
              ),
            );
          }
          actions.appendChild(
            makeRowActionButton(
              svg_bin,
              deleteActionTitle,
              () => deleteNode(node),
              !actionsAllowed || pendingDeleteNodes.has(node.nodeId),
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
        layer.selectedSpatialSkeletonNodeId.value =
          allNodes.length > 0 ? allNodes[0].nodeId : undefined;
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
      for (const [segmentId, segmentNodes] of fullSegmentNodeCache) {
        let cacheChanged = false;
        const nextSegmentNodes = segmentNodes.map((candidate) => {
          if (candidate.nodeId !== nodeId) return candidate;
          const updatedNode = updateNode(candidate);
          cacheChanged ||= updatedNode !== candidate;
          return updatedNode;
        });
        if (cacheChanged) {
          fullSegmentNodeCache.set(segmentId, nextSegmentNodes);
        }
      }
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

    const fetchFullSegmentNodes = async (
      client: CatmaidClient,
      segmentId: number,
    ): Promise<SpatiallyIndexedSkeletonNodeInfo[]> => {
      const cached = fullSegmentNodeCache.get(segmentId);
      if (cached !== undefined) {
        return cached;
      }
      const pending = pendingFullSegmentNodeFetches.get(segmentId);
      if (pending !== undefined) {
        return pending;
      }
      let fetchPromise: Promise<SpatiallyIndexedSkeletonNodeInfo[]>;
      const fetchVersion = fullSkeletonCacheVersion;
      fetchPromise = (async () => {
        const fetchedNodes = await client.getSkeleton(segmentId);
        const dedupedNodes = new Map<
          number,
          SpatiallyIndexedSkeletonNodeInfo
        >();
        for (const fetchedNode of fetchedNodes) {
          const mappedNode = mapCatmaidNodeToNodeInfo(fetchedNode, segmentId);
          if (mappedNode === undefined) continue;
          if (!dedupedNodes.has(mappedNode.nodeId)) {
            dedupedNodes.set(mappedNode.nodeId, mappedNode);
          }
        }
        const normalizedNodes = [...dedupedNodes.values()].sort(
          (a, b) => a.nodeId - b.nodeId,
        );
        if (fullSkeletonCacheVersion === fetchVersion) {
          fullSegmentNodeCache.set(segmentId, normalizedNodes);
        }
        return normalizedNodes;
      })().finally(() => {
        if (pendingFullSegmentNodeFetches.get(segmentId) === fetchPromise) {
          pendingFullSegmentNodeFetches.delete(segmentId);
        }
      });
      pendingFullSegmentNodeFetches.set(segmentId, fetchPromise);
      return fetchPromise;
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
        layer.selectedSpatialSkeletonNodeId.value = undefined;
        nodesSummary.textContent =
          "Set one or more segments active in Seg tab to inspect skeleton nodes.";
        updateList();
        return;
      }

      if (
        fullSkeletonCacheVersion !== layer.spatialSkeletonNodeDataVersion.value
      ) {
        fullSkeletonCacheVersion = layer.spatialSkeletonNodeDataVersion.value;
        fullSegmentNodeCache.clear();
        pendingFullSegmentNodeFetches.clear();
      }
      const activeSegmentIdSet = new Set(activeSegmentIds);
      for (const cachedSegmentId of fullSegmentNodeCache.keys()) {
        if (!activeSegmentIdSet.has(cachedSegmentId)) {
          fullSegmentNodeCache.delete(cachedSegmentId);
        }
      }

      const catmaidClient = getCatmaidClient(skeletonLayer);
      if (catmaidClient === undefined) {
        allNodes = [];
        nodesBySegment = new Map();
        layer.selectedSpatialSkeletonNodeId.value = undefined;
        nodesSummary.textContent =
          "Unable to load full skeleton data: CATMAID client unavailable for the active source.";
        updateList();
        return;
      }

      void (async () => {
        try {
          const fetchedSegments = await Promise.all(
            activeSegmentIds.map(
              async (segmentId) =>
                [
                  segmentId,
                  await fetchFullSegmentNodes(catmaidClient, segmentId),
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
            " Using full CATMAID skeleton data.",
          );
        } catch (error) {
          if (requestId !== refreshRequestId) {
            return;
          }
          const message =
            error instanceof Error ? error.message : String(error);
          StatusMessage.showTemporaryMessage(
            `Failed to load full skeleton data from CATMAID: ${message}`,
          );
          allNodes = [];
          nodesBySegment = new Map();
          layer.selectedSpatialSkeletonNodeId.value = undefined;
          nodesSummary.textContent =
            "Failed to load full skeleton data from CATMAID.";
          updateList();
        }
      })();
    };

    const updateGateStatus = () => {
      const reason = layer.getSpatialSkeletonActionsDisabledReason();
      actionsAllowed = reason === undefined;
      for (const control of gatedControls) {
        control.disabled = !actionsAllowed;
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
