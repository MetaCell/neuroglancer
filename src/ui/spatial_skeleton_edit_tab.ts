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
import { getVisibleSegments } from "#src/segmentation_display_state/base.js";
import type { SpatiallyIndexedSkeletonNodeInfo } from "#src/skeleton/frontend.js";
import { StatusMessage } from "#src/status.js";
import { observeWatchable, registerNested } from "#src/trackable_value.js";
import { makeToolButton } from "#src/ui/tool.js";
import {
  SPATIAL_SKELETON_EDIT_MODE_TOOL_ID,
  SPATIAL_SKELETON_MERGE_MODE_TOOL_ID,
  SPATIAL_SKELETON_SPLIT_MODE_TOOL_ID,
} from "#src/ui/spatial_skeleton_edit_tool.js";
import { Tab } from "#src/widget/tab_view.js";

const MAX_LISTED_NODES = 300;

function formatNodePosition(position: ArrayLike<number>) {
  const x = Number(position[0]);
  const y = Number(position[1]);
  const z = Number(position[2]);
  return `x ${Math.round(x)} y ${Math.round(y)} z ${Math.round(z)}`;
}

export class SpatialSkeletonEditTab extends Tab {
  constructor(public layer: SegmentationUserLayer) {
    super();
    const { element } = this;
    element.classList.add("neuroglancer-spatial-skeleton-tab");

    const toolbox = document.createElement("div");
    toolbox.className = "neuroglancer-segmentation-toolbox";
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
        title: "Toggle skeleton merge mode (placeholder)",
      }),
    );
    toolbox.appendChild(
      makeToolButton(this, layer.toolBinder, {
        toolJson: SPATIAL_SKELETON_SPLIT_MODE_TOOL_ID,
        label: "Split",
        title: "Toggle skeleton split mode (placeholder)",
      }),
    );
    element.appendChild(toolbox);

    const status = document.createElement("div");
    status.classList.add("neuroglancer-spatial-skeleton-tab-status");
    const gateStatus = document.createElement("div");
    const modeStatus = document.createElement("div");
    status.appendChild(gateStatus);
    status.appendChild(modeStatus);
    element.appendChild(status);

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
    nodesList.className = "neuroglancer-spatial-skeleton-nodes";
    nodesSection.appendChild(nodesTitle);
    nodesSection.appendChild(filterInput);
    nodesSection.appendChild(nodesSummary);
    nodesSection.appendChild(nodesList);
    element.appendChild(nodesSection);

    const selectionSection = document.createElement("div");
    selectionSection.className = "neuroglancer-spatial-skeleton-section";
    const selectionTitle = document.createElement("div");
    selectionTitle.className = "neuroglancer-spatial-skeleton-section-title";
    selectionTitle.textContent = "Selection";
    const selectedNodeInfo = document.createElement("div");
    selectedNodeInfo.className = "neuroglancer-spatial-skeleton-summary";
    const selectionActions = document.createElement("div");
    selectionActions.className = "neuroglancer-spatial-skeleton-actions";
    const setTreeEndButton = document.createElement("button");
    setTreeEndButton.textContent = "Set tree end";
    const prevBranchButton = document.createElement("button");
    prevBranchButton.textContent = "Prev branch";
    const nextBranchButton = document.createElement("button");
    nextBranchButton.textContent = "Next branch";
    const goRootButton = document.createElement("button");
    goRootButton.textContent = "Go root";
    const goTreeEndButton = document.createElement("button");
    goTreeEndButton.textContent = "Go tree end";
    for (const button of [
      setTreeEndButton,
      prevBranchButton,
      nextBranchButton,
      goRootButton,
      goTreeEndButton,
    ]) {
      selectionActions.appendChild(button);
    }
    const descriptionLabel = document.createElement("div");
    descriptionLabel.className = "neuroglancer-spatial-skeleton-section-title";
    descriptionLabel.textContent = "Description";
    const descriptionTextarea = document.createElement("textarea");
    descriptionTextarea.className = "neuroglancer-spatial-skeleton-description";
    descriptionTextarea.placeholder = "Describe selected node";
    const saveDescriptionButton = document.createElement("button");
    saveDescriptionButton.textContent = "Save description";
    selectionSection.appendChild(selectionTitle);
    selectionSection.appendChild(selectedNodeInfo);
    selectionSection.appendChild(selectionActions);
    selectionSection.appendChild(descriptionLabel);
    selectionSection.appendChild(descriptionTextarea);
    selectionSection.appendChild(saveDescriptionButton);
    element.appendChild(selectionSection);

    const gatedControls = [
      filterInput,
      setTreeEndButton,
      prevBranchButton,
      nextBranchButton,
      goRootButton,
      goTreeEndButton,
      descriptionTextarea,
      saveDescriptionButton,
    ];

    let allNodes: SpatiallyIndexedSkeletonNodeInfo[] = [];
    let filteredNodes: SpatiallyIndexedSkeletonNodeInfo[] = [];
    let filterText = "";
    let descriptionNodeId: number | undefined;

    const getSelectedNode = () => {
      const selectedId = layer.selectedSpatialSkeletonNodeId.value;
      if (selectedId === undefined) return undefined;
      return allNodes.find((node) => node.nodeId === selectedId);
    };

    const ensureActionsAllowed = () => {
      const reason = layer.getSpatialSkeletonActionsDisabledReason();
      if (reason !== undefined) {
        StatusMessage.showTemporaryMessage(reason);
        return false;
      }
      return true;
    };

    const setSelectedNodeByIndex = (index: number) => {
      if (index < 0 || index >= allNodes.length) return;
      layer.selectedSpatialSkeletonNodeId.value = allNodes[index].nodeId;
    };

    const moveSelectedNode = (delta: number) => {
      if (!ensureActionsAllowed()) return;
      const selectedId = layer.selectedSpatialSkeletonNodeId.value;
      if (selectedId === undefined) return;
      const index = allNodes.findIndex((node) => node.nodeId === selectedId);
      if (index === -1) return;
      setSelectedNodeByIndex(Math.max(0, Math.min(allNodes.length - 1, index + delta)));
    };

    const applyFilter = () => {
      if (filterText.length === 0) {
        filteredNodes = allNodes;
        return;
      }
      filteredNodes = allNodes.filter(
        (node) =>
          String(node.nodeId).includes(filterText) ||
          String(node.segmentId).includes(filterText),
      );
    };

    const updateList = () => {
      nodesList.textContent = "";
      const maxNodes = Math.min(MAX_LISTED_NODES, filteredNodes.length);
      for (let i = 0; i < maxNodes; ++i) {
        const node = filteredNodes[i];
        const row = document.createElement("button");
        row.className = "neuroglancer-spatial-skeleton-node-row";
        row.dataset.selected = String(
          node.nodeId === layer.selectedSpatialSkeletonNodeId.value,
        );
        row.textContent = `s${node.segmentId} n${node.nodeId}  ${formatNodePosition(node.position)}`;
        row.addEventListener("click", () => {
          if (!ensureActionsAllowed()) return;
          layer.selectedSpatialSkeletonNodeId.value = node.nodeId;
        });
        nodesList.appendChild(row);
      }
      if (filteredNodes.length > MAX_LISTED_NODES) {
        const more = document.createElement("div");
        more.className = "neuroglancer-spatial-skeleton-summary";
        more.textContent = `Showing first ${MAX_LISTED_NODES} nodes`;
        nodesList.appendChild(more);
      }
    };

    const updateSelection = () => {
      const selectedNode = getSelectedNode();
      if (selectedNode === undefined) {
        selectedNodeInfo.textContent = "No node selected.";
        descriptionNodeId = undefined;
        descriptionTextarea.value = "";
        return;
      }
      const treeEndNodeId = layer.spatialSkeletonTreeEndNodeId.value;
      const treeEndTag =
        treeEndNodeId === selectedNode.nodeId ? "  [tree end]" : "";
      selectedNodeInfo.textContent = `Node ${selectedNode.nodeId}  ${formatNodePosition(
        selectedNode.position,
      )}${treeEndTag}`;
      if (
        descriptionNodeId !== selectedNode.nodeId ||
        document.activeElement !== descriptionTextarea
      ) {
        descriptionNodeId = selectedNode.nodeId;
        descriptionTextarea.value =
          layer.getSpatialSkeletonNodeDescription(selectedNode.nodeId) ?? "";
      }
    };

    const refreshNodes = () => {
      const skeletonLayer = layer.getSpatiallyIndexedSkeletonLayer();
      const activeSegmentIds = [
        ...getVisibleSegments(layer.displayState.segmentationGroupState.value).keys(),
      ];
      if (
        skeletonLayer === undefined ||
        activeSegmentIds.length === 0
      ) {
        allNodes = [];
        filteredNodes = [];
        layer.selectedSpatialSkeletonNodeId.value = undefined;
        nodesSummary.textContent =
          "Set one or more segments active in Seg tab to inspect skeleton nodes.";
        updateList();
        updateSelection();
        return;
      }
      const nodesById = new Map<number, SpatiallyIndexedSkeletonNodeInfo>();
      for (const segmentId of activeSegmentIds) {
        const segmentNodes = skeletonLayer.getNodes({ segmentId });
        for (const node of segmentNodes) {
          if (!nodesById.has(node.nodeId)) {
            nodesById.set(node.nodeId, node);
          }
        }
      }
      allNodes = [...nodesById.values()].sort((a, b) =>
        a.segmentId === b.segmentId
          ? a.nodeId - b.nodeId
          : a.segmentId - b.segmentId,
      );
      applyFilter();
      const selectedId = layer.selectedSpatialSkeletonNodeId.value;
      if (
        selectedId === undefined ||
        !allNodes.some((node) => node.nodeId === selectedId)
      ) {
        layer.selectedSpatialSkeletonNodeId.value =
          allNodes.length > 0 ? allNodes[0].nodeId : undefined;
      }
      const segmentPreview = activeSegmentIds.slice(0, 5).map(String).join(", ");
      const segmentSuffix = activeSegmentIds.length > 5 ? ", ..." : "";
      nodesSummary.textContent =
        `${allNodes.length} loaded nodes across ${activeSegmentIds.length} active segment(s)` +
        (segmentPreview.length > 0 ? ` (${segmentPreview}${segmentSuffix})` : "") +
        ".";
      updateList();
      updateSelection();
    };

    const updateGateStatus = () => {
      const reason = layer.getSpatialSkeletonActionsDisabledReason();
      const allowed = reason === undefined;
      for (const control of gatedControls) {
        control.disabled = !allowed;
      }
      const available = layer.spatialSkeletonVisibleChunksAvailable.value;
      const needed = layer.spatialSkeletonVisibleChunksNeeded.value;
      gateStatus.textContent = allowed
        ? `Ready (${available}/${needed} visible chunks loaded).`
        : reason!;
      modeStatus.textContent =
        `Modes: edit=${layer.spatialSkeletonEditMode.value ? "on" : "off"}, ` +
        `merge=${layer.spatialSkeletonMergeMode.value ? "on" : "off"}, ` +
        `split=${layer.spatialSkeletonSplitMode.value ? "on" : "off"}.`;
    };

    filterInput.addEventListener("input", () => {
      filterText = filterInput.value.trim();
      applyFilter();
      updateList();
    });

    saveDescriptionButton.addEventListener("click", () => {
      if (!ensureActionsAllowed()) return;
      const selectedNode = getSelectedNode();
      if (selectedNode === undefined) {
        StatusMessage.showTemporaryMessage("No skeleton node is selected.");
        return;
      }
      const description = descriptionTextarea.value;
      layer.setSpatialSkeletonNodeDescription(selectedNode.nodeId, description);
      console.info("[SpatialSkeleton] Simulated CATMAID set node description", {
        nodeId: selectedNode.nodeId,
        description,
      });
      StatusMessage.showTemporaryMessage(
        `Simulated CATMAID node description update for node ${selectedNode.nodeId}.`,
      );
    });

    setTreeEndButton.addEventListener("click", () => {
      if (!ensureActionsAllowed()) return;
      const selectedNode = getSelectedNode();
      if (selectedNode === undefined) {
        StatusMessage.showTemporaryMessage("No skeleton node is selected.");
        return;
      }
      layer.spatialSkeletonTreeEndNodeId.value = selectedNode.nodeId;
      updateSelection();
    });
    prevBranchButton.addEventListener("click", () => moveSelectedNode(-1));
    nextBranchButton.addEventListener("click", () => moveSelectedNode(1));
    goRootButton.addEventListener("click", () => {
      if (!ensureActionsAllowed()) return;
      setSelectedNodeByIndex(0);
    });
    goTreeEndButton.addEventListener("click", () => {
      if (!ensureActionsAllowed()) return;
      const treeEndNodeId = layer.spatialSkeletonTreeEndNodeId.value;
      if (
        treeEndNodeId !== undefined &&
        allNodes.some((node) => node.nodeId === treeEndNodeId)
      ) {
        layer.selectedSpatialSkeletonNodeId.value = treeEndNodeId;
        return;
      }
      if (allNodes.length > 0) {
        setSelectedNodeByIndex(allNodes.length - 1);
      }
    });

    this.registerDisposer(
      observeWatchable(() => updateGateStatus(), layer.spatialSkeletonEditMode),
    );
    this.registerDisposer(
      observeWatchable(() => updateGateStatus(), layer.spatialSkeletonMergeMode),
    );
    this.registerDisposer(
      observeWatchable(() => updateGateStatus(), layer.spatialSkeletonSplitMode),
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
      layer.displayState.segmentSelectionState.changed.add(() => {
        refreshNodes();
      }),
    );
    this.registerDisposer(
      registerNested(
        (context, segmentationGroupState) => {
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
        },
        layer.displayState.segmentationGroupState,
      ),
    );
    this.registerDisposer(
      layer.selectedSpatialSkeletonNodeId.changed.add(() => {
        updateList();
        updateSelection();
      }),
    );
    this.registerDisposer(
      layer.spatialSkeletonTreeEndNodeId.changed.add(() => {
        updateSelection();
      }),
    );
    this.registerDisposer(
      layer.layersChanged.add(() => {
        refreshNodes();
      }),
    );
    this.registerDisposer(
      layer.manager.chunkManager.layerChunkStatisticsUpdated.add(() => {
        refreshNodes();
        updateGateStatus();
      }),
    );
    this.registerDisposer(
      layer.displayState.spatialSkeletonGridLevel2d.changed.add(() => {
        refreshNodes();
      }),
    );
    this.registerDisposer(
      layer.displayState.spatialSkeletonGridLevel3d.changed.add(() => {
        refreshNodes();
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
