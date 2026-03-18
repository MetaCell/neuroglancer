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
import svg_chevron_right from "ikonate/icons/chevron-right.svg?raw";
import svg_chevron_down from "ikonate/icons/chevron-down.svg?raw";
import svg_chevron_up from "ikonate/icons/chevron-up.svg?raw";
import svg_chevrons_left from "ikonate/icons/chevrons-left.svg?raw";
import svg_chevrons_right from "ikonate/icons/chevrons-right.svg?raw";
import svg_circle from "ikonate/icons/circle.svg?raw";
import svg_flag from "ikonate/icons/flag.svg?raw";
import svg_info from "ikonate/icons/info.svg?raw";
import svg_minus from "ikonate/icons/minus.svg?raw";
import svg_origin from "ikonate/icons/origin.svg?raw";
import svg_retweet from "ikonate/icons/retweet.svg?raw";
import svg_share_android from "ikonate/icons/share-android.svg?raw";
import { CATMAID_TRUE_END_LABEL } from "#src/datasource/catmaid/api.js";
import type { SegmentationUserLayer } from "#src/layer/segmentation/index.js";
import { Overlay } from "#src/overlay.js";
import {
  getVisibleSegments,
  removeSegmentFromVisibleSets,
} from "#src/segmentation_display_state/base.js";
import { getBaseObjectColor } from "#src/segmentation_display_state/frontend.js";
import type {
  SpatiallyIndexedSkeletonNavigationTarget,
  SpatiallyIndexedSkeletonOpenLeaf,
} from "#src/skeleton/api.js";
import type { SpatiallyIndexedSkeletonNodeInfo } from "#src/skeleton/frontend.js";
import {
  buildSpatiallyIndexedSkeletonNavigationGraph,
  getBranchEnd as getBranchEndFromGraph,
  getBranchStart as getBranchStartFromGraph,
  getChildNode as getChildNodeFromGraph,
  getFlatListNodeIds,
  getNextCollapsedLevelNode as getNextCollapsedLevelNodeFromGraph,
  getOpenLeaves as getOpenLeavesFromGraph,
  getParentNode as getParentNodeFromGraph,
  getSkeletonRootNode as getSkeletonRootNodeFromGraph,
  type SpatiallyIndexedSkeletonNavigationGraph,
} from "#src/skeleton/navigation.js";
import { getEditableSpatiallyIndexedSkeletonSource } from "#src/skeleton/state.js";
import { StatusMessage } from "#src/status.js";
import { observeWatchable, registerNested } from "#src/trackable_value.js";
import {
  SPATIAL_SKELETON_EDIT_MODE_TOOL_ID,
  SPATIAL_SKELETON_MERGE_MODE_TOOL_ID,
  SPATIAL_SKELETON_SPLIT_MODE_TOOL_ID,
} from "#src/ui/spatial_skeleton_edit_tool.js";
import { makeToolButton } from "#src/ui/tool.js";
import { makeCloseButton } from "#src/widget/close_button.js";
import { makeIcon } from "#src/widget/icon.js";
import { Tab } from "#src/widget/tab_view.js";

const MAX_LISTED_NODES = 300;

type SkeletonNodeType = "root" | "branchStart" | "regular" | "virtualEnd";

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

const CLOSED_END_LABEL_PATTERNS = [
  /^uncertain continuation$/i,
  /^not a branch$/i,
  /^soma$/i,
  /^(really|uncertain|anterior|posterior)?\s?ends?$/i,
];

function hasTrueEndLabel(node: SpatiallyIndexedSkeletonNodeInfo) {
  return (
    node.labels?.some(
      (label) => label.trim().toLowerCase() === CATMAID_TRUE_END_LABEL,
    ) ?? false
  );
}

function isClosedEndLabel(label: string) {
  const normalized = label.trim();
  return (
    normalized.length > 0 &&
    CLOSED_END_LABEL_PATTERNS.some((pattern) => pattern.test(normalized))
  );
}

function formatNodePosition(position: ArrayLike<number>) {
  const x = Number(position[0]);
  const y = Number(position[1]);
  const z = Number(position[2]);
  return `x ${Math.round(x)} y ${Math.round(y)} z ${Math.round(z)}`;
}

function formatNodeCoordinates(position: ArrayLike<number>) {
  const x = Number(position[0]);
  const y = Number(position[1]);
  const z = Number(position[2]);
  return `${Math.round(x)} ${Math.round(y)} ${Math.round(z)}`;
}

function formatEditableNumber(value: number | undefined, fallback = "0") {
  return value === undefined ? fallback : `${value}`;
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
  description: string | undefined,
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
  if (formatNodeCoordinates(node.position).toLowerCase().includes(filterText)) {
    return true;
  }
  if (formatNodePosition(node.position).toLowerCase().includes(filterText)) {
    return true;
  }
  return description?.toLowerCase().includes(filterText) ?? false;
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
    const collapseButton = document.createElement("button");
    collapseButton.className = "neuroglancer-spatial-skeleton-section-toggle";
    collapseButton.type = "button";
    const filterInput = document.createElement("input");
    filterInput.type = "text";
    filterInput.placeholder = "Enter ID, coordinates, tags or description";
    filterInput.className = "neuroglancer-spatial-skeleton-filter";
    const nodesSummaryBar = document.createElement("div");
    nodesSummaryBar.className = "neuroglancer-spatial-skeleton-summary-bar";
    const nodesSummary = document.createElement("div");
    nodesSummary.className = "neuroglancer-spatial-skeleton-summary";
    const nodesList = document.createElement("div");
    nodesList.className = "neuroglancer-spatial-skeleton-tree";
    nodesSection.appendChild(filterInput);
    nodesSummaryBar.appendChild(nodesSummary);
    nodesSummaryBar.appendChild(collapseButton);
    nodesSection.appendChild(nodesSummaryBar);
    nodesSection.appendChild(nodesList);
    element.appendChild(nodesSection);

    let allNodes: SpatiallyIndexedSkeletonNodeInfo[] = [];
    let activeSegmentIds: number[] = [];
    let nodesBySegment = new Map<number, SpatiallyIndexedSkeletonNodeInfo[]>();
    let filterText = "";
    let inspectionAllowed = false;
    let navigationAllowed = false;
    let labelEditingAllowed = false;
    let nodePropertyEditingAllowed = false;
    let nodeDeletionAllowed = false;
    let listCollapsed = true;
    let propertiesDialog: Overlay | undefined;
    let pendingScrollToSelectedNode = false;
    let refreshRequestId = 0;
    let loadedNodeSummarySuffix = "";
    let hoveredViewerNodeId: number | undefined;
    const pendingDeleteNodes = new Set<number>();
    const pendingTrueEndNodes = new Set<number>();
    const renderedRowsByNodeId = new Map<number, HTMLDivElement>();
    const renderedEntriesByNodeId = new Map<number, HTMLDivElement>();
    const skeletonState = layer.spatialSkeletonState;
    const mouseState = layer.manager.root.layerSelectedValues.mouseState;
    const navigationGraphCache = new Map<
      number,
      {
        nodes: readonly SpatiallyIndexedSkeletonNodeInfo[];
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

    const closePropertiesDialog = () => {
      propertiesDialog?.close();
      propertiesDialog = undefined;
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
        | "editNodeProperties"
        | "editNodeLabels"
        | "deleteNodes"
        | readonly (
            | "inspectSkeletons"
            | "editNodeProperties"
            | "editNodeLabels"
            | "deleteNodes"
          )[],
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

    const clearBranchEndAnchor = () => {
      layer.spatialSkeletonTreeEndNodeId.value = undefined;
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

    const getNodeDescriptionText = (node: SpatiallyIndexedSkeletonNodeInfo) => {
      const localDescription = layer
        .getSpatialSkeletonNodeDescription(node.nodeId)
        ?.trim();
      if (localDescription !== undefined && localDescription.length > 0) {
        return localDescription;
      }
      const descriptiveLabels = (node.labels ?? [])
        .map((label) => label.trim())
        .filter((label) => label.length > 0 && !isClosedEndLabel(label));
      return descriptiveLabels.length > 0
        ? descriptiveLabels.join(", ")
        : undefined;
    };

    const getNodeDisplayId = (node: SpatiallyIndexedSkeletonNodeInfo) => {
      return node.nodeId;
    };

    const getHoveredNodeIdFromViewer = () => {
      if (!mouseState.active) return undefined;
      const pickedRenderLayer = mouseState.pickedRenderLayer;
      if (
        pickedRenderLayer !== null &&
        !layer.renderLayers.includes(pickedRenderLayer)
      ) {
        return undefined;
      }
      const pickedNodeId = mouseState.pickedSpatialSkeletonNodeId;
      return typeof pickedNodeId === "number" &&
        Number.isSafeInteger(pickedNodeId) &&
        pickedNodeId > 0
        ? pickedNodeId
        : undefined;
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

    const getNodeTypeDisplayLabel = (
      type: SkeletonNodeType,
      nodeIsTrueEnd: boolean,
    ) => {
      if (nodeIsTrueEnd) return "True end";
      switch (type) {
        case "root":
          return "Root";
        case "branchStart":
          return "Branch start";
        case "virtualEnd":
          return "End";
        default:
          return "Regular";
      }
    };

    const openNodePropertiesDialog = (
      node: SpatiallyIndexedSkeletonNodeInfo,
      type: SkeletonNodeType,
    ) => {
      closePropertiesDialog();
      const overlay = new Overlay();
      propertiesDialog = overlay;
      overlay.registerDisposer(() => {
        if (propertiesDialog === overlay) {
          propertiesDialog = undefined;
        }
      });
      overlay.content.classList.add(
        "neuroglancer-spatial-skeleton-properties-dialog",
      );
      overlay.container.classList.add(
        "neuroglancer-spatial-skeleton-properties-overlay",
      );

      const nodeIsTrueEnd = hasTrueEndLabel(node);
      const typeIconSvg = nodeIsTrueEnd ? svg_flag : NODE_TYPE_ICONS[type];
      const typeIconTitle = nodeIsTrueEnd ? "true end" : NODE_TYPE_LABELS[type];
      const description = getNodeDescriptionText(node);
      const initialRadius = node.radius ?? 0;
      const initialConfidence = node.confidence ?? 0;

      const header = document.createElement("div");
      header.className = "neuroglancer-spatial-skeleton-properties-header";
      const title = document.createElement("div");
      title.className = "neuroglancer-spatial-skeleton-properties-title";
      title.textContent = "Node properties";
      const closeButton = document.createElement("button");
      closeButton.className = "neuroglancer-spatial-skeleton-properties-close";
      closeButton.type = "button";
      closeButton.title = "Close";
      closeButton.appendChild(
        makeCloseButton({
          title: "Close",
          clickable: false,
        }),
      );
      closeButton.addEventListener("click", () => overlay.close());
      header.appendChild(title);
      header.appendChild(closeButton);
      overlay.content.appendChild(header);

      const preview = document.createElement("div");
      preview.className = "neuroglancer-spatial-skeleton-properties-preview";
      const previewIcon = document.createElement("span");
      previewIcon.className =
        "neuroglancer-spatial-skeleton-properties-preview-icon";
      previewIcon.appendChild(
        makeIcon({
          svg: typeIconSvg,
          title: typeIconTitle,
          clickable: false,
        }),
      );
      const previewId = document.createElement("span");
      previewId.className =
        "neuroglancer-spatial-skeleton-properties-preview-id neuroglancer-spatial-skeleton-node-segment-chip";
      const segmentChipColors = getSegmentChipColors(node.segmentId);
      previewId.textContent = String(node.segmentId);
      previewId.style.backgroundColor = segmentChipColors.background;
      previewId.style.color = segmentChipColors.foreground;
      const previewCoordinates = document.createElement("span");
      previewCoordinates.className =
        "neuroglancer-spatial-skeleton-properties-preview-coordinates";
      previewCoordinates.textContent = formatNodeCoordinates(node.position);
      preview.appendChild(previewIcon);
      preview.appendChild(previewId);
      preview.appendChild(previewCoordinates);
      overlay.content.appendChild(preview);

      const body = document.createElement("div");
      body.className = "neuroglancer-spatial-skeleton-properties-body";
      overlay.content.appendChild(body);

      const appendPropertyRow = (
        label: string,
        value: string | HTMLElement,
      ) => {
        const row = document.createElement("div");
        row.className = "neuroglancer-spatial-skeleton-properties-row";
        const labelElement = document.createElement("div");
        labelElement.className =
          "neuroglancer-spatial-skeleton-properties-label";
        labelElement.textContent = label;
        const valueElement = document.createElement("div");
        valueElement.className =
          "neuroglancer-spatial-skeleton-properties-value";
        if (typeof value === "string") {
          valueElement.textContent = value;
        } else {
          valueElement.appendChild(value);
        }
        row.appendChild(labelElement);
        row.appendChild(valueElement);
        body.appendChild(row);
      };

      const coordinatesValue = document.createElement("span");
      coordinatesValue.className =
        "neuroglancer-spatial-skeleton-properties-coordinates";
      for (const [axis, value] of [
        ["x", Math.round(Number(node.position[0]))],
        ["y", Math.round(Number(node.position[1]))],
        ["z", Math.round(Number(node.position[2]))],
      ] as const) {
        if (coordinatesValue.childNodes.length > 0) {
          coordinatesValue.appendChild(document.createTextNode(" "));
        }
        const axisLabel = document.createElement("span");
        axisLabel.className =
          "neuroglancer-spatial-skeleton-properties-coordinate-axis";
        axisLabel.textContent = `${axis} `;
        const axisValue = document.createElement("span");
        axisValue.className =
          "neuroglancer-spatial-skeleton-properties-coordinate-value";
        axisValue.textContent = String(value);
        coordinatesValue.appendChild(axisLabel);
        coordinatesValue.appendChild(axisValue);
      }

      appendPropertyRow("Coordinates", coordinatesValue);
      appendPropertyRow("ID", String(getNodeDisplayId(node)));
      appendPropertyRow(
        "Node type",
        getNodeTypeDisplayLabel(type, nodeIsTrueEnd),
      );
      const radiusInput = document.createElement("input");
      radiusInput.className = "neuroglancer-spatial-skeleton-properties-input";
      radiusInput.type = "number";
      radiusInput.step = "any";
      radiusInput.value = formatEditableNumber(node.radius);
      appendPropertyRow("Radius", radiusInput);
      const confidenceInput = document.createElement("input");
      confidenceInput.className =
        "neuroglancer-spatial-skeleton-properties-input";
      confidenceInput.type = "number";
      confidenceInput.min = "0";
      confidenceInput.max = "100";
      confidenceInput.step = "any";
      confidenceInput.value = formatEditableNumber(node.confidence);
      appendPropertyRow("Confidence level", confidenceInput);
      if (description !== undefined) {
        appendPropertyRow("Description", description);
      }

      const footer = document.createElement("div");
      footer.className = "neuroglancer-spatial-skeleton-properties-footer";
      const saveButton = document.createElement("button");
      saveButton.className = "neuroglancer-spatial-skeleton-properties-save";
      saveButton.type = "button";
      saveButton.textContent = "Save changes";
      footer.appendChild(saveButton);
      overlay.content.appendChild(footer);

      let savePending = false;
      const setInputValidity = (
        input: HTMLInputElement,
        valid: boolean,
        title: string | undefined,
      ) => {
        input.classList.toggle(
          "neuroglancer-spatial-skeleton-properties-input-invalid",
          !valid,
        );
        if (title === undefined) {
          input.removeAttribute("title");
        } else {
          input.title = title;
        }
      };
      const getParsedProperties = () => {
        const radius = Number(radiusInput.value);
        const confidence = Number(confidenceInput.value);
        const radiusValid = Number.isFinite(radius);
        const confidenceValid =
          Number.isFinite(confidence) && confidence >= 0 && confidence <= 100;
        return {
          radius,
          confidence,
          radiusValid,
          confidenceValid,
        };
      };
      const updateDialogState = () => {
        const disabledReason = nodePropertyEditingAllowed
          ? undefined
          : layer.getSpatialSkeletonActionsDisabledReason("editNodeProperties");
        const { radiusValid, confidenceValid } = getParsedProperties();
        const editable = disabledReason === undefined && !savePending;
        radiusInput.disabled = !editable;
        confidenceInput.disabled = !editable;
        saveButton.disabled = !editable || !radiusValid || !confidenceValid;
        saveButton.title =
          disabledReason ??
          (savePending
            ? "Saving changes"
            : radiusValid && confidenceValid
              ? "Save changes"
              : "Enter a valid radius and a confidence between 0 and 100");
        setInputValidity(
          radiusInput,
          radiusValid,
          radiusValid ? undefined : "Radius must be a finite number.",
        );
        setInputValidity(
          confidenceInput,
          confidenceValid,
          confidenceValid ? undefined : "Confidence must be between 0 and 100.",
        );
      };
      radiusInput.addEventListener("input", updateDialogState);
      confidenceInput.addEventListener("input", updateDialogState);
      saveButton.addEventListener("click", () => {
        if (!ensureActionsAllowed("editNodeProperties")) return;
        const { radius, confidence, radiusValid, confidenceValid } =
          getParsedProperties();
        if (!radiusValid || !confidenceValid) {
          updateDialogState();
          return;
        }
        const skeletonLayer = layer.getSpatiallyIndexedSkeletonLayer();
        if (skeletonLayer === undefined) {
          StatusMessage.showTemporaryMessage(
            "No active spatial skeleton layer found for property update.",
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
        const radiusChanged = radius !== initialRadius;
        const confidenceChanged = confidence !== initialConfidence;
        if (!radiusChanged && !confidenceChanged) {
          overlay.close();
          return;
        }
        savePending = true;
        updateDialogState();
        void (async () => {
          try {
            if (radiusChanged) {
              await skeletonSource.updateRadius(node.nodeId, radius);
            }
            if (confidenceChanged) {
              await skeletonSource.updateConfidence(node.nodeId, confidence);
            }
            applyNodePropertiesLocally(node.nodeId, {
              radius,
              confidence,
            });
            StatusMessage.showTemporaryMessage(
              `Updated node ${node.nodeId} properties.`,
            );
            overlay.close();
          } catch (error) {
            const message =
              error instanceof Error ? error.message : String(error);
            StatusMessage.showTemporaryMessage(
              `Failed to update node properties: ${message}`,
            );
          } finally {
            savePending = false;
            if (propertiesDialog === overlay) {
              updateDialogState();
            }
          }
        })();
      });
      updateDialogState();
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
        return getChildNodeFromGraph(
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
            await skeletonSource.setTrueEnd(node.nodeId);
          } else {
            await skeletonSource.removeTrueEnd(node.nodeId);
          }
          StatusMessage.showTemporaryMessage(
            present
              ? `Set node ${node.nodeId} as true end.`
              : `Removed true end from node ${node.nodeId}.`,
          );
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
          clearBranchEndAnchor();
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
      "Go to start of the branch",
      () => {
        const context = getSelectedNavigationContext();
        if (context === undefined) return;
        const { selectedNode, skeletonApi } = context;
        void (async () => {
          try {
            clearBranchEndAnchor();
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
    const goTreeEndButton = makeNavIconButton(
      svg_chevrons_right,
      "Go to end of the branch",
      () => {
        const context = getSelectedNavigationContext();
        if (context === undefined) return;
        const { selectedNode, skeletonApi } = context;
        void (async () => {
          try {
            clearBranchEndAnchor();
            navigateToNodeTarget(await skeletonApi.getBranchEnd(selectedNode.nodeId));
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
    const cycleBranchesButton = makeNavIconButton(
      svg_retweet,
      "Cycle through level nodes",
      () => {
        const context = getSelectedNavigationContext();
        if (context === undefined) return;
        const { selectedNode, skeletonApi } = context;
        void (async () => {
          try {
            clearBranchEndAnchor();
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
    const goParentButton = makeNavIconButton(
      svg_arrow_left,
      "Go to parent",
      () => {
        const context = getSelectedNavigationContext();
        if (context === undefined) return;
        const { selectedNode, skeletonApi } = context;
        void (async () => {
          try {
            clearBranchEndAnchor();
            const target = await skeletonApi.getParentNode(selectedNode.nodeId);
            if (target === undefined) {
              StatusMessage.showTemporaryMessage("Selected node has no parent.");
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
    const goChildButton = makeNavIconButton(
      svg_arrow_right,
      "Go to child",
      () => {
        const context = getSelectedNavigationContext();
        if (context === undefined) return;
        const { selectedNode, skeletonApi } = context;
        void (async () => {
          try {
            clearBranchEndAnchor();
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
    const goUnfinishedBranchButton = makeNavIconButton(
      svg_chevron_right,
      "Go to unfinished node",
      () => {
        clearBranchEndAnchor();
        goToClosestUnfinishedBranch();
      },
    );
    toolbox.appendChild(navTools);
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

    const buildSegmentListRows = (
      segmentId: number,
      segmentNodes: SpatiallyIndexedSkeletonNodeInfo[],
    ) => {
      if (segmentNodes.length === 0) {
        return [];
      }
      const graph = getSegmentNavigationGraph(segmentId);
      const { nodeById, childrenByParent } = graph;

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
        let visible = nodeMatchesFilter(
          node,
          filterText,
          getNodeDescriptionText(node),
        );
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
        type: SkeletonNodeType;
        isLeaf: boolean;
      }> = [];
      for (const nodeId of getFlatListNodeIds(graph)) {
        if (!isNodeVisible(nodeId)) continue;
        const node = nodeById.get(nodeId);
        if (node === undefined) continue;
        const children = childrenByParent.get(nodeId) ?? [];
        const parentInTree =
          node.parentNodeId !== undefined && nodeById.has(node.parentNodeId);
        const type = classifyNodeType(node, children.length, parentInTree);
        if (!(listCollapsed && type === "regular")) {
          rows.push({ node, type, isLeaf: children.length === 0 });
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
      renderedRowsByNodeId.clear();
      renderedEntriesByNodeId.clear();
      if (activeSegmentIds.length === 0) {
        return;
      }
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

      let renderedNodeCount = 0;
      let overflowNodeCount = 0;
      let matchedRows = 0;
      for (const segmentId of activeSegmentIds) {
        const segmentNodes = nodesBySegment.get(segmentId) ?? [];
        const rows = buildSegmentListRows(segmentId, segmentNodes);
        matchedRows += rows.length;
        for (const { node, type, isLeaf } of rows) {
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
                !ensureActionsAllowed("inspectSkeletons", {
                  requireMaxLod: false,
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
                !ensureActionsAllowed("inspectSkeletons", {
                  requireMaxLod: false,
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
                !ensureActionsAllowed("inspectSkeletons", {
                  requireMaxLod: false,
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

          const nodeIsTrueEnd = hasTrueEndLabel(node);
          const typeIconSvg = nodeIsTrueEnd ? svg_flag : NODE_TYPE_ICONS[type];
          const typeIconTitle = nodeIsTrueEnd
            ? "true end"
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
            typeIcon.disabled = !labelEditingAllowed || typeButtonPending;
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
          if (type === "root") {
            const rootChip = document.createElement("span");
            rootChip.className =
              "neuroglancer-spatial-skeleton-node-segment-chip";
            const segmentChipColors = getSegmentChipColors(node.segmentId);
            rootChip.textContent = String(node.segmentId);
            rootChip.style.backgroundColor = segmentChipColors.background;
            rootChip.style.color = segmentChipColors.foreground;
            rootChip.title = `segment ${node.segmentId}`;
            idCell.appendChild(rootChip);
          }

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
          actions.appendChild(
            makeRowActionButton(
              svg_info,
              "show node properties",
              () => {
                selectNode(node, { moveView: false });
                openNodePropertiesDialog(node, type);
              },
              !inspectionAllowed,
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

      if (matchedRows === 0) {
        const empty = document.createElement("div");
        empty.className = "neuroglancer-spatial-skeleton-summary";
        empty.textContent =
          filterText.length === 0 ? "No loaded nodes." : "No matching nodes.";
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

    const summarizeNodeState = (summarySuffix = "") => {
      let branchCount = 0;
      for (const segmentId of activeSegmentIds) {
        const segmentNodes = nodesBySegment.get(segmentId) ?? [];
        if (segmentNodes.length === 0) continue;
        const { childrenByParent, rootNodeIds } =
          getSegmentNavigationGraph(segmentId);
        branchCount += rootNodeIds.length;
        for (const childNodeIds of childrenByParent.values()) {
          if (childNodeIds.length > 1) {
            branchCount += childNodeIds.length - 1;
          }
        }
      }
      nodesSummary.textContent = `${activeSegmentIds.length} skeleton${
        activeSegmentIds.length === 1 ? "" : "s"
      }, ${branchCount} branch${branchCount === 1 ? "" : "es"}, ${
        allNodes.length
      } node${allNodes.length === 1 ? "" : "s"}`;
      if (summarySuffix.trim().length > 0) {
        nodesSummary.title = summarySuffix.trim();
      } else {
        nodesSummary.removeAttribute("title");
      }
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

    const applyNodePropertiesLocally = (
      nodeId: number,
      updates: {
        radius: number;
        confidence: number;
      },
    ) => {
      const updateNode = (node: SpatiallyIndexedSkeletonNodeInfo) => {
        if (
          node.radius === updates.radius &&
          node.confidence === updates.confidence
        ) {
          return node;
        }
        return {
          ...node,
          radius: updates.radius,
          confidence: updates.confidence,
        };
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
      activeSegmentIds = [
        ...getVisibleSegments(
          layer.displayState.segmentationGroupState.value,
        ).keys(),
      ]
        .map((segmentId) => Number(segmentId))
        .filter((segmentId) => Number.isFinite(segmentId))
        .sort((a, b) => a - b);
      if (skeletonLayer === undefined || activeSegmentIds.length === 0) {
        allNodes = [];
        nodesBySegment = new Map();
        closePropertiesDialog();
        layer.clearSpatialSkeletonNodeSelection(false);
        nodesSummary.removeAttribute("title");
        nodesSummary.textContent =
          "Make one or more segments visible in Seg tab to load editable skeleton nodes.";
        updateList();
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
          closePropertiesDialog();
          layer.clearSpatialSkeletonNodeSelection(false);
          nodesSummary.removeAttribute("title");
          nodesSummary.textContent =
            "Failed to load full skeleton data from the active source.";
          updateList();
        }
      })();
    };

    const updateGateStatus = () => {
      const nextInspectionAllowed =
        layer.getSpatialSkeletonActionsDisabledReason("inspectSkeletons", {
          requireMaxLod: false,
          requireVisibleChunks: false,
        }) === undefined;
      const nextNavigationAllowed = nextInspectionAllowed;
      const nextLabelEditingAllowed =
        layer.getSpatialSkeletonActionsDisabledReason("editNodeLabels") ===
        undefined;
      const nextNodePropertyEditingAllowed =
        layer.getSpatialSkeletonActionsDisabledReason("editNodeProperties") ===
        undefined;
      const nextNodeDeletionAllowed =
        layer.getSpatialSkeletonActionsDisabledReason("deleteNodes") ===
        undefined;
      const gateStateChanged =
        inspectionAllowed !== nextInspectionAllowed ||
        navigationAllowed !== nextNavigationAllowed ||
        labelEditingAllowed !== nextLabelEditingAllowed ||
        nodePropertyEditingAllowed !== nextNodePropertyEditingAllowed ||
        nodeDeletionAllowed !== nextNodeDeletionAllowed;

      inspectionAllowed = nextInspectionAllowed;
      navigationAllowed = nextNavigationAllowed;
      labelEditingAllowed = nextLabelEditingAllowed;
      nodePropertyEditingAllowed = nextNodePropertyEditingAllowed;
      nodeDeletionAllowed = nextNodeDeletionAllowed;

      filterInput.disabled = !inspectionAllowed;
      for (const control of gatedControls) {
        control.disabled = !navigationAllowed;
      }
      if (gateStateChanged) {
        updateList();
      }
    };

    filterInput.addEventListener("input", () => {
      filterText = filterInput.value.trim().toLowerCase();
      updateList();
    });
    collapseButton.addEventListener("click", () => {
      listCollapsed = !listCollapsed;
      updateCollapseButton();
      updateList();
    });

    this.registerDisposer(() => {
      closePropertiesDialog();
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
    updateHoveredViewerNode();
    refreshNodes();
  }
}
