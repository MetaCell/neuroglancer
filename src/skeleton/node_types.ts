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
import type { SpatiallyIndexedSkeletonNodeInfo } from "#src/skeleton/frontend.js";

export type SpatialSkeletonDisplayNodeType =
  | "root"
  | "branchStart"
  | "regular"
  | "virtualEnd";

export enum SpatialSkeletonNodeFilterType {
  NONE,
  LEAF,
  VIRTUAL_END,
  TRUE_END,
}

const CLOSED_END_LABEL_PATTERNS = [
  /^uncertain continuation$/i,
  /^not a branch$/i,
  /^soma$/i,
  /^(really|uncertain|anterior|posterior)?\s?ends?$/i,
];

export function hasSpatialSkeletonTrueEndLabel(
  labels: readonly string[] | undefined,
) {
  return (
    labels?.some(
      (label) => label.trim().toLowerCase() === CATMAID_TRUE_END_LABEL,
    ) ?? false
  );
}

export function updateSpatialSkeletonTrueEndLabels(
  labels: readonly string[] | undefined,
  present: boolean,
) {
  const nextLabels = (labels ?? []).filter(
    (label) => label.trim().toLowerCase() !== CATMAID_TRUE_END_LABEL,
  );
  if (present) {
    nextLabels.push(CATMAID_TRUE_END_LABEL);
  }
  return nextLabels.length > 0 ? nextLabels : undefined;
}

export function isSpatialSkeletonClosedEndLabel(label: string) {
  const normalized = label.trim();
  return (
    normalized.length > 0 &&
    CLOSED_END_LABEL_PATTERNS.some((pattern) => pattern.test(normalized))
  );
}

export function classifySpatialSkeletonDisplayNodeType(
  node: Pick<SpatiallyIndexedSkeletonNodeInfo, "parentNodeId">,
  childCount: number | undefined,
  parentInTree = true,
): SpatialSkeletonDisplayNodeType {
  if (node.parentNodeId === undefined || !parentInTree) {
    return "root";
  }
  if (childCount === undefined) {
    return "regular";
  }
  if (childCount > 1) {
    return "branchStart";
  }
  if (childCount === 0) {
    return "virtualEnd";
  }
  return "regular";
}

export function getSpatialSkeletonNodeFilterLabel(
  filterType: SpatialSkeletonNodeFilterType,
) {
  switch (filterType) {
    case SpatialSkeletonNodeFilterType.NONE:
      return "None";
    case SpatialSkeletonNodeFilterType.LEAF:
      return "Leaf";
    case SpatialSkeletonNodeFilterType.VIRTUAL_END:
      return "Virtual end";
    case SpatialSkeletonNodeFilterType.TRUE_END:
      return "True end";
  }
}

export function matchesSpatialSkeletonNodeFilter(
  filterType: SpatialSkeletonNodeFilterType,
  options: {
    isLeaf: boolean;
    nodeIsTrueEnd: boolean;
    nodeType: SpatialSkeletonDisplayNodeType;
  },
) {
  switch (filterType) {
    case SpatialSkeletonNodeFilterType.NONE:
      return true;
    case SpatialSkeletonNodeFilterType.LEAF:
      return options.isLeaf;
    case SpatialSkeletonNodeFilterType.VIRTUAL_END:
      return options.isLeaf && !options.nodeIsTrueEnd;
    case SpatialSkeletonNodeFilterType.TRUE_END:
      return options.nodeIsTrueEnd;
  }
}

export function getSpatialSkeletonNodeIconFilterType(options: {
  nodeIsTrueEnd: boolean;
  nodeType: SpatialSkeletonDisplayNodeType;
}) {
  if (options.nodeIsTrueEnd) {
    return SpatialSkeletonNodeFilterType.TRUE_END;
  }
  if (options.nodeType === "virtualEnd") {
    return SpatialSkeletonNodeFilterType.VIRTUAL_END;
  }
  return undefined;
}
