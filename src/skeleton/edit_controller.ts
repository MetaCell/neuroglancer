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

import type { SpatialSkeletonAction } from "#src/skeleton/actions.js";
import type {
  SpatiallyIndexedSkeletonNode,
  SpatialSkeletonVector,
} from "#src/skeleton/api.js";
import type { SpatialSkeletonCommand } from "#src/skeleton/command_history.js";

export interface SpatialSkeletonNodeFeatureCapabilities {
  description?: boolean;
  trueEnd?: boolean;
  radius?: boolean;
  confidenceValues?: readonly number[];
}

export interface SpatialSkeletonEditCapabilities {
  nodeFeatures?: SpatialSkeletonNodeFeatureCapabilities;
}

export interface SpatialSkeletonAddNodeCommandOptions {
  skeletonId: number;
  parentNodeId: number | undefined;
  positionInModelSpace: SpatialSkeletonVector;
}

export interface SpatialSkeletonInsertNodeCommandOptions {
  skeletonId: number;
  parentNodeId: number;
  childNodeIds: readonly number[];
  positionInModelSpace: SpatialSkeletonVector;
}

export interface SpatialSkeletonMoveNodeCommandOptions {
  node: SpatiallyIndexedSkeletonNode;
  nextPositionInModelSpace: SpatialSkeletonVector;
}

export interface SpatialSkeletonNodeDescriptionCommandOptions {
  node: SpatiallyIndexedSkeletonNode;
  nextDescription?: string;
}

export interface SpatialSkeletonNodeTrueEndCommandOptions {
  node: SpatiallyIndexedSkeletonNode;
  nextIsTrueEnd: boolean;
}

export interface SpatialSkeletonNodePropertiesCommandOptions {
  node: SpatiallyIndexedSkeletonNode;
  next: { radius: number; confidence: number };
}

export interface SpatialSkeletonMergeEndpoint {
  nodeId: number;
  segmentId: number;
  sourceState?: unknown;
}

export interface SpatialSkeletonEditController {
  readonly capabilities?: SpatialSkeletonEditCapabilities;
  supports(action: SpatialSkeletonAction): boolean;
  createAddNodeCommand(
    layer: any,
    options: SpatialSkeletonAddNodeCommandOptions,
  ): SpatialSkeletonCommand;
  createInsertNodeCommand(
    layer: any,
    options: SpatialSkeletonInsertNodeCommandOptions,
  ): SpatialSkeletonCommand;
  createMoveNodeCommand(
    layer: any,
    options: SpatialSkeletonMoveNodeCommandOptions,
  ): SpatialSkeletonCommand;
  createDeleteNodeCommand(
    layer: any,
    node: SpatiallyIndexedSkeletonNode,
  ): SpatialSkeletonCommand;
  createNodeDescriptionCommand?(
    layer: any,
    options: SpatialSkeletonNodeDescriptionCommandOptions,
  ): SpatialSkeletonCommand;
  createNodeTrueEndCommand?(
    layer: any,
    options: SpatialSkeletonNodeTrueEndCommandOptions,
  ): SpatialSkeletonCommand;
  createNodePropertiesCommand?(
    layer: any,
    options: SpatialSkeletonNodePropertiesCommandOptions,
  ): SpatialSkeletonCommand;
  createRerootCommand?(
    layer: any,
    node: Pick<
      SpatiallyIndexedSkeletonNode,
      "nodeId" | "segmentId" | "parentNodeId"
    >,
  ): SpatialSkeletonCommand;
  createSplitCommand(
    layer: any,
    node: Pick<SpatiallyIndexedSkeletonNode, "nodeId" | "segmentId">,
  ): SpatialSkeletonCommand;
  createMergeCommand(
    layer: any,
    firstNode: SpatialSkeletonMergeEndpoint,
    secondNode: SpatialSkeletonMergeEndpoint,
  ): SpatialSkeletonCommand;
}
