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
import type { SpatiallyIndexedSkeletonNode } from "#src/skeleton/api.js";
import type {
  SpatialSkeletonAddNodeCommandOptions,
  SpatialSkeletonEditController,
  SpatialSkeletonInsertNodeCommandOptions,
  SpatialSkeletonMergeEndpoint,
  SpatialSkeletonMoveNodeCommandOptions,
  SpatialSkeletonNodeDescriptionCommandOptions,
  SpatialSkeletonNodePropertiesCommandOptions,
  SpatialSkeletonNodeTrueEndCommandOptions,
} from "#src/skeleton/edit_controller.js";
import type { SpatialSkeletonCommand } from "#src/skeleton/command_history.js";
import { getSpatialSkeletonEditController } from "#src/skeleton/spatial_skeleton_manager.js";
import { StatusMessage } from "#src/status.js";

function getController(
  layer: SegmentationUserLayer,
): SpatialSkeletonEditController {
  const skeletonLayer = layer.getSpatiallyIndexedSkeletonLayer();
  const controller = getSpatialSkeletonEditController(skeletonLayer);
  if (controller === undefined) {
    throw new Error(
      "Unable to resolve editable skeleton source for the active layer.",
    );
  }
  return controller;
}

function requireCommand(
  command: SpatialSkeletonCommand | undefined,
  message: string,
) {
  if (command === undefined) {
    throw new Error(message);
  }
  return command;
}

function executeCommand(
  layer: SegmentationUserLayer,
  command: SpatialSkeletonCommand,
) {
  return layer.spatialSkeletonState.commandHistory.execute(command);
}

function executeCommandWithPendingMessage<T>(
  promise: Promise<T>,
  message: string,
) {
  const status = StatusMessage.showMessage(message);
  return promise.finally(() => status.dispose());
}

export function executeSpatialSkeletonAddNode(
  layer: SegmentationUserLayer,
  options: SpatialSkeletonAddNodeCommandOptions,
) {
  return executeCommandWithPendingMessage(
    executeCommand(
      layer,
      getController(layer).createAddNodeCommand(layer, options),
    ),
    "Creating node...",
  );
}

export function executeSpatialSkeletonInsertNode(
  layer: SegmentationUserLayer,
  options: SpatialSkeletonInsertNodeCommandOptions,
) {
  return executeCommandWithPendingMessage(
    executeCommand(
      layer,
      getController(layer).createInsertNodeCommand(layer, options),
    ),
    "Inserting node...",
  );
}

export function executeSpatialSkeletonMoveNode(
  layer: SegmentationUserLayer,
  options: SpatialSkeletonMoveNodeCommandOptions,
) {
  return executeCommand(
    layer,
    getController(layer).createMoveNodeCommand(layer, options),
  );
}

export function executeSpatialSkeletonDeleteNode(
  layer: SegmentationUserLayer,
  node: SpatiallyIndexedSkeletonNode,
) {
  return executeCommandWithPendingMessage(
    executeCommand(
      layer,
      getController(layer).createDeleteNodeCommand(layer, node),
    ),
    "Deleting node...",
  );
}

export function executeSpatialSkeletonNodeDescriptionUpdate(
  layer: SegmentationUserLayer,
  options: SpatialSkeletonNodeDescriptionCommandOptions,
) {
  const command = requireCommand(
    getController(layer).createNodeDescriptionCommand?.(layer, options),
    "The active skeleton source does not support node description editing.",
  );
  return executeCommand(layer, command);
}

export function executeSpatialSkeletonNodeTrueEndUpdate(
  layer: SegmentationUserLayer,
  options: SpatialSkeletonNodeTrueEndCommandOptions,
) {
  const command = requireCommand(
    getController(layer).createNodeTrueEndCommand?.(layer, options),
    "The active skeleton source does not support node true-end editing.",
  );
  return executeCommand(layer, command);
}

export function executeSpatialSkeletonNodePropertiesUpdate(
  layer: SegmentationUserLayer,
  options: SpatialSkeletonNodePropertiesCommandOptions,
) {
  const command = requireCommand(
    getController(layer).createNodePropertiesCommand?.(layer, options),
    "The active skeleton source does not support node property editing.",
  );
  return executeCommand(layer, command);
}

export function executeSpatialSkeletonReroot(
  layer: SegmentationUserLayer,
  node: Pick<
    SpatiallyIndexedSkeletonNode,
    "nodeId" | "segmentId" | "parentNodeId"
  >,
) {
  const command = requireCommand(
    getController(layer).createRerootCommand?.(layer, node),
    "The active skeleton source does not support skeleton rerooting.",
  );
  return executeCommand(layer, command);
}

export function executeSpatialSkeletonSplit(
  layer: SegmentationUserLayer,
  node: Pick<SpatiallyIndexedSkeletonNode, "nodeId" | "segmentId">,
) {
  return executeCommandWithPendingMessage(
    executeCommand(
      layer,
      getController(layer).createSplitCommand(layer, node),
    ),
    "Splitting skeleton...",
  );
}

export function executeSpatialSkeletonMerge(
  layer: SegmentationUserLayer,
  firstNode: SpatialSkeletonMergeEndpoint,
  secondNode: SpatialSkeletonMergeEndpoint,
) {
  return executeCommandWithPendingMessage(
    executeCommand(
      layer,
      getController(layer).createMergeCommand(layer, firstNode, secondNode),
    ),
    "Merging skeletons...",
  );
}

export async function undoSpatialSkeletonCommand(layer: SegmentationUserLayer) {
  const changed = await layer.spatialSkeletonState.commandHistory.undo();
  if (!changed) {
    return false;
  }
  return true;
}

export async function redoSpatialSkeletonCommand(layer: SegmentationUserLayer) {
  const changed = await layer.spatialSkeletonState.commandHistory.redo();
  if (!changed) {
    return false;
  }
  return true;
}
