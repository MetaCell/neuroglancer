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

/**
 * @file Declarations of the built-in viewer commands.
 *
 * Each entry names an existing DOM action (`id` === the `action:` id dispatched
 * by the default input-event bindings) and gives it an explicit, human-readable
 * label and help description. The registry — not the bindings — is now the
 * authoritative list of commands; the shortcut shown for each command is looked
 * up from whatever binding happens to be installed (see {@link CommandCatalog}).
 *
 * Commands whose behaviour is per-entity or otherwise dynamic (layer toggles,
 * tool activation) are contributed by the catalog at enumeration time and are
 * intentionally *not* declared here.
 *
 * This is the built-in *seed* set, not a required registry. It exists only
 * because these commands correspond to DOM actions that predate the registry.
 * Feature code should NOT add entries here; instead register commands
 * colocated with the feature, for its own lifetime, e.g.
 *
 *     this.registerDisposer(
 *       viewer.commandRegistry.registerCallback({
 *         id: "clip.addPlane",
 *         label: "Add Clip Plane",
 *         invoke: () => this.addPlane(),
 *       }),
 *     );
 *
 * `registerAction` / `registerCallback` each return a disposer, so commands may
 * come and go with the feature (e.g. per-layer). `CommandRegistry` — not this
 * file — is the authoritative, runtime-enumerable list.
 */

import type {
  ActionCommandInfo,
  CommandRegistry,
} from "#src/ui/command_registry.js";

// Every built-in command is action-backed (dispatches `action:<id>`); the type
// is stamped by `registerAction` at registration time.
type BuiltinCommand = Omit<ActionCommandInfo, "type">;

const CATEGORY_VIEW = "View";
const CATEGORY_NAVIGATION = "Navigation";
const CATEGORY_ANNOTATION = "Annotation";
const CATEGORY_LAYERS = "Layers";
const CATEGORY_STATE = "State";
const CATEGORY_TOOLS = "Tools";

const AXES = ["X", "Y", "Z"] as const;

// Directional position nudges (arrow keys / , . / [ ] in the data panels).
function axisMoveCommands(): BuiltinCommand[] {
  const commands: BuiltinCommand[] = [];
  for (const axis of AXES) {
    const lower = axis.toLowerCase();
    commands.push(
      {
        id: `${lower}-`,
        label: `Move −${axis}`,
        description: `Move the view one step in the −${axis} direction.`,
        category: CATEGORY_NAVIGATION,
      },
      {
        id: `${lower}+`,
        label: `Move +${axis}`,
        description: `Move the view one step in the +${axis} direction.`,
        category: CATEGORY_NAVIGATION,
      },
    );
  }
  return commands;
}

// Relative rotations about each axis (r / e and shift+arrow keys).
function axisRotateCommands(): BuiltinCommand[] {
  const commands: BuiltinCommand[] = [];
  for (const axis of AXES) {
    const lower = axis.toLowerCase();
    commands.push(
      {
        id: `rotate-relative-${lower}-`,
        label: `Rotate −${axis}`,
        description: `Rotate the view a small amount about the ${axis} axis (negative direction).`,
        category: CATEGORY_NAVIGATION,
      },
      {
        id: `rotate-relative-${lower}+`,
        label: `Rotate +${axis}`,
        description: `Rotate the view a small amount about the ${axis} axis (positive direction).`,
        category: CATEGORY_NAVIGATION,
      },
    );
  }
  return commands;
}

const STATIC_COMMANDS: readonly BuiltinCommand[] = [
  // View toggles.
  {
    id: "toggle-show-slices",
    label: "Toggle Slices in 3D",
    description: "Show or hide the cross-section slices in the 3D view.",
    category: CATEGORY_VIEW,
  },
  {
    id: "toggle-scale-bar",
    label: "Toggle Scale Bar",
    description: "Show or hide the scale bar overlay.",
    category: CATEGORY_VIEW,
  },
  {
    id: "toggle-axis-lines",
    label: "Toggle Axis Lines",
    description: "Show or hide the axis line indicators.",
    category: CATEGORY_VIEW,
  },
  {
    id: "toggle-orthographic-projection",
    label: "Toggle Orthographic Projection",
    description:
      "Switch the 3D view between perspective and orthographic projection.",
    category: CATEGORY_VIEW,
  },
  {
    id: "toggle-default-annotations",
    label: "Toggle Bounding Box",
    description: "Show or hide the default bounding-box annotations.",
    category: CATEGORY_VIEW,
  },
  {
    id: "toggle-show-statistics",
    label: "Toggle Statistics",
    description: "Show or hide the rendering statistics panel.",
    category: CATEGORY_VIEW,
  },
  {
    id: "toggle-layout",
    label: "Toggle Layout",
    description: "Cycle the data panel layout.",
    category: CATEGORY_VIEW,
  },
  {
    id: "toggle-layout-alternative",
    label: "Toggle Alternative Layout",
    description: "Cycle the alternative data panel layout.",
    category: CATEGORY_VIEW,
  },
  {
    id: "help",
    label: "Show Help",
    description: "Open the keyboard and mouse bindings help panel.",
    category: CATEGORY_VIEW,
  },
  // Navigation.
  {
    id: "snap",
    label: "Snap to Axis",
    description:
      "Snap the view orientation to the nearest axis-aligned orientation.",
    category: CATEGORY_NAVIGATION,
  },
  {
    id: "zoom-in",
    label: "Zoom In",
    description: "Zoom the view in.",
    category: CATEGORY_NAVIGATION,
  },
  {
    id: "zoom-out",
    label: "Zoom Out",
    description: "Zoom the view out.",
    category: CATEGORY_NAVIGATION,
  },
  {
    id: "depth-range-decrease",
    label: "Decrease Depth Range",
    description: "Decrease the visible depth range of the 3D projection.",
    category: CATEGORY_NAVIGATION,
  },
  {
    id: "depth-range-increase",
    label: "Increase Depth Range",
    description: "Increase the visible depth range of the 3D projection.",
    category: CATEGORY_NAVIGATION,
  },
  {
    id: "t-",
    label: "Previous Timestep",
    description: "Step backward one frame along the time axis.",
    category: CATEGORY_NAVIGATION,
  },
  {
    id: "t+",
    label: "Next Timestep",
    description: "Step forward one frame along the time axis.",
    category: CATEGORY_NAVIGATION,
  },
  // Layers / segmentation.
  {
    id: "add-layer",
    label: "Add Layer",
    description: "Add a new layer to the viewer.",
    category: CATEGORY_LAYERS,
  },
  {
    id: "recolor",
    label: "Randomize Colors",
    description: "Assign a new random color seed to segmentation layers.",
    category: CATEGORY_LAYERS,
  },
  {
    id: "clear-segments",
    label: "Clear Selected Segments",
    description: "Deselect all currently selected segments.",
    category: CATEGORY_LAYERS,
  },
  // Annotation.
  {
    id: "finish-annotation",
    label: "Finish Annotation",
    description: "Complete the annotation currently being drawn.",
    category: CATEGORY_ANNOTATION,
  },
  {
    id: "undo-annotation-step",
    label: "Undo Annotation Step",
    description: "Undo the last point added to the in-progress annotation.",
    category: CATEGORY_ANNOTATION,
  },
  // State — these have no default key binding; before the registry they were
  // special-cased so the palette could surface them at all.
  {
    id: "edit-json-state",
    label: "Edit JSON State",
    description: "Open an editor for the raw viewer JSON state.",
    category: CATEGORY_STATE,
  },
  {
    id: "screenshot",
    label: "Screenshot",
    description: "Capture a screenshot of the current view.",
    category: CATEGORY_STATE,
  },
  // Tools.
  {
    id: "deactivate-active-tool",
    label: "Deactivate Active Tool",
    description: "Deactivate whichever tool is currently active.",
    category: CATEGORY_TOOLS,
  },
];

/**
 * Registers the built-in commands into `registry`. Called once during default
 * viewer setup. The registry is owned (and disposed) by the viewer, so no
 * disposers are returned here — the commands live for the viewer's lifetime.
 */
export function registerDefaultCommands(registry: CommandRegistry): void {
  for (const command of STATIC_COMMANDS) {
    registry.registerAction(command);
  }
  for (const command of axisMoveCommands()) {
    registry.registerAction(command);
  }
  for (const command of axisRotateCommands()) {
    registry.registerAction(command);
  }
}
