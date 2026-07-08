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

import "#src/ui/command_palette.css";
import type { LayerManager, SelectedLayerState } from "#src/layer/index.js";
import { UserLayer } from "#src/layer/index.js";
import { Overlay } from "#src/overlay.js";
import {
  getMatchingTools,
  restoreTool,
  type GlobalToolBinder,
} from "#src/ui/tool.js";
import { parseToolQuery } from "#src/ui/tool_query.js";
import type { DebouncedFunction } from "#src/util/animation_frame_debounce.js";
import { animationFrameDebounce } from "#src/util/animation_frame_debounce.js";
import { RefCounted } from "#src/util/disposable.js";
import type {
  ActionIdentifier,
  EventAction,
  NormalizedEventIdentifier,
} from "#src/util/event_action_map.js";
import { friendlyEventIdentifier } from "#src/util/event_action_map.js";
import { Signal } from "#src/util/signal.js";
import type { InputEventBindings, Viewer } from "#src/viewer.js";

export interface CommandCatalogContext {
  globalToolBinder: GlobalToolBinder;
  layerManager: LayerManager;
  selectedLayer: SelectedLayerState;
  inputEventBindings: InputEventBindings;
}

const SUPPLEMENTAL_COMMANDS: readonly {
  actionId: ActionIdentifier;
  label: string;
}[] = [
  { actionId: "edit-json-state", label: "Edit JSON State" },
  { actionId: "screenshot", label: "Screenshot" },
];

export interface ActionBinding {
  readonly actionId: ActionIdentifier;
  readonly eventAction: EventAction;
}

interface CommandPaletteEntryBase {
  readonly label: string;
  readonly shortcut: string;
}

// Dispatched as an `action:<actionId>` DOM event, exactly as the keyboard
// shortcut would be.
export interface ActionCommandEntry extends CommandPaletteEntryBase {
  readonly kind: "action";
  readonly actionId: ActionIdentifier;
}

// Runs a callback directly (no DOM action exists for it).
export interface ExecuteCommandEntry extends CommandPaletteEntryBase {
  readonly kind: "execute";
  readonly execute: () => void;
}

// Opens a sub-palette of `children` instead of activating anything itself.
export interface GroupCommandEntry extends CommandPaletteEntryBase {
  readonly kind: "group";
  readonly children: readonly CommandPaletteEntry[];
}

export type CommandPaletteEntry =
  | ActionCommandEntry
  | ExecuteCommandEntry
  | GroupCommandEntry;

function formatKeyStroke(stroke: string): string {
  return stroke
    .split("+")
    .map((part) => {
      if (part.startsWith("key")) return part.substring(3);
      if (part.startsWith("digit")) return part.substring(5);
      if (part.startsWith("arrow")) return part.substring(5);
      return part;
    })
    .join("+");
}

function actionIdToLabel(actionId: ActionIdentifier): string {
  return actionId
    .split("-")
    .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
    .join(" ");
}

function isKeyboardEvent(normalizedId: NormalizedEventIdentifier): boolean {
  return (
    !normalizedId.includes("mouse") &&
    !normalizedId.includes("wheel") &&
    !normalizedId.includes("touch") &&
    !normalizedId.includes("click")
  );
}

// Creates a Tool instance from a palette-form JSON object (with optional "layer" field).
// Caller is responsible for disposing the returned tool.
function createToolFromJson(context: CommandCatalogContext, toolJson: unknown) {
  try {
    const json =
      typeof toolJson === "object" && toolJson !== null
        ? (toolJson as Record<string, unknown>)
        : undefined;
    const layerName = typeof json?.layer === "string" ? json.layer : undefined;
    if (layerName !== undefined) {
      const { layer: _ignored, ...rest } = json!;
      const managedLayer = context.layerManager.getLayerByName(layerName);
      const userLayer = managedLayer?.layer ?? null;
      if (userLayer === null) return undefined;
      return restoreTool(userLayer, rest);
    }
    // context is the viewer instance; restoreTool walks its prototype chain
    // to find the registered tool factory.
    return restoreTool(context, toolJson);
  } catch {
    return undefined;
  }
}

function getToolDescription(
  context: CommandCatalogContext,
  toolJson: unknown,
): string {
  const tool = createToolFromJson(context, toolJson);
  if (tool === undefined) return toolJsonToLabel(toolJson);
  const label =
    tool.context instanceof UserLayer
      ? `${tool.description} — ${tool.context.managedLayer.name}`
      : tool.description;
  tool.dispose();
  return label;
}

// Fallback label derived purely from the JSON structure (no instantiation).
function toolJsonToLabel(toolJson: unknown): string {
  const json =
    typeof toolJson === "object" && toolJson !== null
      ? (toolJson as Record<string, unknown>)
      : undefined;
  const typeName =
    typeof toolJson === "string"
      ? toolJson
      : typeof json?.type === "string"
        ? json.type
        : undefined;
  const layerName = typeof json?.layer === "string" ? json.layer : undefined;
  const base =
    typeName !== undefined
      ? typeName
          .replace(/([A-Z])/g, " $1")
          .replace(/-./g, (s) => " " + s[1].toUpperCase())
          .replace(/^./, (s) => s.toUpperCase())
          .trim()
      : "Unknown Tool";
  return layerName !== undefined ? `${base} — ${layerName}` : base;
}

function isToolLayerVisible(
  context: CommandCatalogContext,
  toolJson: unknown,
): boolean {
  const json =
    typeof toolJson === "object" && toolJson !== null
      ? (toolJson as Record<string, unknown>)
      : undefined;
  const layerName = typeof json?.layer === "string" ? json.layer : undefined;
  if (layerName === undefined) return true;
  const managedLayer = context.layerManager.getLayerByName(layerName);
  return managedLayer !== undefined && managedLayer.visible;
}

function activateUnboundTool(
  context: CommandCatalogContext,
  toolJson: unknown,
): void {
  const tool = createToolFromJson(context, toolJson);
  if (tool === undefined) return;
  // If the same tool is already bound to a key, activate that key directly
  // rather than creating a duplicate.
  const existingKey = tool.localBinder.jsonToKey.get(
    JSON.stringify(tool.toJSON()),
  );
  if (existingKey !== undefined) {
    tool.dispose();
    context.globalToolBinder.activate(existingKey);
    return;
  }
  // No key binding — activate directly without allocating a letter slot.
  context.globalToolBinder.activateDirect(tool);
}

/**
 * Walk the event action maps available on the viewer and produce a list of
 * every action with any keyboard binding. The first binding found for each
 * action is kept; subsequent bindings for the same action are ignored.
 */
export function collectActionBindings(
  inputEventBindings: InputEventBindings,
): readonly ActionBinding[] {
  const seenBindings = new Map<ActionIdentifier, EventAction>();

  const collect = (
    bindings: Iterable<[NormalizedEventIdentifier, EventAction]>,
  ) => {
    for (const [normalizedId, eventAction] of bindings) {
      if (!isKeyboardEvent(normalizedId)) continue;
      if (eventAction.action === "open-command-palette") continue;
      if (!seenBindings.has(eventAction.action)) {
        seenBindings.set(eventAction.action, eventAction);
      }
    }
  };

  collect(inputEventBindings.global.entries());
  collect(inputEventBindings.sliceView.entries());
  collect(inputEventBindings.perspectiveView.entries());

  return Array.from(seenBindings.entries(), ([actionId, eventAction]) => ({
    actionId,
    eventAction,
  }));
}

/**
 * Persistent, signal-driven catalog of command palette entries. Subscribes to
 * tool-binding and layer changes and rebuilds automatically via
 * animationFrameDebounce so the palette always reflects current viewer state
 * without rebuilding from scratch on every open.
 *
 * Actions can be represented hierarchically, with parent entries that
 * expand to show child entries when activated. For example,
 * layer actions (toggle-layer-N, select-layer-N, toggle-pick-layer-N) are
 * replaced by three hierarchical entries whose children are the individual
 * layer rows, enabling a two-step layer picker instead of a flat list.
 */
export class CommandCatalog extends RefCounted {
  commands: readonly CommandPaletteEntry[] = [];
  readonly changed = new Signal();
  private readonly debouncedRebuild: DebouncedFunction;

  constructor(private readonly context: CommandCatalogContext) {
    super();
    const debouncedRebuild = (this.debouncedRebuild = this.registerCancellable(
      animationFrameDebounce(() => this.rebuild()),
    ));
    this.registerDisposer(
      context.globalToolBinder.changed.add(debouncedRebuild),
    );
    this.registerDisposer(
      context.globalToolBinder.localBindersChanged.add(debouncedRebuild),
    );
    this.registerDisposer(
      context.layerManager.layersChanged.add(debouncedRebuild),
    );
    this.rebuild();
  }

  private rebuild() {
    const {
      globalToolBinder,
      layerManager,
      selectedLayer,
      inputEventBindings,
    } = this.context;
    const commands: CommandPaletteEntry[] = [];

    // "Deactivate Active Tool" is always present — harmless no-op when nothing is active.
    commands.push({
      kind: "action",
      label: "Deactivate Active Tool",
      shortcut: "",
      actionId: "deactivate-active-tool",
    });

    // Hierarchical layer actions — each group entry opens a sub-palette of layers.
    // The first 9 layers carry their digit-key shortcuts so users can see they
    // still work directly from the keyboard without opening the sub-palette.
    const layers = layerManager?.managedLayers ?? [];

    commands.push({
      kind: "group",
      label: "Toggle Layer",
      shortcut: "1–9",
      children: layers.map((layer, index) => ({
        kind: "execute",
        label: layer.name,
        shortcut: index < 9 ? String(index + 1) : "",
        execute: () => layer.setVisible(!layer.visible),
      })),
    });

    commands.push({
      kind: "group",
      label: "Select Layer",
      shortcut: "Ctrl+1–9",
      children: layers.map((layer, index) => ({
        kind: "execute",
        label: layer.name,
        shortcut: index < 9 ? `Ctrl+${index + 1}` : "",
        execute: () => {
          selectedLayer.layer = layer;
          selectedLayer.visible = true;
        },
      })),
    });

    commands.push({
      kind: "group",
      label: "Toggle Pick Layer",
      shortcut: "Alt+1–9",
      children: layers.map((layer, index) => ({
        kind: "execute",
        label: layer.name,
        shortcut: index < 9 ? `Alt+${index + 1}` : "",
        execute: () => {
          layer.pickEnabled = !layer.pickEnabled;
        },
      })),
    });

    const bindings = collectActionBindings(inputEventBindings);
    const shortcutByAction = new Map<ActionIdentifier, string>();
    for (const { actionId, eventAction } of bindings) {
      shortcutByAction.set(
        actionId,
        formatKeyStroke(
          friendlyEventIdentifier(eventAction.originalEventIdentifier ?? ""),
        ),
      );
    }

    for (const { actionId, eventAction } of bindings) {
      if (/^tool-[A-Z]$/.test(actionId)) continue;
      // Layer-index actions are replaced by hierarchical group entries above.
      if (/^(toggle|select|toggle-pick)-layer-\d+$/.test(actionId)) continue;

      const label = actionIdToLabel(actionId);
      const shortcut = formatKeyStroke(
        friendlyEventIdentifier(eventAction.originalEventIdentifier ?? ""),
      );
      commands.push({ kind: "action", label, shortcut, actionId });
    }

    for (const { actionId, label } of SUPPLEMENTAL_COMMANDS) {
      commands.push({ kind: "action", label, shortcut: "", actionId });
    }

    const toolQueryResult = parseToolQuery("+");
    if ("query" in toolQueryResult) {
      // Tool listers report changes to their available tool set (e.g. controls
      // that appear once a data source resolves) via this callback
      let toolSetChanged = false;
      const onListableToolsChanged = () => {
        if (toolSetChanged) return;
        toolSetChanged = true;
        this.debouncedRebuild();
      };
      const toolMatches = getMatchingTools(
        globalToolBinder,
        toolQueryResult.query,
        onListableToolsChanged,
      );

      // Build a reverse lookup from palette-JSON key to letter for currently-bound tools.
      // Keys must include getCommonToolProperties() to match the keys produced by
      // getMatchingTools, which merges commonProperties into every yielded tool JSON.
      const boundByJsonKey = new Map<string, string>();
      for (const [letter, tool] of globalToolBinder.bindings) {
        const paletteJson = {
          ...tool.localBinder.convertLocalJSONToPaletteJSON(tool.toJSON()),
          ...tool.localBinder.getCommonToolProperties(),
        };
        boundByJsonKey.set(JSON.stringify(paletteJson), letter);
      }

      for (const [jsonKey, toolJson] of toolMatches) {
        if (!isToolLayerVisible(this.context, toolJson)) continue;
        const boundLetter = boundByJsonKey.get(jsonKey);
        if (boundLetter !== undefined) {
          const actionId: ActionIdentifier = `tool-${boundLetter}`;
          const tool = globalToolBinder.bindings.get(boundLetter)!;
          const label =
            tool.context instanceof UserLayer
              ? `${tool.description} — ${tool.context.managedLayer.name}`
              : tool.description;
          commands.push({
            kind: "action",
            label,
            shortcut: shortcutByAction.get(actionId) ?? "",
            actionId,
          });
        } else {
          const capturedToolJson = toolJson;
          commands.push({
            kind: "execute",
            label: getToolDescription(this.context, toolJson),
            shortcut: "",
            execute: () => activateUnboundTool(this.context, capturedToolJson),
          });
        }
      }
    }

    this.commands = commands;
    this.changed.dispatch();
  }

  filter(searchString: string): readonly CommandPaletteEntry[] {
    if (searchString === "") return this.commands;

    const query = searchString.toLowerCase();
    const prefixMatches: CommandPaletteEntry[] = [];
    const substringMatches: CommandPaletteEntry[] = [];

    for (const command of this.commands) {
      const label = command.label.toLowerCase();
      if (label.startsWith(query)) prefixMatches.push(command);
      else if (label.includes(query)) substringMatches.push(command);
    }

    return [...prefixMatches, ...substringMatches];
  }
}

export class CommandPalette extends Overlay {
  private readonly searchInput: HTMLInputElement;
  private readonly resultsList: HTMLElement;
  private readonly rowByCommand = new Map<CommandPaletteEntry, HTMLElement>();
  private readonly emptyElement: HTMLElement;
  private readonly pickerHeaderElement: HTMLElement;
  private filteredCommands: readonly CommandPaletteEntry[] = [];
  private filteredRows: HTMLElement[] = [];
  private activeIndex = 0;
  private currentCommands: readonly CommandPaletteEntry[];
  private readonly levelStack: {
    commands: readonly CommandPaletteEntry[];
    label: string;
  }[] = [];

  private readonly keyHandlers: Partial<
    Record<string, (event: KeyboardEvent) => void>
  > = {
    ArrowDown: (event) => {
      event.preventDefault();
      this.setActive(this.activeIndex + 1);
    },
    ArrowUp: (event) => {
      event.preventDefault();
      this.setActive(this.activeIndex - 1);
    },
    Enter: (event) => {
      event.preventDefault();
      event.stopPropagation();
      if (this.filteredCommands.length > 0)
        this.run(this.filteredCommands[this.activeIndex]);
    },
    Backspace: () => {
      if (this.levelStack.length > 0 && this.searchInput.value === "") {
        this.goBack();
      }
    },
    ArrowLeft: (event) => {
      if (
        this.levelStack.length > 0 &&
        this.searchInput.selectionStart === 0 &&
        this.searchInput.selectionEnd === 0
      ) {
        event.preventDefault();
        this.goBack();
      }
    },
    Escape: () => {
      if (this.levelStack.length > 0) {
        this.goBack();
      } else {
        this.closeAndRestoreFocus();
      }
    },
  };

  constructor(
    private readonly catalog: CommandCatalog,
    private readonly actionDispatchTarget: HTMLElement,
  ) {
    super();
    this.content.classList.add("neuroglancer-command-palette");

    this.currentCommands = this.catalog.commands;

    const pickerHeader = (this.pickerHeaderElement =
      document.createElement("div"));
    pickerHeader.className = "neuroglancer-command-palette-picker-header";
    pickerHeader.setAttribute("hidden", "");
    pickerHeader.addEventListener("click", () => this.goBack());

    const emptyElement = (this.emptyElement = document.createElement("div"));
    emptyElement.className = "neuroglancer-command-palette-empty";
    emptyElement.textContent = "No commands found.";

    const inputContainer = document.createElement("div");
    inputContainer.className = "neuroglancer-command-palette-input-row";
    inputContainer.appendChild(pickerHeader);
    const searchInput = (this.searchInput = document.createElement("input"));
    searchInput.type = "text";
    searchInput.className = "neuroglancer-command-palette-input";
    searchInput.placeholder = "Type a command...";
    searchInput.autocomplete = "off";
    searchInput.spellcheck = false;
    inputContainer.appendChild(searchInput);
    this.content.appendChild(inputContainer);

    const resultsList = (this.resultsList = document.createElement("div"));
    resultsList.className = "neuroglancer-command-palette-results";
    this.content.appendChild(resultsList);

    this.buildRows(this.catalog.commands);

    searchInput.addEventListener("input", () => {
      this.activeIndex = 0;
      this.render();
    });

    resultsList.addEventListener("mousedown", (event) =>
      event.preventDefault(),
    );

    this.content.addEventListener(
      "keydown",
      (event: KeyboardEvent) => this.keyHandlers[event.key]?.(event),
      { capture: true },
    );

    // Tools register keydown on window (bubble); stop propagation here after searchInput receives the event.
    this.content.addEventListener("keydown", (event) => {
      event.stopPropagation();
    });

    // The catalog may rebuild while this palette is open (a layer or tool
    // change, or an async lister resolving). Build rows for the new entries so
    // top-level filtering can be applied to the new entries.
    this.registerDisposer(
      this.catalog.changed.add(() => {
        this.buildRows(this.catalog.commands);
        if (this.levelStack.length === 0) {
          this.currentCommands = this.catalog.commands;
          this.render();
        }
      }),
    );

    this.render();
    searchInput.focus();
  }

  private buildRows(commands: readonly CommandPaletteEntry[]) {
    for (const command of commands) {
      if (this.rowByCommand.has(command)) continue;

      const commandRow = document.createElement("div");
      commandRow.className = "neuroglancer-command-palette-row";
      commandRow.addEventListener("click", () => this.run(command));

      const labelElement = document.createElement("span");
      labelElement.textContent = command.label;
      commandRow.appendChild(labelElement);

      if (command.shortcut) {
        const shortcutElement = document.createElement("span");
        shortcutElement.className = "neuroglancer-command-palette-shortcut";
        shortcutElement.textContent = command.shortcut;
        commandRow.appendChild(shortcutElement);
      }

      this.rowByCommand.set(command, commandRow);

      if (command.kind === "group") {
        this.buildRows(command.children);
      }
    }
  }

  private filterCurrentLevel(): readonly CommandPaletteEntry[] {
    if (this.levelStack.length === 0) {
      return this.catalog.filter(this.searchInput.value);
    }
    const query = this.searchInput.value.toLowerCase();
    if (query === "") return this.currentCommands;
    const prefixMatches: CommandPaletteEntry[] = [];
    const substringMatches: CommandPaletteEntry[] = [];
    for (const entry of this.currentCommands) {
      const label = entry.label.toLowerCase();
      if (label.startsWith(query)) prefixMatches.push(entry);
      else if (label.includes(query)) substringMatches.push(entry);
    }
    return [...prefixMatches, ...substringMatches];
  }

  private render() {
    this.filteredCommands = this.filterCurrentLevel();
    if (this.activeIndex >= this.filteredCommands.length) {
      this.activeIndex = Math.max(0, this.filteredCommands.length - 1);
    }

    if (this.filteredCommands.length === 0) {
      this.resultsList.replaceChildren(this.emptyElement);
      return;
    }

    this.filteredRows = this.filteredCommands.map(
      (command) => this.rowByCommand.get(command)!,
    );
    this.filteredRows.forEach((commandRow, rowIndex) => {
      commandRow.toggleAttribute("data-active", rowIndex === this.activeIndex);
    });
    this.resultsList.replaceChildren(...this.filteredRows);
  }

  private setActive(targetIndex: number) {
    if (this.filteredRows.length === 0) return;
    this.activeIndex =
      ((targetIndex % this.filteredRows.length) + this.filteredRows.length) %
      this.filteredRows.length;
    this.filteredRows.forEach((commandRow, rowIndex) => {
      commandRow.toggleAttribute("data-active", rowIndex === this.activeIndex);
      if (rowIndex === this.activeIndex)
        commandRow.scrollIntoView({ block: "nearest" });
    });
  }

  private updateHeader() {
    if (this.levelStack.length > 0) {
      this.pickerHeaderElement.textContent = `← ${this.levelStack.at(-1)!.label}`;
      this.pickerHeaderElement.removeAttribute("hidden");
    } else {
      this.pickerHeaderElement.setAttribute("hidden", "");
    }
  }

  private goBack() {
    if (this.levelStack.length === 0) {
      this.closeAndRestoreFocus();
      return;
    }
    const previous = this.levelStack.pop()!;
    this.currentCommands = previous.commands;
    this.searchInput.value = "";
    this.searchInput.placeholder = "Type a command...";
    this.updateHeader();
    this.activeIndex = 0;
    this.render();
  }

  // Non-toggle tools register a window bubble-phase keydown handler that
  // calls preventDefault() on all keys. Restoring focus to the viewer element
  // before the next keydown ensures F1 bubbles through the viewer's
  // KeyboardEventBinder and can reopen the palette.
  private closeAndRestoreFocus() {
    const target = this.actionDispatchTarget;
    this.close();
    target.focus({ preventScroll: true });
  }

  private run(command: CommandPaletteEntry) {
    if (command.kind === "group" && command.children.length > 0) {
      this.levelStack.push({
        commands: this.currentCommands,
        label: command.label,
      });
      this.currentCommands = command.children;
      this.searchInput.value = "";
      this.searchInput.placeholder = `Filter ${command.label}…`;
      this.updateHeader();
      this.activeIndex = 0;
      this.render();
      return;
    }

    this.closeAndRestoreFocus();

    if (command.kind === "execute") {
      command.execute();
    } else if (command.kind === "action") {
      this.actionDispatchTarget.dispatchEvent(
        new CustomEvent(`action:${command.actionId}`, {
          bubbles: true,
          cancelable: true,
          detail: {},
        }),
      );
    }
  }
}

/**
 * Binds the command palette to a viewer by handling the "open-command-palette"
 * action at the viewer element level, the same way every other global action
 * (e.g. "help") is bound. The action is dispatched by the configured
 * `control+keyp` binding in the viewer's input event map. This intentionally
 * does not install any document-level key listener, so the palette opens from
 * the main viewer UI but does not intercept keystrokes globally.
 */
export function bindCommandPalette(
  viewer: Viewer,
  catalog: CommandCatalog,
): void {
  let openPalette: CommandPalette | undefined;
  const openCommandPalette = () => {
    if (openPalette !== undefined && !openPalette.wasDisposed) return;
    const prevFocused = document.activeElement;
    // Tracking the dispatch target lets an activated command target the
    // specific element that had focus (e.g. the "snap" action in the panel the
    // user was in), falling back to the viewer element.
    const dispatchTarget =
      prevFocused instanceof HTMLElement && viewer.element.contains(prevFocused)
        ? prevFocused
        : viewer.element;
    openPalette = new CommandPalette(catalog, dispatchTarget);
  };
  viewer.bindAction("open-command-palette", openCommandPalette);
}
