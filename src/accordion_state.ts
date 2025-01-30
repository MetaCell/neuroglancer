import { TrackableBoolean } from "#src/trackable_boolean.js";
import { RefCounted } from "#src/util/disposable.js";
import { NullarySignal } from "#src/util/signal.js";
import { type Trackable } from "#src/util/trackable.js";

export type TabId = 'source' | 'rendering' | 'annotations' | 'segments' | 'graph';

export interface TabAccordionState {
  [tabId: string]: {
    [accordionId: string]: TrackableBoolean;
  }
}

export class TrackableTabAccordionState extends RefCounted implements Trackable {
  changed = new NullarySignal();
  private states: TabAccordionState = {};

  getState(tabId: TabId, accordionId: string): TrackableBoolean {
    if (!this.states[tabId]) {
      this.states[tabId] = {};
    }

    if (!this.states[tabId][accordionId]) {
      const state = new TrackableBoolean(false);
      this.states[tabId][accordionId] = state;
      this.registerDisposer(state.changed.add(() => this.changed.dispatch()));
    }

    return this.states[tabId][accordionId];
  }

  restoreState(obj: any) {
    if (obj && typeof obj === 'object') {
      for (const [tabId, tabStates] of Object.entries(obj)) {
        if (typeof tabStates === 'object' && tabStates !== null) {
          for (const [accordionId, value] of Object.entries(tabStates)) {
            this.getState(tabId as TabId, accordionId).restoreState(value);
          }
        }
      }
    }
  }

  toJSON() {
    const result: {[tabId: string]: {[accordionId: string]: boolean}} = {};
    for (const [tabId, tabStates] of Object.entries(this.states)) {
      const tabResult: {[accordionId: string]: boolean} = {};
      for (const [accordionId, state] of Object.entries(tabStates)) {
        tabResult[accordionId] = state.value;
      }
      if (Object.keys(tabResult).length > 0) {
        result[tabId] = tabResult;
      }
    }
    return Object.keys(result).length > 0 ? result : undefined;
  }

  reset() {
    for (const tabStates of Object.values(this.states)) {
      for (const state of Object.values(tabStates)) {
        state.reset();
      }
    }
  }
}
