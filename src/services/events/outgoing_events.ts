import type { ScreenshotActionState } from "#src/python_integration/screenshots.js";
import { getCachedJson } from "#src/util/trackable.js";

export const STATE_UPDATE = "STATE_UPDATE" as const;
export const NEW_FIGURE = "NEW_FIGURE" as const;
export const OTHER = "OTHER" as const;

export type MessageType = typeof STATE_UPDATE | typeof NEW_FIGURE | typeof OTHER;


export interface SessionUpdatePayload {
    url: string;
    state: any;
}

export type DispatchablePayload = SessionUpdatePayload | ScreenshotActionState;


export function dispatchMessage(type: MessageType, payload: DispatchablePayload): void {
    window.postMessage({ type, payload }, "*");
}

export function getDeepClonedState(viewer: { state: any }): any {
    const cachedState = getCachedJson(viewer.state);
    return JSON.parse(JSON.stringify(cachedState?.value || {}));
}