import { getCachedJson } from "#src/util/trackable.js";

type MessageType = "STATE_UPDATE" | "OTHER";

export interface SessionUpdatePayload {
    url: string;
    state: any; // Adjust as needed for more specific typing
}

export function dispatchToParent(type: MessageType, payload: SessionUpdatePayload): void {
    if (window !== window.parent) {
        window.parent.postMessage({ type, payload }, "*");
    }
}

export function getDeepClonedState(viewer: { state: any }): any {
    const cachedState = getCachedJson(viewer.state);
    return JSON.parse(JSON.stringify(cachedState?.value || {}));
}