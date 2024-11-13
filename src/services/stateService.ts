import { getCachedJson } from "#src/util/trackable.js";

export const STATE_UPDATE = "STATE_UPDATE" as const;
export const NEW_FIGURE = "NEW_FIGURE" as const;
export const OTHER = "OTHER" as const;

export type MessageType = typeof STATE_UPDATE | typeof NEW_FIGURE | typeof OTHER;

export interface ViewerState {
    dimensions: Record<string, [number, string]>;
    position: [number, number, number];
    crossSectionScale: number;
    projectionScale: number;
    layers: Layer[];
    selectedLayer: SelectedLayer;
    layout: string;
  }
  
  export interface Layer {
    type: string;
    source: string;
    tab: string;
    name: string;
  }
  
  export interface SelectedLayer {
    visible: boolean;
    layer: string;
  }
  
  export interface Screenshot {
    id: string;
    image: string;  // base64 encoded image
    imageType: string;
    width: number;
    height: number;
  }
  
  export interface NewFigurePayload {
    viewerState: ViewerState;
    selectedValues: Record<string, any>;  
    screenshot: Screenshot;
  }
    
  export interface SessionUpdatePayload {
    url: string;
    state: any; // Adjust as needed for more specific typing
}

export type DispatchablePayload = SessionUpdatePayload | NewFigurePayload;


export function dispatchMessage(type: MessageType, payload: DispatchablePayload): void {
    window.postMessage({ type, payload }, "*");
}

export function getDeepClonedState(viewer: { state: any }): any {
    const cachedState = getCachedJson(viewer.state);
    return JSON.parse(JSON.stringify(cachedState?.value || {}));
}