import { buildAnnotationsTab } from "#src/accordions/annotations.js";
import { buildRenderingTab } from "#src/accordions/rendering.js";
import { builSourceTab } from "#src/accordions/source.js";

export type TabId =
  | "source"
  | "rendering"
  | "annotations"
  | "segments"
  | "graph";

export function isTabId(id: string): id is TabId {
  return Object.keys(tabBuilder).includes(id);
}

type TabAccordionBuilder = (tabId: TabId, root: Element | null) => void;

export const tabBuilder: Record<TabId, TabAccordionBuilder> = {
  source: builSourceTab,
  rendering: buildRenderingTab,
  annotations: buildAnnotationsTab,
  segments: () => {
    console.warn("accordion for the segments tab is not implemented");
  },
  graph: () => {
    console.warn("accordion for the graph tab is not implemented");
  },
};

export function accordify(id: string, root: HTMLDivElement) {
  if (!isTabId(id)) {
    console.error("accordion builde: unsupported tabId:", id);
    return;
  }
  tabBuilder[id](id, root);
}
