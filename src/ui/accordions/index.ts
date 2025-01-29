import { buildAnnotationsTab } from "#src/ui/accordions/annotations.js";
import { buildRenderingTab } from "#src/ui/accordions/rendering.js";
import { buildSegmentsTab } from "#src/ui/accordions/segments.js";
import { builSourceTab } from "#src/ui/accordions/source.js";

type TabId = "source" | "rendering" | "annotations" | "segments" | "graph";

export function isTabId(id: string): id is TabId {
  return Object.keys(tabBuilder).includes(id);
}

type TabAccordionBuilder = (root: HTMLDivElement) => void;

export const tabBuilder: Record<TabId, TabAccordionBuilder> = {
  source: builSourceTab,
  rendering: buildRenderingTab,
  annotations: buildAnnotationsTab,
  segments: buildSegmentsTab,
  graph: () => {
    console.error("accordion for the graph tab is not implemented");
  },
};

export function accordify(id: string, root: HTMLDivElement) {
  if (!isTabId(id)) {
    console.error("accordion builde: unsupported tabId:", id);
    return;
  }
  console.log("build tab: id=", id, root);
  tabBuilder[id](root);
}
