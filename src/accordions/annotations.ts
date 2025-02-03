import { buildAccordion } from "#src/accordions/build_accordion.js";
import type { TabId } from "#src/accordions/index.js";

export function buildAnnotationsTab(tabId: TabId, root: HTMLDivElement) {
  buildAccordion(tabId, root, [
    {
      id: "annotations-tab-annotations",
      title: "Annotations",
      open: true,
      itemsClassNames: ["annotations-toolbox-container"],
      selectors: {
        classNames: ["neuroglancer-annotation-layer-view"],
      },
    },
    {
      id: "annotations-tab-spacing",
      title: "Spacing",
      itemsClassNames: ["projections-container"],
      selectors: {
        ids: ["projectionAnnotationSpacing"], // PROJECTION_RENDER_SCALE_JSON_KEY
      },
    },
    {
      id: "annotations-tab-segment-filtering",
      title: "Segment Filtering",
      itemsClassNames: ["segment-filtering-container"],
      selectors: {
        ids: ["segmentFilterLabel", "segmentationLayersWidget"],
      },
    },
  ]);
}
