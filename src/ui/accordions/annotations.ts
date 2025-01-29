import { buildAccordion } from "#src/ui/accordions/accordion.js";

export function buildAnnotationsTab(root: HTMLDivElement) {
  buildAccordion(root, [
    {
      title: "Annotations",
      open: true,
      itemsClassNames: ["annotations-toolbox-container"],
      selectors: {
        classNames: ["neuroglancer-annotation-layer-view"],
      },
    },
    {
      title: "Spacing",
      itemsClassNames: ["projections-container"],
      selectors: {
        ids: ["projectionAnnotationSpacing"], // PROJECTION_RENDER_SCALE_JSON_KEY
      },
    },
    {
      title: "Segment Filtering",
      itemsClassNames: ["segment-filtering-container"],
      selectors: {
        ids: ["segmentFilterLabel", "segmentationLayersWidget"],
      },
    },
  ]);
}
