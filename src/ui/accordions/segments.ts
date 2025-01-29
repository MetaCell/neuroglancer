import * as json_keys from "#src/layer/segmentation/json_keys.js";
import { LAYER_CONTROLS } from "#src/layer/segmentation/layer_controls.js";
import { buildAccordion } from "#src/ui/accordions/accordion.js";
import { type AccordionOptions } from "#src/ui/accordions/accordion.js";

const segmentsTabSelectors: AccordionOptions[] = [
  {
    title: "Visibility",
    itemsClassNames: ["visibility-container"],
    selectors: {
      ids: [
        json_keys.HIDE_SEGMENT_ZERO_JSON_KEY,
        json_keys.IGNORE_NULL_VISIBLE_SET_JSON_KEY,
        "linkedSegmentationGroup",
      ],
    },
  },
  {
    title: "Appearance",
    itemsClassNames: ["appearance-container"],
    selectors: {
      ids: [
        json_keys.SEGMENT_DEFAULT_COLOR_JSON_KEY,
        json_keys.COLOR_SEED_JSON_KEY,
        json_keys.SATURATION_JSON_KEY,
        json_keys.BASE_SEGMENT_COLORING_JSON_KEY,
        json_keys.HOVER_HIGHLIGHT_JSON_KEY,
        "linkedSegmentationColorGroup",
      ],
    },
  },
  {
    title: "Slice 2D",
    itemsClassNames: ["slice2d-container"],
    selectors: {
      ids: [
        json_keys.SELECTED_ALPHA_JSON_KEY,
        json_keys.NOT_SELECTED_ALPHA_JSON_KEY,
        json_keys.CROSS_SECTION_RENDER_SCALE_JSON_KEY,
      ],
    },
  },
  {
    title: "Mesh 3D",
    itemsClassNames: ["mesh3d-container"],
    selectors: {
      ids: [
        json_keys.MESH_RENDER_SCALE_JSON_KEY,
        json_keys.OBJECT_ALPHA_JSON_KEY,
        json_keys.MESH_SILHOUETTE_RENDERING_JSON_KEY,
      ],
    },
  },
  {
    title: "Channels",
    itemsClassNames: ["shader-container"],
    selectors: {
      ids: [json_keys.SKELETON_RENDERING_SHADER_CONTROL_TOOL_ID],
    },
  },
  {
    title: "Skeletons",
    itemsClassNames: ["skeleton-container"],
    selectors: {
      ids: LAYER_CONTROLS.filter((c) =>
        String(c.toolJson).startsWith(json_keys.SKELETON_RENDERING_JSON_KEY),
      ).map((c) => c.toolJson) as string[],
    },
  },
];

export function buildSegmentsTab(root: HTMLDivElement) {
  buildAccordion(root, segmentsTabSelectors);
}
