import type { TabId } from "#src/accordion_state.js";
import { buildAccordion } from "#src/accordions/build_accordion.js";
import { type AccordionOptions } from "#src/accordions/build_accordion.js";
import * as json_keys from "#src/layer/segmentation/json_keys.js";
import { LAYER_CONTROLS } from "#src/layer/segmentation/layer_controls.js";

// see: src/layer/image/index.rs
const renderingLayerImageTabSelectors: AccordionOptions[] = [
  {
    id: "rendering-tab-image-layer-slice-2d",
    title: "Slice 2D",
    itemsClassNames: ["slice-2d-container"],
    selectors: {
      ids: [
        "crossSectionRenderScale", // CROSS_SECTION_RENDER_SCALE_JSON_KEY
        "blend", // BLEND_JSON_KEY
        "opacity", // OPACITY_JSON_KEY
      ],
    },
  },
  {
    id: "rendering-tab-image-layer-volume-rendering",
    title: "Volume Rendering",
    itemsClassNames: ["volume-rendering-container"],
    selectors: {
      ids: [
        "volumeRendering", // VOLUME_RENDERING_JSON_KEY
        "volumeRenderingGain", // VOLUME_RENDERING_GAIN_JSON_KEY
        "volumeRenderingDepthSamples", // VOLUME_RENDERING_DEPTH_SAMPLES_JSON_KEY
      ],
    },
  },
  {
    id: "rendering-tab-image-layer-shader",
    title: "Shader",
    open: true,
    itemsClassNames: ["channels-container", "shader"],
    selectors: {
      ids: [
        "shader", // SHADER_JSON_KEY
        "shaderControls", // SHADER_CONTROLS_JSON_KEY
      ],
    },
  },
] as const;

// builds the accordions for the render tab when the layer
// is an image layer.
export function buildRenderingLayerImageTab(tabId: TabId, root: Element) {
  buildAccordion(tabId, root, renderingLayerImageTabSelectors);
}

const renderingLayerSegTabSelectors: AccordionOptions[] = [
  {
    id: "rendering-tab-seg-layer-visibility",
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
    id: "rendering-tab-seg-layer-appearance",
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
    id: "rendering-tab-seg-layer-slice-2d",
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
    id: "rendering-tab-seg-layer-mesh-3d",
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
    id: "rendering-tab-seg-layer-channels",
    title: "Channels",
    itemsClassNames: ["shader-container"],
    selectors: {
      ids: [], // TODO: what should be here?
    },
  },
  {
    id: "rendering-tab-seg-layer-skeletons",
    title: "Skeletons",
    itemsClassNames: ["skeleton-container"],
    selectors: {
      ids: [
        ...(LAYER_CONTROLS.filter((c) =>
          String(c.toolJson).startsWith(json_keys.SKELETON_RENDERING_JSON_KEY),
        ).map((c) => c.toolJson) as string[]),
        json_keys.SKELETON_RENDERING_SHADER_CONTROL_TOOL_ID,
      ],
    },
  },
];

// builds the accordions for the render tab when the layer
// is a segmentation layer.
export function buildRenderingLayerSegTab(tabId: TabId, root: Element) {
  buildAccordion(tabId, root, renderingLayerSegTabSelectors);
}

export function buildRenderingTab(tabId: TabId, root: Element) {
  // infer layer type based on root classname
  if (root.classList.contains("neuroglancer-segmentation-rendering-tab")) {
    buildRenderingLayerSegTab(tabId, root);
    return;
  }
  if (root.classList.contains("neuroglancer-annotation-rendering-tab")) {
    console.warn(
      "no accordion definition for the rendering tab in an annotation layer",
    );
    return;
  }

  buildRenderingLayerImageTab(tabId, root);
}
