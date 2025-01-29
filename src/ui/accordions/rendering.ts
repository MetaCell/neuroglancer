import { buildAccordion } from "#src/ui/accordions/accordion.js";
import { type AccordionOptions } from "#src/ui/accordions/accordion.js";

// see: src/layer/image/index.rs
const renderingTabSelectors: AccordionOptions[] = [
  {
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

export function buildRenderingTab(root: HTMLDivElement) {
  buildAccordion(root, renderingTabSelectors);
}
