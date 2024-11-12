import {
  BLEND_JSON_KEY,
  CROSS_SECTION_RENDER_SCALE_JSON_KEY,
  OPACITY_JSON_KEY,
  SHADER_CONTROLS_JSON_KEY,
  SHADER_JSON_KEY,
  VOLUME_RENDERING_DEPTH_SAMPLES_JSON_KEY,
  VOLUME_RENDERING_GAIN_JSON_KEY,
  VOLUME_RENDERING_JSON_KEY,
} from "#src/layer/image/index.js";
import { Accordion } from "#src/widget/accordion.js";
import { type AccordionItem } from "#src/widget/accordion.js";

const CONTROL_CATEGORIES = {
  slice2D: {
    title: "Slice 2D",
    className: ["slice-2d-container"],
    controls: [
      CROSS_SECTION_RENDER_SCALE_JSON_KEY,
      BLEND_JSON_KEY,
      OPACITY_JSON_KEY,
    ] as string[],
  },
  volume: {
    title: "Volume Rendering",
    className: ["volume-rendering-container"],
    controls: [
      VOLUME_RENDERING_JSON_KEY,
      VOLUME_RENDERING_GAIN_JSON_KEY,
      VOLUME_RENDERING_DEPTH_SAMPLES_JSON_KEY,
    ] as string[],
  },
  shader: {
    title: "Shader",
    className: ["channels-container", "shader"],
    controls: [SHADER_JSON_KEY, SHADER_CONTROLS_JSON_KEY] as string[],
  },
} as const;

type ControlCategory = keyof typeof CONTROL_CATEGORIES;

function buildAccordion(root: Element) {
  const categoryElements = new Map<ControlCategory, Element[]>();
  Object.keys(CONTROL_CATEGORIES).forEach((category) => {
    categoryElements.set(category as ControlCategory, []);
  });

  // categorize children so we can understand how to group them
  Array.from(root.children).forEach((child) => {
    const controlId = child.id;
    if (!controlId) {
      return;
    }

    const categoryMatch = Object.entries(CONTROL_CATEGORIES).find(
      ([_, categoryData]) => categoryData.controls.includes(controlId),
    )?.[0] as ControlCategory | undefined;

    if (categoryMatch) {
      categoryElements.get(categoryMatch)?.push(child);
    }
  });

  const accordionItems: AccordionItem[] = [];

  categoryElements.forEach((elements, category) => {
    if (elements.length === 0) {
      return;
    }

    const containerDiv = document.createElement("div");
    CONTROL_CATEGORIES[category].className.forEach((className) => {
      containerDiv.classList.add(className);
    });

    elements.forEach((element) => {
      containerDiv.appendChild(element);
    });

    accordionItems.push({
      title: CONTROL_CATEGORIES[category].title,
      content: containerDiv,
    });
  });

  if (accordionItems.length === 0) {
    return;
  }

  const accordion = new Accordion(accordionItems);
  root.appendChild(accordion.getElement());
}

export function buildAccordions(root: HTMLElement) {
  const dropdowns = root.getElementsByClassName("neuroglancer-image-dropdown");
  if (dropdowns.length === 0) {
    return;
  }

  Array.from(dropdowns).forEach((dropdown) => {
    buildAccordion(dropdown);
  });
}
