import { PROJECTION_RENDER_SCALE_JSON_KEY } from "#src/layer/annotation/index.js";
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

interface AccordionItemSelector {
  title: string;

  // Classnames to be add to the
  // accordion item.
  classNames: string[];

  // Ids used to select elements to
  // build an accordion item.
  selectIds?: string[];

  // ClassNames used to select elements to
  // build an accordion item.
  selectClassNames?: string[];
}

function buildAccordion(
  root: Element,
  selectors: Record<any, AccordionItemSelector>,
) {
  const categoryElements = new Map<string, Element[]>();
  Object.keys(selectors).forEach((category) => {
    categoryElements.set(category, []);
  });

  // categorize children so we can understand how to group them
  Array.from(root.children).forEach((child) => {
    const matchIds = Object.entries(selectors).find(([_, data]) =>
      data.selectIds?.includes(child.id),
    )?.[0];

    const matchClasses = Object.entries(selectors).find(([_, data]) =>
      data.selectClassNames?.some((className) =>
        child.classList.contains(className),
      ),
    )?.[0];

    const category = matchIds || matchClasses;
    if (category) {
      categoryElements.get(category)?.push(child);
    }
  });

  const accordionItems: AccordionItem[] = [];

  categoryElements.forEach((elements, category) => {
    if (elements.length === 0) {
      return;
    }

    const containerDiv = document.createElement("div");
    selectors[category].classNames.forEach((className) => {
      containerDiv.classList.add(className);
    });

    elements.forEach((element) => {
      containerDiv.appendChild(element);
    });

    accordionItems.push({
      title: selectors[category].title,
      content: containerDiv,
    });
  });

  if (accordionItems.length === 0) {
    return;
  }

  const accordion = new Accordion(accordionItems);
  root.appendChild(accordion.getElement());
}

const LAYER_RENDERING_ACCORDION_SELECTOR: Record<
  string,
  AccordionItemSelector
> = {
  slice2D: {
    title: "Slice 2D",
    classNames: ["slice-2d-container"],
    selectIds: [
      CROSS_SECTION_RENDER_SCALE_JSON_KEY,
      BLEND_JSON_KEY,
      OPACITY_JSON_KEY,
    ],
  },
  volume: {
    title: "Volume Rendering",
    classNames: ["volume-rendering-container"],
    selectIds: [
      VOLUME_RENDERING_JSON_KEY,
      VOLUME_RENDERING_GAIN_JSON_KEY,
      VOLUME_RENDERING_DEPTH_SAMPLES_JSON_KEY,
    ],
  },
  shader: {
    title: "Shader",
    classNames: ["channels-container", "shader"],
    selectIds: [SHADER_JSON_KEY, SHADER_CONTROLS_JSON_KEY],
  },
} as const;

// Builds the accordion for the layer side panel rendering tab.
function buildLayerRenderingAccordion(root: HTMLElement) {
  const dropdowns = root.getElementsByClassName("neuroglancer-image-dropdown");
  if (dropdowns.length === 0) {
    return;
  }

  Array.from(dropdowns).forEach((dropdown) => {
    buildAccordion(dropdown, LAYER_RENDERING_ACCORDION_SELECTOR);
  });
}

const ANNOTATIONS_USER_LAYER_ACCORDION_SELECTOR: Record<
  string,
  AccordionItemSelector
> = {
  projections: {
    title: "Spacing",
    classNames: ["projections-container"],
    selectIds: [PROJECTION_RENDER_SCALE_JSON_KEY],
  },
  segmentFiltering: {
    title: "Segment Filtering",
    classNames: ["segment-filtering-container"],
    selectIds: ["segmentFilterLabel", "segmentationLayersWidget"],
  },
} as const;

function buildAnnotationsUserLayerAccordion(root: HTMLElement) {
  const accordions = root.getElementsByClassName(
    "neuroglancer-annotation-layer-view",
  );
  if (accordions.length === 0) {
    return;
  }

  Array.from(accordions).forEach((accordion) => {
    buildAccordion(accordion, ANNOTATIONS_USER_LAYER_ACCORDION_SELECTOR);
  });
}

const DATA_SOURCES_ACCORDION_SELECTOR: Record<string, AccordionItemSelector> = {
  enabledComponents: {
    title: "Enabled components",
    classNames: ["data-source-container", "enabled-components-container"],
    selectIds: ["enableDefaultSubsourcesLabel"],
    selectClassNames: ["neuroglancer-layer-data-source-subsource"],
  },
  scaleAndTranslation: {
    title: "Scale and translation",
    classNames: ["data-source-container", "transform-container"],
    selectIds: ["transformWidget"],
  },
};

function buildDataSourcesAccordion(root: HTMLElement) {
  const dataSourceAccordion = root.getElementsByClassName(
    "neuroglancer-layer-data-source",
  );
  if (dataSourceAccordion.length === 0) {
    return;
  }

  Array.from(dataSourceAccordion).forEach((accordion) => {
    buildAccordion(accordion, {
      dataSource: {
        title: "Data source",
        classNames: ["data-source-container"],
        selectIds: ["dataSourceUrlInputElement"],
      },
    });
  });

  const accordions = root.getElementsByClassName(
    "neuroglancer-layer-data-source-div",
  );
  if (accordions.length === 0) {
    return;
  }

  Array.from(accordions).forEach((accordion) => {
    buildAccordion(accordion, DATA_SOURCES_ACCORDION_SELECTOR);
  });
}

const ANNOTATIONS_ACCORDION_SELECTOR: Record<string, AccordionItemSelector> = {
  annotations: {
    title: "Annotations",
    classNames: ["annotations-toolbox-container"],
    selectClassNames: ["neuroglancer-annotation-layer-view"],
  },
};

function buildAnnotationAccordion(root: HTMLElement) {
  const accordions = root.getElementsByClassName(
    "neuroglancer-annotations-tab",
  );
  if (accordions.length === 0) {
    return;
  }

  Array.from(accordions).forEach((accordion) => {
    buildAccordion(accordion, ANNOTATIONS_ACCORDION_SELECTOR);
  });
}

export function buildAccordions(root: HTMLElement) {
  buildLayerRenderingAccordion(root);
  buildAnnotationAccordion(root);
  buildAnnotationsUserLayerAccordion(root);
  buildDataSourcesAccordion(root);
}
