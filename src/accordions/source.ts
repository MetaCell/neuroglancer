import { buildAccordion } from "#src/accordions/build_accordion.js";
import type { TabId } from "#src/accordions/index.js";

export function builSourceTab(tabId: TabId, root: HTMLDivElement) {
  // there can be multiple data sources container
  const dataSourcesContainers = root.getElementsByClassName(
    "neuroglancer-layer-data-source",
  );
  if (dataSourcesContainers.length === 0) {
    return;
  }

  Array.from(dataSourcesContainers).forEach((container) => {
    const dataSourcesControlsContainer = container.querySelector(
      ".neuroglancer-layer-data-source-div",
    );

    buildAccordion(tabId, dataSourcesControlsContainer, [
      {
        id: "source-tab-enabled-components",
        title: "Enabled components",
        itemsClassNames: [
          "data-source-container",
          "enabled-components-container",
        ],
        selectors: {
          ids: ["enableDefaultSubsourcesLabel"],
          classNames: ["neuroglancer-layer-data-source-subsource"],
        },
      },
      {
        id: "source-tab-scale-and-translation",
        title: "Scale and translation",
        itemsClassNames: ["data-source-container", "transform-container"],
        selectors: {
          ids: ["transformWidget"],
        },
      },
    ]);

    buildAccordion(tabId, container, [
      {
        id: "source-tab-data-source",
        title: "Data source",
        open: true,
        selectors: {
          ids: ["dataSourceUrlInputElement"],
          classNames: ["data-source-container"],
        },
      },
    ]);
  });
}
