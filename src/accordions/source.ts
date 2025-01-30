import { buildAccordion } from "#src/accordions/accordion.js";

export function builSourceTab(root: HTMLDivElement) {
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

    buildAccordion(dataSourcesControlsContainer, [
      {
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
        title: "Scale and translation",
        itemsClassNames: ["data-source-container", "transform-container"],
        selectors: {
          ids: ["transformWidget"],
        },
      },
    ]);

    buildAccordion(container, [
      {
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
