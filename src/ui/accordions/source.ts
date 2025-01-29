import { buildAccordion } from "#src/ui/accordions/accordion.js";

export function builSourceTab(root: HTMLDivElement) {
  const dataSourcesControlsContainer = root?.querySelector(
    ".neuroglancer-layer-data-source-div",
  );
  const dataSourcesContainer = root.querySelector(
    ".neuroglancer-layer-data-source",
  );

  buildAccordion(dataSourcesContainer, [
    {
      title: "Data source",
      open: true,
      selectors: {
        ids: ["dataSourceUrlInputElement"],
        classNames: ["data-source-container"],
      },
    },
  ]);

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
}
