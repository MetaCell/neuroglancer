import type { TrackableTabAccordionState } from "#src/accordion_state.js";
import type { TabId } from "#src/accordions/index.js";
import { Accordion } from "#src/widget/accordion.js";
import { type AccordionItem } from "#src/widget/accordion.js";

export interface AccordionOptions {
  id: string;
  title: string;
  open?: boolean;

  // Classnames to be add to the accordion items
  itemsClassNames?: string[];
  selectors?: {
    // Ids used to select elements to build an accordion item
    ids?: string[];
    // ClassNames used to select elements to build an accordion item
    classNames?: string[];
  };
}

// NOTE: don't love doing this here but there is not fair
// way to pass viewer to buildAccordion
declare global {
  interface Window {
    viewer?: { tabAccordionState?: TrackableTabAccordionState };
  }
}

export function buildAccordion(
  tabId: TabId,
  root: Element | null,
  opts: AccordionOptions[],
) {
  if (!root) {
    return;
  }

  const categoryElements = new Map<number, Element[]>();

  opts.forEach((_, index) => {
    categoryElements.set(index, []);
  });

  Array.from(root.children).forEach((child) => {
    const categoryIndex = opts.findIndex(
      (opt) =>
        opt.selectors?.ids?.includes(child.id) ||
        opt.selectors?.classNames?.some((className) =>
          child.classList.contains(className),
        ),
    );

    if (categoryIndex !== -1) {
      categoryElements.get(categoryIndex)?.push(child);
    }
  });

  const accordionItems: AccordionItem[] = [];

  categoryElements.forEach((elements, index) => {
    if (elements.length === 0) return;

    const containerDiv = document.createElement("div");
    opts[index].itemsClassNames?.forEach((className) => {
      containerDiv.classList.add(className);
    });

    elements.forEach((element) => {
      containerDiv.appendChild(element);
    });

    // if all the elements are empty, the accordion
    // item should not be rendered
    if (!hasContent(containerDiv)) {
      return;
    }

    const item: AccordionItem = {
      title: opts[index].title,
      content: containerDiv,
    };

    if (window.viewer?.tabAccordionState && opts[index].id) {
      item.state = window.viewer.tabAccordionState.getState(
        tabId,
        opts[index].id,
      );
    } else {
      item.open = opts[index].open;
    }

    accordionItems.push(item);
  });

  if (accordionItems.length === 0) return;

  const accordion = new Accordion(accordionItems);
  root.appendChild(accordion.getElement());
}

function hasContent(element: Element): boolean {
  if (!element.hasChildNodes()) {
    return false;
  }
  return Array.from(element.children).some((child) => child.hasChildNodes());
}
