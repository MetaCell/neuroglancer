import { Accordion } from "#src/widget/accordion.js";
import { type AccordionItem } from "#src/widget/accordion.js";

export interface AccordionOptions {
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

export function buildAccordion(root: Element | null, opts: AccordionOptions[]) {
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

    accordionItems.push({
      title: opts[index].title,
      content: containerDiv,
      open: opts[index].open,
    });
  });

  if (accordionItems.length === 0) return;

  const accordion = new Accordion(accordionItems);
  root.appendChild(accordion.getElement());
}
