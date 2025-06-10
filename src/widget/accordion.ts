import "#src/widget/accordion.css";

import { removeChildren } from "#src/util/dom.js";

export function makeAccordionHeader(text: string) {
  const header = document.createElement("div");
  header.textContent = text;
  header.classList.add("neuroglancer-accordion-header");
  return header;
}

export function applyAccordion(
  element: HTMLElement,
  headerClass = "neuroglancer-accordion-header",
) {
  const children = Array.from(element.children);
  const accordion = document.createElement("div");
  accordion.classList.add("neuroglancer-accordion");
  let currentSection: HTMLDetailsElement | null = null;
  let body: HTMLElement | null = null;
  for (const child of children) {
    if (child.classList.contains(headerClass)) {
      currentSection = document.createElement("details");
      currentSection.classList.add("neuroglancer-accordion-section");
      const summary = document.createElement("summary");
      summary.classList.add(headerClass);
      summary.appendChild(child);
      currentSection.appendChild(summary);
      body = document.createElement("div");
      body.classList.add("neuroglancer-accordion-section-body");
      currentSection.appendChild(body);
      accordion.appendChild(currentSection);
    } else if (currentSection !== null) {
      body!.appendChild(child);
    } else {
      accordion.appendChild(child);
    }
  }
  removeChildren(element);
  element.appendChild(accordion);
}
