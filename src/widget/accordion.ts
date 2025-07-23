import "#src/widget/accordion.css";
import { Tab } from "#src/widget/tab_view.js";

interface Section {
  headerText: string;
  container: HTMLElement;
  header: HTMLElement;
  body: HTMLElement;
}

export class AccordionTab extends Tab {
  sections: Section[] = [];
  constructor(public defaultHeader = "Settings") {
    super();
    this.element.classList.add("neuroglancer-accordion");
  }
  makeSection(headerText: string): Section {
    const { sections } = this;
    const section = sections.find((e) => e.headerText === headerText);
    if (section !== undefined) {
      return section;
    }
    const newSection: Section = {
      headerText: headerText,
      container: document.createElement("div"),
      header: document.createElement("div"),
      body: document.createElement("div"),
    };
    sections.push(newSection);

    const { container, header, body } = newSection;
    container.classList.add("neuroglancer-accordion-item");
    body.classList.add("neuroglancer-accordion-body");
    header.classList.add("neuroglancer-accordion-header");
    container.appendChild(newSection.header);
    container.appendChild(newSection.body);
    this.element.appendChild(container);

    newSection.header.textContent = headerText;
    container.dataset.expanded = "false";

    const toggleExpanded = () => {
      const expanded = container.dataset.expanded === "true";
      container.dataset.expanded = expanded ? "false" : "true";
      newSection.header.setAttribute(
        "aria-expanded",
        !expanded ? "true" : "false",
      );
    };
    newSection.header.addEventListener("click", toggleExpanded);

    return newSection;
  }

  appendChild(content: HTMLElement, header: string = this.defaultHeader) {
    const section = this.makeSection(header);
    section.body.appendChild(content);
  }
}
