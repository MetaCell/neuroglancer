import { Tab } from "#src/widget/tab_view.js";

interface Section {
  headerText: string;
  headerElement: HTMLElement;
  content: HTMLElement;
}

// TODO maybe should be just RefCounted, not sure yet
export class Accordion extends Tab {
  sections: Section[] = [];
  constructor(public defaultHeader = "Settings") {
    super();
    this.element.classList.add("neuroglancer-accordion");
  }
  makeSection(header: string): Section {
    const section = this.sections.find((e) => e.headerText === header);
    if (section !== undefined) {
      return section;
    }
    // TODO not quite exact logic here, just fast for now
    const newSection: Section = {
      headerText: header,
      headerElement: document.createElement("div"),
      content: document.createElement("div"),
    };
    newSection.content.classList.add("neuroglancer-accordion-content");
    this.sections.push(newSection);
    this.element.appendChild(newSection.headerElement);
    this.element.appendChild(newSection.content);
    newSection.headerElement.classList.add("neuroglancer-accordion-header");
    newSection.headerElement.textContent = header;
    newSection.headerElement.addEventListener("click", () => {
      // TODO css attribute
      const isOpen = newSection.content.style.display === "block";
      newSection.content.style.display = isOpen ? "none" : "block";
    });
    return newSection;
  }
  appendChild(content: HTMLElement, header: string = this.defaultHeader) {
    const section = this.makeSection(header);
    section.content.appendChild(content);
  }
}
