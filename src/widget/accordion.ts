import { TrackableBoolean } from "#src/trackable_boolean.js";
import { WatchableValueInterface } from "#src/trackable_value.js";
import { Owned, RefCounted } from "#src/util/disposable.js";
import { parseArray, verifyOptionalObjectProperty } from "#src/util/json.js";
import { NullarySignal } from "#src/util/signal.js";
import "#src/widget/accordion.css";
import { OptionSpecification, Tab } from "#src/widget/tab_view.js";

export const ACCORDION_JSON_KEY = "accordion";

export interface AccordionInfo {
  name: string;
  expanded: boolean;
}

interface Section {
  headerText: string;
  container: HTMLElement;
  header: HTMLElement;
  body: HTMLElement;
}

// Modelled on the layer_side_panel_state.ts
export class AccordionSectionState extends RefCounted {
  isExpanded: WatchableValueInterface<boolean>;
  constructor(
    public sections: AccordionSectionStates,
    public name: string = "",
    private defaultExpanded = false,
  ) {
    super();
    this.isExpanded = new TrackableBoolean(defaultExpanded, defaultExpanded);
    sections.registerDisposer(this);
    this.registerDisposer(
      this.isExpanded.changed.add(() => {
        sections.specificationChanged.dispatch();
      }),
    );
    sections.sections.push(this);
  }

  restoreState(obj: unknown) {
    if (obj === undefined) {
      return;
    }
    verifyOptionalObjectProperty(obj, "headerText", (headerText) => {
      this.name = headerText;
    });
    verifyOptionalObjectProperty(obj, "expanded", (expanded) => {
      if (typeof expanded === "boolean") {
        this.isExpanded.value = expanded;
      } else {
        console.warn(
          `AccordionSectionState: expected boolean for expanded, got ${expanded}`,
        );
      }
    });
  }

  // TODO either provide friendly keys or process name for key
  toJSON() {
    if (this.isExpanded.value === this.defaultExpanded) {
      return undefined;
    }
    return {
      headerText: this.name,
      expanded: this.isExpanded.value,
    };
  }
}

export class AccordionSectionStates extends RefCounted {
  sections: AccordionSectionState[] = [];
  specificationChanged = new NullarySignal();
  constructor() {
    super();
    this.sections = [];
  }

  restoreState(obj: unknown) {
    if (obj === undefined) {
      return;
    }
    verifyOptionalObjectProperty(obj, ACCORDION_JSON_KEY, (accordion) => {
      console.log("restoreState accordion", accordion);
      parseArray(accordion, (section) => {
        new AccordionSectionState(this).restoreState(section);
      });
      this.specificationChanged.dispatch();
    });
  }

  setSectionExpanded(headerText: string, expand?: boolean): void {
    let section = this.sections.find((s) => s.name === headerText);
    if (section === undefined) {
      section = new AccordionSectionState(this, headerText);
    }
    console.log(expand);
    section.isExpanded.value = expand ?? !section.isExpanded.value;
  }

  toJSON() {
    // Don't return undefined sections.
    const allSections = this.sections.map((section) => section.toJSON());
    const filteredSections = allSections.filter(
      (section) => section !== undefined,
    );
    if (filteredSections.length === 0) {
      return undefined;
    }
    return filteredSections;
  }
}

export class AccordionSpecification extends OptionSpecification<{
  label: string;
  order?: number;
  getter: () => Owned<AccordionTab>;
  expanded?: WatchableValueInterface<boolean>;
}> {}

export class AccordionTab extends Tab {
  sections: Section[] = [];
  constructor(
    protected accordionTabState: AccordionSectionStates,
    private defaultHeader = "Settings",
  ) {
    super();
    this.element.classList.add("neuroglancer-accordion");
    this.registerDisposer(
      this.accordionTabState.specificationChanged.add(() =>
        this.updateSectionsExpanded(),
      ),
    );
  }
  /**
   * Set the section expanded state.
   * If expand is undefined, toggle the current state.
   * @param headerText The header text of the section to set.
   * @param expand Optional boolean to set the expanded state.
   */
  setSectionExpanded(headerText: string, expand?: boolean): void {
    this.accordionTabState.setSectionExpanded(headerText, expand);
    console.log(this.accordionTabState);
  }

  updateSectionsExpanded() {
    this.accordionTabState.sections.forEach((state) => {
      const section = this.getSection(state.name);
      section.container.dataset.expanded = String(state.isExpanded.value);
      section.header.setAttribute(
        "aria-expanded",
        String(state.isExpanded.value),
      );
    });
  }

  getSection(headerText: string): Section {
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

    newSection.header.addEventListener(
      "click",
      () => this.setSectionExpanded(headerText)
    );

    return newSection;
  }

  appendChild(content: HTMLElement, header: string = this.defaultHeader) {
    header = header.trim();
    if (header === "") {
      header = this.defaultHeader;
    }
    console.log("appendChild to accordion section", header);
    const section = this.getSection(header);
    section.body.appendChild(content);
  }
}
