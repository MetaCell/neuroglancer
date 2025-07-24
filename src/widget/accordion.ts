import { TrackableBoolean } from "#src/trackable_boolean.js";
import { WatchableValueInterface } from "#src/trackable_value.js";
import { Owned, RefCounted } from "#src/util/disposable.js";
import { parseArray, verifyOptionalObjectProperty } from "#src/util/json.js";
import { Signal } from "#src/util/signal.js";
import "#src/widget/accordion.css";
import { OptionSpecification, Tab } from "#src/widget/tab_view.js";
import { UserLayer } from "../layer";

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
  layer: UserLayer;
  expanded: WatchableValueInterface<boolean> = new TrackableBoolean(false);
  name: string = "";
  constructor(public sections: AccordionSectionStates) {
    super();
    this.layer = sections.layer;
  }
}

export class AccordionSectionStates {
  sections: AccordionSectionState[] = [];
  specificationChanged = new Signal();
  constructor(public layer: UserLayer) {
    this.sections = [layer.registerDisposer(new AccordionSectionState(this))];
  }

  restoreState(obj: unknown) {
    if (obj === undefined) {
      return;
    }
    verifyOptionalObjectProperty(obj, ACCORDION_JSON_KEY, (accordion) => {
      console.log("restoreState accordion", accordion);
      // TODO might be better to use keys established on the layer
      // For now, let's just assume we get an array of sections
      parseArray(accordion, (section) => {
        const { headerText, expanded } = section;
        const state = new AccordionSectionState(this);
        state.expanded.value = expanded ?? false;
        state.name = headerText;
        this.sections.push(state);
      });
      this.specificationChanged.dispatch();
    });
  }

  toJSON() {
    // TODO actually instead compare to default and only return changed sections
    console.log("toJSON accordion", this.sections);
    const sections = this.sections.map((section) => {
      return {
        headerText: section.name,
        expanded: section.expanded.value,
      };
    });
    return {
      [ACCORDION_JSON_KEY]: sections,
    };
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
        this.updateSections(),
      ),
    );
  }
  updateSections() {
    this.accordionTabState.sections.forEach((state) => {
      this.setSectionExpanded(state.name, state.expanded.value);
    });
  }
  /**
   * Set the section expanded state.
   * If expand is undefined, toggle the current state.
   * @param headerText The header text of the section to set.
   * @param expand Optional boolean to set the expanded state.
   */
  setSectionExpanded(headerText: string, expand?: boolean): void {
    const section = this.getSection(headerText);
    if (expand === undefined) {
      expand = section.container.dataset.expanded !== "true";
    }
    section.container.dataset.expanded = String(expand);
    section.header.setAttribute("aria-expanded", String(expand));
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
      this.setSectionExpanded.bind(this, headerText),
    );

    return newSection;
  }

  appendChild(content: HTMLElement, header: string = this.defaultHeader) {
    const section = this.getSection(header);
    section.body.appendChild(content);
  }
}
