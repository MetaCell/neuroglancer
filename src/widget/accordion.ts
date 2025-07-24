import { TrackableBoolean } from "#src/trackable_boolean.js";
import { WatchableValueInterface } from "#src/trackable_value.js";
import { RefCounted } from "#src/util/disposable.js";
import { parseArray } from "#src/util/json.js";
import { NullarySignal } from "#src/util/signal.js";
import "#src/widget/accordion.css";
import { Tab } from "#src/widget/tab_view.js";

export const ACCORDION_JSON_KEY = "accordion";

interface AccordionSectionOptions {
  jsonKey: string;
  displayName: string;
  defaultExpanded?: boolean;
  isDefaultKey?: boolean;
}

interface AccordionSection {
  name: string;
  jsonKey: string;
  container: HTMLElement;
  header: HTMLElement;
  body: HTMLElement;
}

// Modelled on the layer_side_panel_state.ts
export class AccordionSectionState extends RefCounted {
  isExpanded: WatchableValueInterface<boolean>;
  constructor(
    public parentState: AccordionSectionStates,
    public jsonKey: string,
    private defaultExpanded = false,
  ) {
    super();
    this.isExpanded = new TrackableBoolean(defaultExpanded, defaultExpanded);
    parentState.registerDisposer(this);
    this.registerDisposer(
      this.isExpanded.changed.add(() => {
        parentState.specificationChanged.dispatch();
      }),
    );
    parentState.sectionStates.push(this);
  }

  toJSON() {
    if (this.isExpanded.value === this.defaultExpanded) {
      return undefined;
    }
    return { [this.jsonKey]: this.isExpanded.value };
  }
}

export class AccordionSectionStates extends RefCounted {
  sectionStates: AccordionSectionState[] = [];
  specificationChanged = new NullarySignal();
  constructor() {
    super();
    this.sectionStates = [];
  }

  restoreState(obj: unknown) {
    if (obj === undefined) {
      return;
    }
    console.log("Restoring accordion state", obj);
    parseArray(obj, (section) => {
      const jsonKey = Object.keys(section)[0];
      const isExpanded = section[jsonKey];
      const newSection = new AccordionSectionState(this, jsonKey);
      newSection.isExpanded.value = isExpanded;
    });
    console.log("Restored accordion state", this.sectionStates);
    this.specificationChanged.dispatch();
  }

  setSectionExpanded(jsonKey: string, expand?: boolean): void {
    let section = this.sectionStates.find((s) => s.jsonKey === jsonKey);
    if (section === undefined) {
      section = new AccordionSectionState(this, jsonKey);
    }
    section.isExpanded.value = expand ?? !section.isExpanded.value;
  }

  toJSON() {
    const allSections = this.sectionStates.map((section) => section.toJSON());
    const filteredSections = allSections.filter(
      (section) => section !== undefined,
    );
    if (filteredSections.length === 0) {
      return undefined;
    }
    return filteredSections;
  }
}

export class AccordionTab extends Tab {
  sections: AccordionSection[] = [];
  defaultKey: string;
  constructor(
    protected accordionTabState: AccordionSectionStates,
    public options: AccordionSectionOptions[],
  ) {
    super();
    this.element.classList.add("neuroglancer-accordion");
    this.registerDisposer(
      this.accordionTabState.specificationChanged.add(() =>
        this.updateSectionsExpanded(),
      ),
    );
    options.forEach((option) => {
      this.makeNewSection(option);
    });
    if (this.defaultKey === undefined) {
      this.defaultKey = options[0].jsonKey;
    }
    this.updateSectionsExpanded();
  }

  private setSectionExpanded(jsonKey: string, expand?: boolean): void {
    this.accordionTabState.setSectionExpanded(jsonKey, expand);
  }

  private updateSectionsExpanded() {
    this.accordionTabState.sectionStates.forEach((state) => {
      const section = this.getSectionByKey(state.jsonKey);
      console.log(state.jsonKey, this.sections);
      if (section === undefined) {
        console.warn(
          `AccordionTab: No section found for key ${state.jsonKey}. This may be due to a mismatch in section names.`,
        );
        return;
      }
      section.container.dataset.expanded = String(state.isExpanded.value);
      section.header.setAttribute(
        "aria-expanded",
        String(state.isExpanded.value),
      );
    });
  }

  private makeNewSection(
    option: AccordionSectionOptions,
  ): AccordionSection | undefined {
    const newSection: AccordionSection = {
      name: option.displayName,
      jsonKey: option.jsonKey,
      container: document.createElement("div"),
      header: document.createElement("div"),
      body: document.createElement("div"),
    };
    this.sections.push(newSection);
    const { container, header, body } = newSection;
    container.classList.add("neuroglancer-accordion-item");
    body.classList.add("neuroglancer-accordion-body");
    header.classList.add("neuroglancer-accordion-header");
    container.appendChild(newSection.header);
    container.appendChild(newSection.body);
    this.element.appendChild(container);

    newSection.header.textContent = newSection.name;
    container.dataset.expanded = String(option.defaultExpanded ?? false);

    if (option.isDefaultKey) {
      this.defaultKey = option.jsonKey;
    }

    this.registerEventListener(newSection.header, "click", () =>
      this.setSectionExpanded(option.jsonKey),
    );
    return newSection;
  }

  getSectionByKey(jsonKey: string): AccordionSection | undefined {
    return this.sections.find((e) => e.jsonKey === jsonKey);
  }

  appendChild(content: HTMLElement, jsonKey?: string): void {
    const section =
      this.getSectionByKey(jsonKey ?? this.defaultKey) ??
      this.getSectionByKey(this.defaultKey);
    section!.body.appendChild(content);
  }
}
