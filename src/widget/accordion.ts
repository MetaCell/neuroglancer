import { TrackableBoolean } from "#src/trackable_boolean.js";
import { WatchableValueInterface } from "#src/trackable_value.js";
import { RefCounted } from "#src/util/disposable.js";
import { NullarySignal } from "#src/util/signal.js";
import "#src/widget/accordion.css";
import { Tab } from "#src/widget/tab_view.js";

export interface AccordionOptions {
  accordionJsonKey: string;
  sections: AccordionSectionOptions[];
}

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

export class AccordionSectionState extends RefCounted {
  isExpanded: WatchableValueInterface<boolean>;
  constructor(
    public parentState: AccordionState,
    public jsonKey: string,
    private defaultExpanded = false,
  ) {
    super();
    this.isExpanded = new TrackableBoolean(defaultExpanded, defaultExpanded);
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

export class AccordionState extends RefCounted {
  sectionStates: AccordionSectionState[] = [];
  specificationChanged = new NullarySignal();
  constructor(public accordionOptions: AccordionOptions) {
    super();
    for (const sectionOptions of accordionOptions.sections) {
      this.registerSection(sectionOptions);
    }
  }

  restoreState(obj: unknown) {
    if (obj === undefined || obj === null || typeof obj !== "object") {
      return;
    }
    for (const [jsonKey, isExpanded] of Object.entries(obj)) {
      let existingSection = this.sectionStates.find(
        (s) => s.jsonKey === jsonKey,
      );
      if (existingSection === undefined) {
        existingSection = new AccordionSectionState(this, jsonKey);
      }
      existingSection.isExpanded.value = isExpanded;
      this.specificationChanged.dispatch();
    }
  }

  registerSection(sectionOptions: AccordionSectionOptions) {
    const existingSection = this.sectionStates.find(
      (s) => s.jsonKey === sectionOptions.jsonKey,
    );
    if (existingSection !== undefined) return;
    const newSection = this.registerDisposer(
      new AccordionSectionState(
        this,
        sectionOptions.jsonKey,
        sectionOptions.defaultExpanded,
      ),
    );
    this.sectionStates.push(newSection);
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
      return {};
    }
    const mergedSections = Object.assign({}, ...filteredSections);
    return mergedSections;
  }
}

export class AccordionTab extends Tab {
  sections: AccordionSection[] = [];
  defaultKey: string;
  constructor(protected accordionState: AccordionState) {
    super();
    const options = accordionState.accordionOptions;
    this.element.classList.add("neuroglancer-accordion");
    this.registerDisposer(
      this.accordionState.specificationChanged.add(() =>
        this.updateSectionsExpanded(),
      ),
    );
    options.sections.forEach((option) => {
      this.makeNewSection(option);
    });
    if (this.defaultKey === undefined && options.sections.length > 0) {
      this.defaultKey = options.sections[0].jsonKey;
    }
    this.updateSectionsExpanded();
  }

  private setSectionExpanded(jsonKey: string, expand?: boolean): void {
    this.accordionState.setSectionExpanded(jsonKey, expand);
  }

  private updateSectionsExpanded() {
    this.accordionState.sectionStates.forEach((state) => {
      const section = this.getSectionByKey(state.jsonKey);
      if (section === undefined) {
        console.warn(`AccordionTab: No section found for key ${state.jsonKey}`);
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
    container.style.display = "none";
    container.dataset.expanded = String(option.defaultExpanded ?? false);

    if (option.isDefaultKey) {
      this.defaultKey = option.jsonKey;
    }

    this.registerEventListener(newSection.header, "click", () =>
      this.setSectionExpanded(option.jsonKey),
    );

    this.accordionState.registerSection(option);
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
    section!.container.style.display = "block";
  }
}
