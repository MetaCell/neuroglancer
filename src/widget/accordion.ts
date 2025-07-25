import { TrackableBoolean } from "#src/trackable_boolean.js";
import { WatchableValueInterface } from "#src/trackable_value.js";
import { RefCounted } from "#src/util/disposable.js";
import { NullarySignal } from "#src/util/signal.js";
import "#src/widget/accordion.css";
import { Tab } from "#src/widget/tab_view.js";
import svg_chevron_up from "ikonate/icons/chevron-up.svg?raw";

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
    if (this.isExpanded.value === this.defaultExpanded) return undefined;
    return { [this.jsonKey]: this.isExpanded.value };
  }
}

export class AccordionState extends RefCounted {
  sectionStates: AccordionSectionState[] = [];
  specificationChanged = new NullarySignal();

  constructor(public accordionOptions: AccordionOptions) {
    super();
    for (const sectionOptions of accordionOptions.sections) {
      this.getOrCreateSectionState(sectionOptions);
    }
  }

  getOrCreateSectionState(sectionOptions: AccordionSectionOptions) {
    const { jsonKey, defaultExpanded } = sectionOptions;
    let sectionState = this.getSectionState(jsonKey);
    if (sectionState === undefined) {
      sectionState = this.registerDisposer(
        new AccordionSectionState(this, jsonKey, defaultExpanded),
      );
    }
    return sectionState;
  }

  getSectionState(jsonKey: string): AccordionSectionState | undefined {
    return this.sectionStates.find((s) => s.jsonKey === jsonKey);
  }

  restoreState(obj: unknown) {
    if (obj === undefined || obj === null || typeof obj !== "object") {
      return;
    }
    for (let [jsonKey, isExpanded] of Object.entries(obj)) {
      if (typeof isExpanded === "boolean") {
        this.setSectionExpanded(jsonKey, isExpanded);
      }
    }
  }

  setSectionExpanded(jsonKey: string, expand?: boolean): void {
    const section = this.getSectionState(jsonKey);
    if (section !== undefined) {
      section.isExpanded.value = expand ?? !section.isExpanded.value;
    }
  }

  toJSON() {
    const sectionsData = this.sectionStates
      .map((section) => section.toJSON())
      .filter((data) => data !== undefined);

    return sectionsData.length === 0
      ? undefined
      : Object.assign({}, ...sectionsData);
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
      this.createAccordionSection(option);
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
      if (section === undefined) return;
      section.container.dataset.expanded = String(state.isExpanded.value);
      section.header.setAttribute(
        "aria-expanded",
        String(state.isExpanded.value),
      );
    });
  }

  private createAccordionSection(
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

    const chevron = document.createElement("span");
    chevron.classList.add("neuroglancer-accordion-chevron");
    chevron.innerHTML = svg_chevron_up;
    const headerText = document.createElement("span");
    headerText.classList.add("neuroglancer-accordion-header-text");
    headerText.textContent = option.displayName;
    header.appendChild(headerText);
    header.appendChild(chevron);

    container.style.display = "none";
    container.dataset.expanded = String(option.defaultExpanded ?? false);

    if (option.isDefaultKey) {
      this.defaultKey = option.jsonKey;
    }

    this.registerEventListener(newSection.header, "click", () =>
      this.setSectionExpanded(option.jsonKey),
    );

    // Usually, the state is pre-propulated with all the relevant sections.
    // However, because appendChild is public and can be called with
    // a jsonKey that is not in the initial accordionOptions, we need to
    // add the section into the state if that happens
    // This state wouldn't get properly restored if that occurs,
    // but in case there is some unforeseen section added, at least
    // the controls to expand/collapse it will still work because of this
    this.accordionState.getOrCreateSectionState(option);
    return newSection;
  }

  private getSectionByKey(jsonKey: string): AccordionSection | undefined {
    return this.sections.find((e) => e.jsonKey === jsonKey);
  }

  private getSectionWithFallback(jsonKey?: string): AccordionSection {
    return (this.getSectionByKey(jsonKey ?? this.defaultKey) ??
      this.getSectionByKey(this.defaultKey))!;
  }

  appendChild(content: HTMLElement, jsonKey?: string, hidden?: boolean): void {
    const section = this.getSectionWithFallback(jsonKey);
    section!.body.appendChild(content);
    if (!hidden) section!.container.style.display = "";
  }

  setSectionHidden(jsonKey: string, hidden: boolean): void {
    const section = this.getSectionByKey(jsonKey);
    if (section !== undefined) {
      section.container.style.display = hidden ? "none" : "";
    }
  }

  showSection(jsonKey: string): void {
    this.setSectionHidden(jsonKey, false);
  }

  hideSection(jsonKey: string): void {
    this.setSectionHidden(jsonKey, true);
  }
}
