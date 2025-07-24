import { TrackableBoolean } from "#src/trackable_boolean.js";
import { WatchableValueInterface } from "#src/trackable_value.js";
import { Owned, RefCounted } from "#src/util/disposable.js";
import { parseArray } from "#src/util/json.js";
import { NullarySignal } from "#src/util/signal.js";
import "#src/widget/accordion.css";
import { OptionSpecification, Tab } from "#src/widget/tab_view.js";

export const ACCORDION_JSON_KEY = "accordion";

// TODO this is not so ideal
const JSON_KEY_TRANSLATIONS = {
  source: "Source",
  imageSlice: "Slice",
  imageVolume: "Volume Rendering",
  shader: "Shader controls",
  default: "Settings",
};

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

export class AccordionSpecification extends OptionSpecification<{
  label: string;
  order?: number;
  getter: () => Owned<AccordionTab>;
  expanded?: WatchableValueInterface<boolean>;
}> {}

export class AccordionTab extends Tab {
  sections: AccordionSection[] = [];
  constructor(
    protected accordionTabState: AccordionSectionStates,
    private defaultKey: keyof typeof JSON_KEY_TRANSLATIONS = "default",
  ) {
    super();
    this.element.classList.add("neuroglancer-accordion");
    this.registerDisposer(
      this.accordionTabState.specificationChanged.add(() =>
        this.updateSectionsExpanded(),
      ),
    );
    this.updateSectionsExpanded();
  }

  private setSectionExpanded(jsonKey: string, expand?: boolean): void {
    this.accordionTabState.setSectionExpanded(jsonKey, expand);
  }

  private updateSectionsExpanded() {
    this.accordionTabState.sectionStates.forEach((state) => {
      const section = this.makeOrGetSection(state.jsonKey);
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

  private makeNewSection(jsonKey: string): AccordionSection | undefined {
    if (!(jsonKey in JSON_KEY_TRANSLATIONS)) return undefined;
    const newSection: AccordionSection = {
      name: JSON_KEY_TRANSLATIONS[
        jsonKey as keyof typeof JSON_KEY_TRANSLATIONS
      ],
      jsonKey,
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
    container.dataset.expanded = "false";

    this.registerEventListener(newSection.header, "click", () =>
      this.setSectionExpanded(jsonKey),
    );
    return newSection;
  }

  makeOrGetSection(jsonKey: string): AccordionSection | undefined {
    const section = this.sections.find((e) => e.jsonKey === jsonKey);
    return section ?? this.makeNewSection(jsonKey);
  }

  getSectionByKey(jsonKey: string): AccordionSection | undefined {
    return this.sections.find((e) => e.jsonKey === jsonKey);
  }

  appendChild(
    content: HTMLElement,
    jsonKey: keyof typeof JSON_KEY_TRANSLATIONS = this.defaultKey,
  ) {
    const section =
      this.makeOrGetSection(jsonKey ?? this.defaultKey) ??
      this.makeOrGetSection(this.defaultKey);
    section!.body.appendChild(content);
  }
}
