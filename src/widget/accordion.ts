import type { WatchableValueInterface } from "#src/trackable_value.js";
import { RefCounted } from "#src/util/disposable.js";
import { removeFromParent } from "#src/util/dom.js";
import "#src/widget/accordion.css";

export type AccordionItem = {
  title: string;
  content: HTMLDivElement; // Expecting actual HTML elements for content
  state?: WatchableValueInterface<boolean>;
  open?: boolean;
};

export class Accordion extends RefCounted {
  private items: AccordionItem[];
  private element: HTMLElement;

  constructor(items: AccordionItem[]) {
    super();
    this.items = items;
    this.element = document.createElement("div");
    this.element.classList.add("accordion");

    this.render();
  }

  private render() {
    for (const item of this.items) {
      const itemElement = document.createElement("div");
      itemElement.classList.add("accordion-item");

      const titleElement = document.createElement("div");
      titleElement.classList.add("accordion-title");
      titleElement.textContent = item.title;

      const contentElement = document.createElement("div");
      contentElement.classList.add("accordion-content");
      contentElement.style.display = "none"; // Hide content initially

      // Append the actual content (DOM element) instead of using textContent
      contentElement.appendChild(item.content);

      if (item.state) {
        this.toggleContent(itemElement, contentElement, item.state.value);

        this.registerDisposer(
          item.state.changed.add(() => {
            this.toggleContent(itemElement, contentElement, item.state!.value);
          }),
        );

        this.registerEventListener(titleElement, "click", () => {
          item.state!.value = !item.state!.value;
        });
      } else {
        // if there is no state tracking, we use item.open to
        // provide the accordion state, should't be needed tho

        this.toggleContent(itemElement, contentElement, item.open ?? false);

        this.registerEventListener(titleElement, "click", () => {
          item.open = !item.open;
          this.toggleContent(itemElement, contentElement, item.open ?? false);
        });
      }

      itemElement.appendChild(titleElement);
      itemElement.appendChild(contentElement);
      this.element.appendChild(itemElement);
    }
  }

  private toggleContent(
    itemElement: HTMLElement,
    contentElement: HTMLElement,
    shouldShow: boolean,
  ) {
    if (shouldShow) {
      contentElement.style.display = "block";
      contentElement.classList.add("show");

      // Add 'accordion-expanded' to both the main accordion element and the current item
      this.element.classList.add("accordion-expanded");
      itemElement.classList.add("accordion-expanded");
    } else {
      contentElement.style.display = "none";
      contentElement.classList.remove("show");

      // Remove 'accordion-expanded' from both the main accordion element and the current item
      itemElement.classList.remove("accordion-expanded");

      // Check if any other items are expanded before removing from the main accordion element
      const anyExpanded = this.element.querySelector(
        ".accordion-item.accordion-expanded",
      );
      if (!anyExpanded) {
        this.element.classList.remove("accordion-expanded");
      }
    }
  }

  public getElement(): HTMLElement {
    return this.element;
  }

  disposed() {
    removeFromParent(this.element);
    super.disposed();
  }
}
