import "#src/widget/accordion.css";

export type AccordionItem = {
  title: string;
  content: HTMLDivElement; // Expecting actual HTML elements for content
  open?: boolean;
};

export class Accordion {
  private items: AccordionItem[];
  private element: HTMLElement;

  constructor(items: AccordionItem[]) {
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

      if (item.open) {
        this.toggleContent(itemElement, contentElement, true);
      }

      // Event listener to toggle the content display
      titleElement.addEventListener("click", () => {
        const isVisible = contentElement.style.display === "block";
        this.toggleContent(itemElement, contentElement, !isVisible); // Pass the itemElement to toggle the expanded class
      });

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
}
