import "#src/widget/accordion.css";

export type AccordionItem = {
  title: string;
  content: HTMLDivElement; // Expecting actual HTML elements for content
}

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

      // Event listener to toggle the content display
      titleElement.addEventListener("click", () => {
        const isVisible = contentElement.style.display === "block";
        this.toggleContent(contentElement, !isVisible); // Toggle content
      });

      itemElement.appendChild(titleElement);
      itemElement.appendChild(contentElement);
      this.element.appendChild(itemElement);
    }
  }

  private toggleContent(contentElement: HTMLElement, shouldShow: boolean) {
    if (shouldShow) {
      contentElement.style.display = "block";
      contentElement.classList.add("show");
    } else {
      contentElement.style.display = "none";
      contentElement.classList.remove("show");
    }
  }

  public getElement(): HTMLElement {
    return this.element;
  }
}
