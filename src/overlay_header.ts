import close from "ikonate/icons/close.svg?raw";
import { makeIcon } from "#src/widget/icon.js";

interface OverlayHeaderOptions {
  parentElem: HTMLDivElement;
  title: string;
  onClose: () => void;
}

export function makeOverlayHeader({
  parentElem,
  title,
  onClose,
}: OverlayHeaderOptions) {
  parentElem.classList.add("neuroglancer-state-editor");

  const wrapper = document.createElement("div");
  wrapper.className = "overlay-header";

  const heading = document.createElement("p");
  heading.className = "overlay-heading";
  heading.textContent = title;

  wrapper.appendChild(heading);

  wrapper.appendChild(
    makeIcon({
      svg: close,
      onClick: onClose,
    }),
  );

  parentElem.appendChild(wrapper);
}
