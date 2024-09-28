import svg_plus from "#src/ui/images/add.svg?raw";
import type { MakeIconOptions } from "#src/widget/icon.js";
import { makeIcon } from "#src/widget/icon.js";

export function makeAddMoreButton(options: MakeIconOptions = {}) {
  // Create a button element
  const button = document.createElement("button");
  button.classList.add("metacell-neuroglancer-add-button");
  
  // Create the icon using makeIcon
  const icon = makeIcon({ svg: svg_plus, ...options });
  
  // Create a span for the text
  const textSpan = document.createElement("span");
  textSpan.textContent = options.title || ''; // Use title from options for the text
  
  // Append the icon and the text to the button
  button.appendChild(icon);
  button.appendChild(textSpan);
  
  return button;
}