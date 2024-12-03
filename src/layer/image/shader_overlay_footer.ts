export function makeFooterBtnGroup(onClose: () => void) {
  // const wrapper = document.createElement("div");
  // wrapper.className =  "button-wrapper";

  // const hiddenButton = document.createElement("button");
  // hiddenButton.hidden = true;
  // wrapper.appendChild(hiddenButton);

  const buttonApply = document.createElement("button");
  buttonApply.textContent = "Close";
  buttonApply.classList.add("primary-button");
  // wrapper.appendChild(buttonApply);
  buttonApply.addEventListener("click", () => {
    onClose();
  });

  return buttonApply;
}
