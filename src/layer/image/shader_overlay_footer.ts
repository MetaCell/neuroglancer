export function makeFooterBtnGroup(onClose: () => void) {
  const buttonApply = document.createElement("button");
  buttonApply.textContent = "Close";
  buttonApply.classList.add("cancel-button");

  buttonApply.addEventListener("click", () => {
    onClose();
  });

  return buttonApply;
}
