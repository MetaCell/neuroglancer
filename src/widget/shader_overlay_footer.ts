export function makeFooterBtnGroup() {
  const wrapper = document.createElement("div");
  wrapper.className = "button-group";

  const downloadButton = document.createElement("button");
  downloadButton.classList.add("text-primary");
  downloadButton.textContent = "Download";
  downloadButton.title = "Download state as a JSON file";
  wrapper.appendChild(downloadButton);
  // downloadButton.addEventListener("click", () => this.downloadState());

  const buttonApply = document.createElement("button");
  buttonApply.textContent = "Save";
  buttonApply.classList.add("outlined-primary");
  wrapper.appendChild(buttonApply);
  // buttonApply.addEventListener("click", () => {
  //   // this.applyChanges();
  // });
  buttonApply.disabled = true;

  const buttonClose = document.createElement("button");
  buttonClose.classList.add("contained-primary");
  buttonClose.textContent = "Save & Close";
  wrapper.appendChild(buttonClose);
  // buttonClose.addEventListener("click", () => {
  //   // this.applyChanges();
  //   this.dispose();
  // });
  return wrapper;
}