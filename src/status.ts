/**
 * @license
 * Copyright 2016 Google Inc.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

import "#src/status.css";

import { makeCloseButton } from "#src/widget/close_button.js";

let statusContainer: HTMLElement | undefined;
let modalStatusContainer: HTMLElement | undefined;

// Exported for use by #tests/fixtures/status_message_handler.js
export const statusMessages = new Set<StatusMessage>();

export const DEFAULT_STATUS_DELAY = 200;

export type Delay = boolean | number;

function setupStatusContainer(container: HTMLElement) {
  container.addEventListener("mousedown", (event) => {
    // Prevent focus changes due to clicking on status message.
    event.preventDefault();
  });
}

function getStatusContainer() {
  if (statusContainer === undefined) {
    statusContainer = document.createElement("ul");
    setupStatusContainer(statusContainer);
    statusContainer.id = "neuroglancer-status-container";
    const el: HTMLElement | null = document.getElementById(
      "neuroglancer-container",
    );
    if (el) {
      el.appendChild(statusContainer);
    } else {
      document.body.appendChild(statusContainer);
    }
  }
  return statusContainer;
}

function getModalStatusContainer() {
  if (modalStatusContainer === undefined) {
    modalStatusContainer = document.createElement("ul");
    setupStatusContainer(modalStatusContainer);
    modalStatusContainer.id = "neuroglancer-status-container-modal";
    const el: HTMLElement | null = document.getElementById(
      "neuroglancer-container",
    );
    if (el) {
      el.appendChild(modalStatusContainer);
    } else {
      document.body.appendChild(modalStatusContainer);
    }
  }
  return modalStatusContainer;
}

// For use by #tests/fixtures/status_message_handler.js
export function getStatusMessageContainers() {
  return [getStatusContainer(), getModalStatusContainer()];
}

export class StatusMessage {
  element: HTMLElement;
  private modalElementWrapper: HTMLElement | undefined;
  private modalHeader: HTMLElement | undefined;
  private modalContent: HTMLElement | undefined;
  private modalFooter: HTMLElement | undefined;
  private timer: number | null;
  private visibility = true;
  constructor(delay: Delay = false, modal = false) {
    const element = document.createElement("li");
    this.element = element;
    if (delay === true) {
      delay = DEFAULT_STATUS_DELAY;
    }
    this.setModal(modal);
    if (delay !== false) {
      this.setVisible(false);
      this.timer = window.setTimeout(this.setVisible.bind(this, true), delay);
    } else {
      this.timer = null;
    }
    statusMessages.add(this);
  }

  [Symbol.dispose]() {
    this.dispose();
  }

  dispose() {
    if (this.modalElementWrapper) {
      modalStatusContainer!.removeChild(this.modalElementWrapper);
    } else {
      statusContainer!.removeChild(this.element);
    }
    if (this.timer !== null) {
      clearTimeout(this.timer);
    }
    statusMessages.delete(this);
  }
  setText(text: string, makeVisible?: boolean) {
    this.element.textContent = text;
    if (makeVisible) {
      this.setVisible(true);
    }
  }
  setHTML(text: string, makeVisible?: boolean) {
    this.element.innerHTML = text;
    if (makeVisible) {
      this.setVisible(true);
    }
  }
  setModal(value: boolean) {
    if (value) {
      if (this.modalElementWrapper === undefined) {
        const modalElementWrapper = document.createElement("div");
        modalElementWrapper.className = "neuroglancer-status-modal-wrapper"
        const modalHeader = document.createElement("div");
        modalHeader.className = "neuroglancer-status-modal-header";
        this.modalHeader = modalHeader;
        
        const dismissModalElement = makeCloseButton({
          title: "Dismiss",
          onClick: () => {
            this.setModal(false);
          },
        });
        dismissModalElement.classList.add("neuroglancer-dismiss-modal");
        
        const titleElement = document.createElement("div");
        titleElement.className = "neuroglancer-status-modal-title";
        modalHeader.appendChild(titleElement);
        modalHeader.appendChild(dismissModalElement);
        
        // Create content area
        const modalContent = document.createElement("div");
        modalContent.className = "neuroglancer-status-modal-content";
        this.modalContent = modalContent;
        
        // Create footer for buttons
        const modalFooter = document.createElement("div");
        modalFooter.className = "neuroglancer-status-modal-footer";
        this.modalFooter = modalFooter;

        modalContent.appendChild(this.element);
        
        modalElementWrapper.appendChild(modalHeader);
        modalElementWrapper.appendChild(modalContent);
        modalElementWrapper.appendChild(modalFooter);
        
        this.modalElementWrapper = modalElementWrapper;
        this.applyVisibility();
        getModalStatusContainer().appendChild(modalElementWrapper);
      }
    } else {
      if (this.modalElementWrapper !== undefined) {
        modalStatusContainer!.removeChild(this.modalElementWrapper);
        this.modalElementWrapper = undefined;
        this.modalHeader = undefined;
        this.modalContent = undefined;
        this.modalFooter = undefined;
        getStatusContainer().appendChild(this.element);
      } else if (this.element.parentElement === null) {
        getStatusContainer().appendChild(this.element);
      }
    }
  }

  private applyVisibility() {
    const newVisibility = this.visibility ? "" : "none";
    this.element.style.display = newVisibility;
    const { modalElementWrapper } = this;
    if (modalElementWrapper !== undefined) {
      modalElementWrapper.style.display = newVisibility;
    }
  }

  setVisible(value: boolean) {
    if (this.timer !== null) {
      clearTimeout(this.timer);
      this.timer = null;
    }
    if (value !== this.visibility) {
      this.visibility = value;
      this.applyVisibility();
    }
  }

  static forPromise<T>(
    promise: Promise<T>,
    options: { initialMessage: string; delay?: Delay; errorPrefix: string },
  ): Promise<T> {
    const status = new StatusMessage(options.delay);
    status.setText(options.initialMessage);
    const dispose = status.dispose.bind(status);
    promise.then(dispose, (reason) => {
      let msg: string;
      if (reason instanceof Error) {
        msg = reason.message;
      } else {
        msg = "" + reason;
      }
      const { errorPrefix = "" } = options;
      status.setErrorMessage(errorPrefix + msg);
      status.setVisible(true);
    });
    return promise;
  }

  setErrorMessage(message: string) {
    this.element.textContent = message + " ";
    const button = document.createElement("button");
    button.textContent = "Dismiss";
    button.addEventListener("click", () => {
      this.dispose();
    });
    this.element.appendChild(button);
  }

  static showMessage(message: string): StatusMessage {
    const msg = new StatusMessage();
    msg.element.textContent = message;
    msg.setVisible(true);
    return msg;
  }

  static showTemporaryMessage(
    message: string,
    closeAfter = 2000,
  ): StatusMessage {
    const msg = StatusMessage.showMessage(message);
    window.setTimeout(() => msg.dispose(), closeAfter);
    return msg;
  }

  setTitle(title: string) {
    if (this.modalHeader) {
      const titleElement = this.modalHeader.querySelector(".neuroglancer-status-modal-title");
      if (titleElement) {
        titleElement.textContent = title;
      }
    }
  }

  addButtonToFooter(button: HTMLButtonElement, cancelHandler?: () => void) {
    if (this.modalFooter) {
      // Create cancel button
      const cancelButton = document.createElement("button");
      cancelButton.textContent = "Cancel";
      cancelButton.classList.add("neuroglancer-status-modal-cancel-button");
      
      cancelButton.addEventListener("click", cancelHandler || (() => {
        this.dispose();
      }));

      button.className = "neuroglancer-status-modal-action-button";

      this.modalFooter.appendChild(cancelButton);
      this.modalFooter.appendChild(document.createTextNode(' '));
      this.modalFooter.appendChild(button);
    }
  }

  setContentText(text: string) {
    if (this.modalContent) {
      this.element.textContent = text;
    } else {
      this.setText(text);
    }
  }
}
