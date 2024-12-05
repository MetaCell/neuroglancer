import { RefCounted } from "#src/util/disposable.js";

export type MessagePayload = {
  type: string;
  payload: Record<string, any>;
};

export const CREATE_FIGURE = "CREATE_FIGURE" as const;


export class IncomingEventsHandler extends RefCounted {
  constructor(public viewer: any) {
    super();
    this.initialize();
  }

  private initialize() {
    this.registerEventListener(window, 'message', this.handlePostMessage.bind(this));
  }


  private handlePostMessage(event: MessageEvent) {
    const { data } = event;

    if (!data || typeof data.type !== 'string') {
      return;
    }

    const message: MessagePayload = data;
    this.handleMessageType(message);
  }

  private handleMessageType(message: MessagePayload) {
    switch (message.type) {
      case CREATE_FIGURE:
        this.createFigure();
        break;
    }
  }

  private createFigure() {
    this.viewer.showScreenshotDialog()
  }
}