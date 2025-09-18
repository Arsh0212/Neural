// components/WebSocketManager.js
class WebSocketManager {
    constructor(url, callbacks = {}) {
        this.url = url;
        this.callbacks = callbacks;
        this.ws = null;
        this.isConnected = false;
        this.reconnectDelay = 3000;
    }

    connect() {
        try {
            this.ws = new WebSocket(this.url);
            console.log("Hello")
            this.ws.onopen = () => {
                console.log("Connection Established")
                this.isConnected = true;
                if (this.callbacks.onOpen) {
                    this.callbacks.onOpen();
                }
            };
            
            this.ws.onmessage = (event) => {
                const data = JSON.parse(event.data);
                if (this.callbacks.onMessage) {
                    this.callbacks.onMessage(data);
                }
            };
            
            this.ws.onclose = () => {
                this.isConnected = false;
                if (this.callbacks.onClose) {
                    this.callbacks.onClose();
                }
                setTimeout(() => this.connect(), this.reconnectDelay);
            };
            
            this.ws.onerror = (error) => {
                if (this.callbacks.onError) {
                    this.callbacks.onError(error);
                }
            };
        } catch (error) {
            if (this.callbacks.onError) {
                this.callbacks.onError(error);
            }
        }
    }

    disconnect() {
        if (this.ws) {
            this.ws.close();
        }
    }

    send(data) {
        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
            this.ws.send(data);
        } else {
            console.warn("WebSocket is not open. Cannot send data.");
        }
    }
}