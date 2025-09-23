// components/MetricsPanel.js
class MetricsPanel {
    constructor() {
        this.config = {
            epochs: 100,
            learningRate: 0.01,
            batchSize: 32,
            activationFunction: 'relu',
            datasetFunction: 1
        };
        this.bindElements();
        this.bindEvents();
        this.initWebSocket();
    }

    bindElements() {
        this.epochElement = document.getElementById('epochValue');
        this.lossElement = document.getElementById('lossValue');
        this.statusIndicator = document.getElementById('statusIndicator');
        this.statusText = document.getElementById('statusText');
        
        this.epochsInput = document.getElementById('epochsInput');
        this.learningRateSelect = document.getElementById('learningRateSelect');
        this.batchSizeInput = document.getElementById('batchSizeInput');
        this.activationSelect = document.getElementById('activationSelect');
        this.datasetSelect = document.getElementById('datasetSelect');
    }

    initWebSocket() {
        const loc = window.location;
        let wsStart = loc.protocol === "https:" ? "wss://" : "ws://";
        let endpoint = wsStart + loc.host + "/ws/training/";
        this.webSocket = new WebSocket(endpoint)

        this.webSocket.onopen = () => {
            console.log("Metrics Connection established");
            this.updateStatus(true);
        }

        this.webSocket.onclose = () => {
            console.log("Metrics connection closed");
            this.updateStatus(false);
        }

        this.webSocket.onmessage = (event) => {
            try {
                // FIXED: Parse JSON only once and use proper logic
                const data = JSON.parse(event.data);
                console.log("Random data received")
                // Handle config updates
                if (data.type === "config" && data.config) {
                    console.log("config data received")
                    this.updateFromBackend(data.config);
                    return;
                }
                
                // Handle epoch updates from neural network training
                if (data.type === "send_epoch_update" && data.data) {
                    this.handleUpdate({
                        epoch: data.data.epoch,
                        loss: data.data.loss
                    });
                    return;
                }            
            } catch (error) {
                console.error("Error parsing WebSocket message:", error, event.data);
            }
        }
    }

    updateFromBackend(config) {
        this.config = config;

        // Update inputs with backend values
        if (this.epochsInput) this.epochsInput.value = config.epochs;
        if (this.learningRateSelect) this.learningRateSelect.value = config.learningRate;
        if (this.batchSizeInput) this.batchSizeInput.value = config.batchSize;
        if (this.activationSelect) this.activationSelect.value = config.activationFunction;
        if (this.datasetSelect) this.datasetSelect.value = config.datasetFunction;
    }

    bindEvents() {
        setTimeout(() => {
            if (this.epochsInput) {
                this.epochsInput.addEventListener('change', (e) => {
                    this.config.epochs = parseInt(e.target.value) || 100;
                    this.updateValues();
                });
            }

            if (this.learningRateSelect) {
                this.learningRateSelect.addEventListener('change', (e) => {
                    this.config.learningRate = parseFloat(e.target.value) || 0.01;
                    this.updateValues();
                });
            }

            if (this.batchSizeInput) {
                this.batchSizeInput.addEventListener('change', (e) => {
                    this.config.batchSize = parseInt(e.target.value) || 32;
                    this.updateValues();
                });
            }

            if (this.activationSelect) {
                this.activationSelect.addEventListener('change', (e) => {
                    this.config.activationFunction = e.target.value;
                    this.updateValues();
                });
            }

            if (this.datasetSelect) {
                this.datasetSelect.addEventListener('change', (e) => {
                    this.config.datasetFunction = parseInt(e.target.value);
                    console.log(this.config)
                    this.updateValues();
                });
            }

            // Enhanced Train Button
            const trainBtn = document.getElementById("trainBtn");
            if (trainBtn) {
                trainBtn.addEventListener("click", () => {
                    // Add loading state
                    trainBtn.classList.add('loading');
                    trainBtn.innerHTML = '<span class="btn-spinner"></span>Training...';
                    trainBtn.disabled = true;
                    
                    // Reset display when starting training
                    this.updateEpoch(0);
                    this.updateLoss(null);
                    
                    fetch("/train/")
                        .then(response => response.json())
                        .then(data => {
                            console.log("Training started:", data);
                            // Reset button state
                            trainBtn.classList.remove('loading');
                            trainBtn.innerHTML = '<span class="btn-icon">🚀</span>Train Model';
                            trainBtn.disabled = false;
                        })
                        .catch(err => {
                            console.error("Training error:", err);
                            // Reset button state and show error
                            trainBtn.classList.remove('loading');
                            trainBtn.classList.add('error');
                            trainBtn.innerHTML = '<span class="btn-icon">❌</span>Error';
                            trainBtn.disabled = false;
                            
                            // Reset error state after 3 seconds
                            setTimeout(() => {
                                trainBtn.classList.remove('error');
                                trainBtn.innerHTML = '<span class="btn-icon">🚀</span>Train Model';
                            }, 3000);
                            
                            alert("Training Error: " + err);
                        });
                });
            }

            // New Page Navigation Button
            const pageBtn = document.getElementById("pageBtn");
            if (pageBtn) {
                pageBtn.addEventListener("click", () => {
                    // Add loading state
                    pageBtn.classList.add('loading');
                    pageBtn.innerHTML = '<span class="btn-spinner"></span>Loading...';
                    pageBtn.disabled = true;
                    
                    const targetUrl = "/graphs/"; 
                    
                    // Simulate loading delay for better UX
                    setTimeout(() => {
                        window.location.href = targetUrl;
                        
                        // Reset button state
                        pageBtn.classList.remove('loading');
                        pageBtn.innerHTML = '<span class="btn-icon">📊</span>Dashboard';
                        pageBtn.disabled = false;
                    }, 500);
                });
            }
        }, 100);
    }

    handleUpdate(data) {        
        if (data && typeof data === 'object') {
            // Update epoch if provided and valid
            if (data.epoch !== undefined && data.epoch !== null) {
                this.updateEpoch(data.epoch);
            }
        
            // Update loss if provided and valid
            if (data.loss !== undefined && data.loss !== null && data.loss !== '-') {
                this.updateLoss(data.loss);
            }
        }
    }

    updateEpoch(epoch) {
        if (this.epochElement && epoch != 0) {
            // Ensure we show a valid number, default to 0 if invalid
            const epochValue = (epoch !== null && epoch !== undefined) ? parseInt(epoch) : 0;
            this.epochElement.textContent = epochValue;
        }
    }

    updateLoss(loss) {
        if (this.lossElement) {
            if (loss !== null && loss !== undefined && !isNaN(loss) && loss !== '-') {
                const lossValue = parseFloat(loss);
                this.lossElement.textContent = lossValue.toFixed(4);
            } 
        }
    }

    updateStatus(isConnected) {
        if (this.statusIndicator) {
            if (isConnected) {
                this.statusIndicator.classList.add('connected');
                if (this.statusText) this.statusText.textContent = 'Connected';
            } else {
                this.statusIndicator.classList.remove('connected');
                if (this.statusText) this.statusText.textContent = 'Disconnected';
            }
        }
    }

    setSendHandler(handler) {
        this.sendHandler = handler;
    }

    updateValues() {
        if (this.config && this.webSocket && this.webSocket.readyState === WebSocket.OPEN) {
            this.webSocket.send(JSON.stringify({
                type: "config",
                config: this.config
            }));
        }
    }

    render() {
        return `
            <div class="header">
                <div class="metrics-section">
                    <div class="metrics">
                        <div class="metric">
                            <div class="metric-label">Epoch</div>
                            <div class="metric-value" id="epochValue">0</div>
                        </div>
                        <div class="metric">
                            <div class="metric-label">Loss</div>
                            <div class="metric-value" id="lossValue">-</div>
                        </div>
                    </div>
                </div>
                
                <div class="controls-section">
                    <div class="control-group">
                        <label class="control-label">
                            <span class="label-icon">⚡</span>
                            Max Epochs
                        </label>
                        <input 
                            type="number" 
                            id="epochsInput" 
                            class="control-input" 
                            value="${this.config.epochs}"
                            min="10"
                            max="10000"
                            step="10"
                        />
                    </div>
                    
                    <div class="control-group">
                        <label class="control-label">
                            <span class="label-icon">📊</span>
                            Learning Rate
                        </label>
                        <select id="learningRateSelect" class="control-select enhanced-select">
                            <option value="1" ${this.config.learningRate === 1 ? 'selected' : ''}>1.0</option>
                            <option value="0.1" ${this.config.learningRate === 0.1 ? 'selected' : ''}>0.1</option>
                            <option value="0.01" ${this.config.learningRate === 0.01 ? 'selected' : ''}>0.01</option>
                            <option value="0.001" ${this.config.learningRate === 0.001 ? 'selected' : ''}>0.001</option>
                            <option value="0.0001" ${this.config.learningRate === 0.0001 ? 'selected' : ''}>0.0001</option>
                        </select>
                    </div>
                    
                    <div class="control-group">
                        <label class="control-label">
                            <span class="label-icon">📦</span>
                            Batch Size
                        </label>
                        <input 
                            type="number" 
                            id="batchSizeInput" 
                            class="control-input" 
                            value="${this.config.batchSize}"
                            min="10"
                            max="1000"
                            step="10"
                        />
                    </div>
                    
                    <div class="control-group">
                        <label class="control-label">
                            <span class="label-icon">🔄</span>
                            Activation
                        </label>
                        <select id="activationSelect" class="control-select enhanced-select">
                            <option value="relu" ${this.config.activationFunction === 'relu' ? 'selected' : ''}>ReLU</option>
                            <option value="sigmoid" ${this.config.activationFunction === 'sigmoid' ? 'selected' : ''}>Sigmoid</option>
                            <option value="tanh" ${this.config.activationFunction === 'tanh' ? 'selected' : ''}>Tanh</option>
                            <option value="linear" ${this.config.activationFunction === 'linear' ? 'selected' : ''}>Linear</option>
                        </select>
                    </div>

                    <div class="control-group">
                        <label class="control-label">
                            <span class="label-icon"></span>
                            Dataset
                        </label>
                        <select id="datasetSelect" class="control-select enhanced-select">
                            <option value="1" ${this.config.dataset === 1 ? 'selected' : ''}>Moons</option>
                            <option value="2" ${this.config.dataset === 2 ? 'selected' : ''}>Circles</option>
                            <option value="3" ${this.config.dataset === 3 ? 'selected' : ''}>Blobs</option>
                            <option value="4" ${this.config.dataset === 4 ? 'selected' : ''}>Linear</option>
                        </select>
                    </div>

                </div>

                <div class="actions-section neural-actions-section">
                    <div class="action-buttons neural-action-buttons">
                        <button 
                            type="button" 
                            id="trainBtn" 
                            class="action-btn primary-btn"
                        >
                            <span class="btn-icon">🚀</span>
                            Train Model
                        </button>
                        
                        <button 
                            type="button" 
                            id="pageBtn" 
                            class="action-btn secondary-btn"
                        >
                            <span class="btn-icon">📊</span>
                            Graph
                        </button>
                    </div>
                    
                    <div class="status">
                        <div class="status-indicator" id="statusIndicator"></div>
                        <span id="statusText">Connecting...</span>
                    </div>
                </div>
            </div>
        `;
    }
}
