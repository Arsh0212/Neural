// components/NeuralNetworkApp.js
class NeuralNetworkApp {
    constructor(containerId) {
        this.container = document.getElementById(containerId);
        this.width = 0;
        this.height = 0;
        
        // Performance optimization flags
        this.isUpdating = false;
        this.pendingUpdates = [];
        this.lastUpdateTime = 0;
        this.updateInterval = 100; // Minimum 100ms between renders
        
        // Component state tracking
        this.lastRenderState = null;
        this.matrixUpdateNeeded = true;
        
        // Initialize components
        this.layout = new NetworkLayout();
        this.metricsPanel = new MetricsPanel();
        
        // Initialize the app
        this.init();
    }

    init() {
        this.render();
        this.setupComponents();
        this.setupEventListeners();
        this.initWebSocket();
    }

    render() {
        this.container.innerHTML = `
            <div class="container">
                ${this.metricsPanel.render()}
                
                <div class="main-content">
                    <div class="visualization-panel">
                        <div class="toggle-panel" onclick="window.neuralApp.toggleMatrixPanel()">Toggle Matrix</div>
                        <svg id="network-svg"></svg>
                    </div>
                    
                    <div class="matrix-panel" id="matrixPanel" style="display: none;">
                        <div class="matrix-section">
                            <div class="matrix-title">Node Activations</div>
                            <div id="activationsMatrix"></div>
                        </div>
                        
                        <div class="matrix-section">
                            <div class="matrix-title">Layer Weights</div>
                            <div id="weightsMatrix"></div>
                        </div>
                    </div>
                </div>
            </div>
        `;
        this.metricsPanel.bindElements();
    }

    setupComponents() {
        // Initialize renderer and matrix display after DOM is created
        this.renderer = new NetworkRenderer('#network-svg');
        this.matrixDisplay = new MatrixDisplay('activationsMatrix', 'weightsMatrix');
        
        // Calculate initial layout
        this.updateDimensions();
        this.layout.calculateLayout(this.width, this.height);
        this.layout.createConnections();
        
        // Initial render
        this.renderer.setViewBox(this.width, this.height);
        this.renderer.render(this.layout);
        this.updateMatrixTables();
    }

    setupEventListeners() {
        // Debounced resize handler
        let resizeTimeout;
        window.addEventListener('resize', () => {
            clearTimeout(resizeTimeout);
            resizeTimeout = setTimeout(() => this.handleResize(), 250);
        });
        
        // Make this instance globally accessible for toggle function
        window.neuralApp = this;
    }

    updateDimensions() {
        const panel = document.querySelector('.visualization-panel');
        if (panel) {
            const newWidth = panel.clientWidth;
            const newHeight = panel.clientHeight;
            
            // Only update if dimensions actually changed
            if (newWidth !== this.width || newHeight !== this.height) {
                this.width = newWidth;
                this.height = newHeight;
                return true;
            }
        }
        return false;
    }

    handleResize() {
        if (this.updateDimensions()) {
            this.renderer.setViewBox(this.width, this.height);
            this.layout.calculateLayout(this.width, this.height);
            this.scheduleRender();
        }
    }

    toggleMatrixPanel() {
        const panel = document.getElementById('matrixPanel');
        if (panel) {
            const isHidden = panel.style.display === 'none';
            panel.style.display = isHidden ? 'block' : 'none';
            
            // Only update matrix if becoming visible
            if (isHidden && this.matrixUpdateNeeded) {
                this.updateMatrixTables();
                this.matrixUpdateNeeded = false;
            }
        }
    }

    initWebSocket() {
        // Setup WebSocket with optimized message handling
        this.wsManager = new WebSocketManager('ws://localhost:8000/ws/training/main', {
            onOpen: () => {
                console.log('WebSocket connected');
                this.metricsPanel.updateStatus(true);
            },
            onMessage: (data) => {
                this.queueUpdate(data);
            },
            onClose: () => {
                console.log('WebSocket disconnected');
                this.metricsPanel.updateStatus(false);
            },
            onError: (error) => {
                console.error('WebSocket error:', error);
                this.startDemo();
            }
        });

        // Give MetricsPanel a way to send messages
        this.metricsPanel.setSendHandler((msg) => {
            this.wsManager.send(JSON.stringify(msg));
        });

        this.wsManager.connect();
    }

    queueUpdate(data) {
        // Add update to queue
        this.pendingUpdates.push({
            timestamp: Date.now(),
            data: data
        });

        // Keep queue size manageable
        if (this.pendingUpdates.length > 10) {
            this.pendingUpdates = this.pendingUpdates.slice(-5);
        }

        // Process updates with throttling
        this.scheduleUpdate();
    }

    scheduleUpdate() {
        if (this.isUpdating) {
            return; // Already processing
        }

        const now = Date.now();
        const timeSinceLastUpdate = now - this.lastUpdateTime;

        if (timeSinceLastUpdate < this.updateInterval) {
            // Schedule for later
            setTimeout(() => this.scheduleUpdate(), this.updateInterval - timeSinceLastUpdate);
            return;
        }

        this.processUpdates();
    }

    processUpdates() {
        if (this.isUpdating || this.pendingUpdates.length === 0) {
            return;
        }

        this.isUpdating = true;

        // Get the latest update of each type
        const latestUpdates = {};
        this.pendingUpdates.forEach(update => {
            const updateType = update.data.type || 'default';
            if (!latestUpdates[updateType] || update.timestamp > latestUpdates[updateType].timestamp) {
                latestUpdates[updateType] = update;
            }
        });

        // Process the latest updates
        Object.values(latestUpdates).forEach(update => {
            if (update.data.type === 'send_epoch_update') {
                this.handleUpdate(update.data.data);
            }
        });

        // Clear the queue
        this.pendingUpdates = [];
        this.lastUpdateTime = Date.now();
        this.isUpdating = false;
    }

    handleUpdate(data) {
        console.log("Processing epoch update:", data.epoch);
        
        let needsRender = false;

        // Update metrics (lightweight)
        if (data.epoch !== undefined) {
            this.metricsPanel.updateEpoch(data.epoch);
        }
        
        if (data.loss !== undefined) {
            this.metricsPanel.updateLoss(data.loss);
        }

        // Handle activation nodes update
        if (data.activated_nodes && data.activated_nodes.length > 0) {
            data.activated_nodes.forEach((layerNodes, index) => {
                const activations = Array.isArray(layerNodes) ? 
                    layerNodes.map(val => Array.isArray(val) ? val[0] || val : val) : 
                    [layerNodes];
                
                if (this.layout.updateActivations(index, activations)) {
                    needsRender = true;
                }
            });
        }
        
        // Handle weights update
        if (data.weights && data.weights.length > 0) {
            if (this.layout.updateWeights(data.weights)) {
                needsRender = true;
            }
        }

        // Only render if something actually changed
        if (needsRender) {
            this.scheduleRender();
        }

        // Mark matrix update as needed (but don't update if panel is hidden)
        const matrixPanel = document.getElementById('matrixPanel');
        if (matrixPanel && matrixPanel.style.display !== 'none') {
            this.updateMatrixTables();
        } else {
            this.matrixUpdateNeeded = true;
        }
    }

    scheduleRender() {
        // Use requestAnimationFrame for smooth rendering
        if (!this.renderScheduled) {
            this.renderScheduled = true;
            requestAnimationFrame(() => {
                this.renderer.render(this.layout);
                this.renderScheduled = false;
            });
        }
    }

    updateMatrixTables() {
        try {
            // Only update if matrix panel is visible
            const matrixPanel = document.getElementById('matrixPanel');
            if (!matrixPanel || matrixPanel.style.display === 'none') {
                this.matrixUpdateNeeded = true;
                return;
            }

            this.matrixDisplay.updateActivationsMatrix(this.layout.layers);
            this.matrixDisplay.updateWeightsMatrix(this.layout.weights, this.layout.layers);
            this.matrixUpdateNeeded = false;
        } catch (error) {
            console.error('Error updating matrix tables:', error);
        }
    }

    startDemo() {
        console.log('Starting demo mode');
        
        // Clear any existing demo interval
        if (this.demoInterval) {
            clearInterval(this.demoInterval);
        }

        this.demoInterval = setInterval(() => {
            // Simulate training data with reduced frequency
            if (Math.random() > 0.3) { // Only update 70% of the time
                const mockData = {
                    type: 'send_epoch_update',
                    data: {
                        epoch: (this.demoEpoch = (this.demoEpoch || 0) + 1),
                        loss: Math.random() * 0.5 + 0.1,
                        accuracy: Math.random() * 0.3 + 0.7,
                        activated_nodes: this.layout.layers.map(layer => 
                            layer.nodes.map(() => Math.random())
                        ),
                        weights: [
                            [[0]], // Input layer placeholder
                            ...this.layout.weights.slice(1).map(layerWeights => 
                                layerWeights ? layerWeights.map(row => 
                                    row.map(() => Math.random() * 2 - 1)
                                ) : [[0]]
                            )
                        ]
                    }
                };

                this.queueUpdate(mockData);
            }
        }, 800); // Slower demo updates
    }

    destroy() {
        // Cleanup method
        if (this.demoInterval) {
            clearInterval(this.demoInterval);
        }
        
        if (this.wsManager) {
            this.wsManager.disconnect();
        }

        // Clear pending updates
        this.pendingUpdates = [];
        this.isUpdating = false;
    }
}
