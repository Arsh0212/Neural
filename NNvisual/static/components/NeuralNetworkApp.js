// components/NeuralNetworkApp.js
class NeuralNetworkApp {
    constructor(containerId) {
        this.container = document.getElementById(containerId);
        this.width = 0;
        this.height = 0;
        
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
                    
                    <div class="matrix-panel" id="matrixPanel">
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
        this.metricsPanel.bindElements(); // <-- Add this line
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
        window.addEventListener('resize', () => this.handleResize());
        
        // Make this instance globally accessible for toggle function
        window.neuralApp = this;
    }

    updateDimensions() {
        const panel = document.querySelector('.visualization-panel');
        if (panel) {
            this.width = panel.clientWidth;
            this.height = panel.clientHeight;
        }
    }

    handleResize() {
        this.updateDimensions();
        this.renderer.setViewBox(this.width, this.height);
        this.layout.calculateLayout(this.width, this.height);
        this.renderer.render(this.layout);
    }

    toggleMatrixPanel() {
        const panel = document.getElementById('matrixPanel');
        if (panel) {
            panel.style.display = panel.style.display === 'none' ? 'block' : 'none';
        }
    }

    initWebSocket() {
        // Initialize with demo data
        // this.layout.layers.forEach((layer, i) => {
        //     this.layout.updateActivations(i, layer.activations);
        // });
        
        // Setup WebSocket
        this.wsManager = new WebSocketManager('ws://localhost:8000/ws/training/main', {
            onOpen: () => {
                console.log(this.wsManager)
                this.metricsPanel.updateStatus(true);
            },
            onMessage: (data) => {
                console.log("On epoch end",data)
                if (data.type === 'send_epoch_update'){
                this.handleUpdate(data.data);
                }
            },
            onClose: () => {
                this.metricsPanel.updateStatus(false);
            },
            onError: () => {
                this.startDemo();
            }
        });
        // Give MetricsPanel a way to send messages
        this.metricsPanel.setSendHandler((msg) => {
            this.wsManager.send(JSON.stringify(msg));
            });

        this.wsManager.connect();
    }

    
    handleUpdate(data) {
        const { epoch, weights, activated_nodes, loss } = data;
        
        this.metricsPanel.updateEpoch(epoch || 0);
        this.metricsPanel.updateLoss(loss);
        
        if (activated_nodes && activated_nodes.length > 0) {
            activated_nodes.forEach((layerNodes, index) => {
                const activations = Array.isArray(layerNodes) ? 
                    layerNodes.map(val => Array.isArray(val) ? val : val) : 
                    layerNodes;
                this.layout.updateActivations(index, activations);
            });
        }
        
        if (weights && weights.length > 0) {
            this.layout.updateWeights(weights);
        }

        this.renderer.render(this.layout);
        this.updateMatrixTables();
    }

    updateMatrixTables() {
        this.matrixDisplay.updateActivationsMatrix(this.layout.layers);
        this.matrixDisplay.updateWeightsMatrix(this.layout.weights, this.layout.layers);
    }

    startDemo() {
        setInterval(() => {
            // Simulate training data
            this.layout.layers.forEach((layer, i) => {
                const activations = layer.activations.map(() => Math.random());
                this.layout.updateActivations(i, activations);
            });
            
            // Update weights occasionally
            if (Math.random() > 0.7) {
                const demoWeights = [
                    null,
                    Array(8).fill().map(() => Array(2).fill().map(() => Math.random() * 2 - 1)),
                    Array(8).fill().map(() => Array(8).fill().map(() => Math.random() * 2 - 1)),
                    Array(1).fill().map(() => Array(8).fill().map(() => Math.random() * 2 - 1))
                ];
                this.layout.updateWeights(demoWeights);
            }

            this.renderer.render(this.layout);
            this.updateMatrixTables();
        }, 500);
    }
}
