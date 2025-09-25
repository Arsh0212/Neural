{/* <script id="NetworkLayout.js"> */}
class NetworkLayout {
    constructor() {
        this.layers = [
            { name: 'Input (2)', count: 2, nodes: [], activations: [-1, -1] },
            { name: 'Hidden 1 (8)', count: 8, nodes: [], activations: Array(8).fill(-1) },
            { name: 'Hidden 2 (8)', count: 8, nodes: [], activations: Array(8).fill(-1) },
            { name: 'Output (1)', count: 1, nodes: [], activations: [-1] }
                    ];
        this.connections = [];
        this.weights = [];
    }

    calculateLayout(width, height) {
        const padding = 80;
        const layerSpacing = (width - 2 * padding) / Math.max(1, this.layers.length - 1);
        const maxNodes = Math.max(...this.layers.map(layer => layer.count));
        const maxNodeSpacing = Math.min(60, (height - 2 * padding) / Math.max(1, maxNodes - 1));
        
        this.layers.forEach((layer, i) => {
            layer.nodes = [];
            const x = padding + i * layerSpacing;
            
            let nodeSpacing, startY;
            
            if (layer.count === 1) {
                startY = height / 2;
                nodeSpacing = 0;
            } else {
                const layerHeight = Math.min(
                    height - 2 * padding,
                    (layer.count - 1) * maxNodeSpacing
                );
                nodeSpacing = layerHeight / Math.max(1, layer.count - 1);
                startY = (height - layerHeight) / 2;
            }
            
            for (let j = 0; j < layer.count; j++) {
                const y = layer.count === 1 ? startY : startY + j * nodeSpacing;
                layer.nodes.push({
                    id: `${i}-${j}`,
                    x, y,
                    activation: 0,
                    targetActivation: 0
                });
            }
        });
    }

    createConnections() {
        this.connections = [];
        for (let i = 0; i < this.layers.length - 1; i++) {
            const current = this.layers[i];
            const next = this.layers[i + 1];
            
            current.nodes.forEach(source => {
                next.nodes.forEach(target => {
                    this.connections.push({
                        id: `${source.id}-${target.id}`,
                        source, target,
                        weight: Math.random() * 2 - 1
                    });
                });
            });
        }
    }

    updateActivations(layerIndex, activations) {
        if (layerIndex >= this.layers.length) return;
        
        const layer = this.layers[layerIndex];
        layer.activations = [...activations];
        
        layer.nodes.forEach((node, i) => {
            if (i < activations[0].length) {
                node.targetActivation = activations[0][i];
                const diff = node.targetActivation - node.activation[0];
                node.activation = activations[0][i]; // Immediate update
            }
        });
    }

    updateWeights(weights) {
        this.weights = weights;
        let connIndex = 0;
        
        for (let i = 1; i < weights.length && i < this.layers.length; i++) {
            const layerWeights = weights[i];
            if (Array.isArray(layerWeights) && Array.isArray(layerWeights[0])) {
                layerWeights.forEach(nodeWeights => {
                    nodeWeights.forEach(weight => {
                        if (connIndex < this.connections.length) {
                            this.connections[connIndex].weight = weight;
                            connIndex++;
                        }
                    });
                });
            }
        }
    }
}
