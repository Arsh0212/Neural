// components/MatrixDisplay.js
class MatrixDisplay {
    constructor(activationsContainer, weightsContainer) {
        this.activationsContainer = document.getElementById(activationsContainer);
        this.weightsContainer = document.getElementById(weightsContainer);
        
        // Cache for performance optimization
        this.lastActivationsData = null;
        this.lastWeightsData = null;
        this.activationCells = new Map();
        this.weightCells = new Map();
        
        // Pre-created elements to avoid DOM recreation
        this.activationsTable = null;
        this.weightsDiv = null;
        
        // Performance settings
        this.maxDisplayNodes = 50; // Limit displayed nodes for performance
        this.maxDisplayLayers = 10; // Limit displayed layers
    }

    updateActivationsMatrix(layers) {
        if (!layers || layers.length === 0) return;
        
        // Check if data actually changed
        const currentDataHash = this.hashActivationsData(layers);
        if (currentDataHash === this.lastActivationsData) {
            return; // No changes, skip update
        }
        this.lastActivationsData = currentDataHash;

        // Limit layers for performance
        const displayLayers = layers.slice(0, this.maxDisplayLayers);
        
        // Create table only once
        if (!this.activationsTable) {
            this.createActivationsTable();
        }

        this.updateActivationsTableData(displayLayers);
    }

    createActivationsTable() {
        this.activationsTable = document.createElement('table');
        this.activationsTable.className = 'matrix-table';
        
        const headerRow = this.activationsTable.insertRow();
        headerRow.insertCell().textContent = 'Layer';
        headerRow.insertCell().textContent = 'Node';
        headerRow.insertCell().textContent = 'Activation';
        
        this.activationsContainer.innerHTML = '';
        this.activationsContainer.appendChild(this.activationsTable);
    }

    updateActivationsTableData(layers) {
        // Clear existing data rows (keep header)
        const rowCount = this.activationsTable.rows.length;
        for (let i = rowCount - 1; i > 0; i--) {
            this.activationsTable.deleteRow(i);
        }
        
        // Clear cell cache
        this.activationCells.clear();
        
        // Use DocumentFragment for batch DOM operations
        const fragment = document.createDocumentFragment();
        const tempTable = document.createElement('table');
        
        layers.forEach((layer, layerIndex) => {
            if (!layer.activations) return;
            
            // Handle nested arrays
            const activations = Array.isArray(layer.activations[0]) ? 
                layer.activations[0] : layer.activations;
            
            // Limit nodes for performance
            const displayActivations = activations.slice(0, this.maxDisplayNodes);
            
            displayActivations.forEach((activation, nodeIndex) => {
                const row = tempTable.insertRow();
                
                const layerCell = row.insertCell();
                layerCell.textContent = layerIndex;
                
                const nodeCell = row.insertCell();
                nodeCell.textContent = nodeIndex;
                
                const activationCell = row.insertCell();
                this.updateActivationCell(activationCell, activation);
                
                // Cache cell reference
                const cellKey = `${layerIndex}-${nodeIndex}`;
                this.activationCells.set(cellKey, activationCell);
                
                fragment.appendChild(row);
            });
        });
        
        // Batch append all rows
        while (tempTable.rows.length > 0) {
            this.activationsTable.appendChild(tempTable.rows[0]);
        }
    }

    updateActivationCell(cell, activation) {
        const value = typeof activation === "number" ? activation : parseFloat(activation) || 0;
        cell.textContent = value.toFixed(3);
        cell.className = 'activation-cell';
        
        // Optimize color calculation
        const intensity = Math.max(0, Math.min(1, Math.abs(value)));
        const colorValue = Math.round(255 * (0.3 + intensity * 0.7)); // Better contrast
        cell.style.color = `rgb(${colorValue}, ${colorValue}, ${colorValue})`;
        
        // Add background color for better visualization
        if (value > 0.5) {
            cell.style.backgroundColor = `rgba(0, 255, 0, ${intensity * 0.3})`;
        } else if (value < -0.5) {
            cell.style.backgroundColor = `rgba(255, 0, 0, ${intensity * 0.3})`;
        } else {
            cell.style.backgroundColor = 'transparent';
        }
    }

    updateWeightsMatrix(weights, layers) {
        if (!weights || weights.length === 0) {
            this.weightsContainer.innerHTML = '<div style="color: #666;">No weight data available</div>';
            return;
        }
        
        // Check if data actually changed
        const currentDataHash = this.hashWeightsData(weights);
        if (currentDataHash === this.lastWeightsData) {
            return; // No changes, skip update
        }
        this.lastWeightsData = currentDataHash;

        // Create container only once
        if (!this.weightsDiv) {
            this.weightsDiv = document.createElement('div');
            this.weightsContainer.innerHTML = '';
            this.weightsContainer.appendChild(this.weightsDiv);
        }

        this.updateWeightsData(weights, layers);
    }

    updateWeightsData(weights, layers) {
        // Clear existing content
        this.weightsDiv.innerHTML = '';
        this.weightCells.clear();
        
        // Use DocumentFragment for performance
        const fragment = document.createDocumentFragment();
        
        // Limit layers for performance
        const maxLayers = Math.min(weights.length, layers.length, this.maxDisplayLayers);
        
        for (let i = 1; i < maxLayers; i++) {
            const layerWeights = weights[i];
            if (!Array.isArray(layerWeights) || !Array.isArray(layerWeights[0])) continue;
            
            const layerSection = this.createWeightLayerSection(layerWeights, i);
            fragment.appendChild(layerSection);
        }
        
        this.weightsDiv.appendChild(fragment);
    }

    createWeightLayerSection(layerWeights, layerIndex) {
        const section = document.createElement('div');
        section.className = 'weight-layer-section';
        
        const title = document.createElement('div');
        title.textContent = `Layer ${layerIndex-1} → ${layerIndex}`;
        title.style.cssText = 'font-weight: bold; margin: 10px 0 5px 0; color: #ccc;';
        section.appendChild(title);
        
        const table = document.createElement('table');
        table.className = 'matrix-table';
        table.style.marginBottom = '15px';
        
        // Limit nodes for performance
        const displayWeights = layerWeights.slice(0, this.maxDisplayNodes);
        
        displayWeights.forEach((nodeWeights, nodeIndex) => {
            const row = table.insertRow();
            
            const labelCell = row.insertCell();
            labelCell.textContent = `N${nodeIndex}`;
            labelCell.style.background = '#f1e9e9ff';
            labelCell.style.minWidth = '30px';
            
            // Limit weight connections displayed
            const displayNodeWeights = nodeWeights.slice(0, 20); // Max 20 connections per node
            
            displayNodeWeights.forEach((weight, weightIndex) => {
                const cell = row.insertCell();
                this.updateWeightCell(cell, weight);
                
                // Cache cell reference
                const cellKey = `${layerIndex}-${nodeIndex}-${weightIndex}`;
                this.weightCells.set(cellKey, cell);
            });
            
            // Add "..." indicator if there are more weights
            if (nodeWeights.length > 20) {
                const moreCell = row.insertCell();
                moreCell.textContent = '...';
                moreCell.style.color = '#666';
                moreCell.style.fontStyle = 'italic';
            }
        });
        
        // Add "..." indicator if there are more nodes
        if (layerWeights.length > this.maxDisplayNodes) {
            const moreRow = table.insertRow();
            const moreCell = moreRow.insertCell();
            moreCell.colSpan = Math.min(21, layerWeights[0]?.length + 1 || 1);
            moreCell.textContent = `... and ${layerWeights.length - this.maxDisplayNodes} more nodes`;
            moreCell.style.textAlign = 'center';
            moreCell.style.color = '#666';
            moreCell.style.fontStyle = 'italic';
        }
        
        section.appendChild(table);
        return section;
    }

    updateWeightCell(cell, weight) {
        const value = typeof weight === "number" ? weight : parseFloat(weight) || 0;
        cell.textContent = value.toFixed(3);
        cell.className = 'weight-cell';
        
        // Remove old classes
        cell.classList.remove('positive', 'negative', 'neutral');
        
        // Add appropriate class based on weight value
        if (value > 0.1) {
            cell.classList.add('positive');
        } else if (value < -0.1) {
            cell.classList.add('negative');
        } else {
            cell.classList.add('neutral');
        }
        
        // Add visual intensity based on absolute value
        const intensity = Math.min(1, Math.abs(value));
        cell.style.fontWeight = intensity > 0.5 ? 'bold' : 'normal';
    }

    // Helper methods for change detection
    hashActivationsData(layers) {
        if (!layers || layers.length === 0) return '';
        
        return layers.map(layer => {
            if (!layer.activations) return '';
            const activations = Array.isArray(layer.activations[0]) ? 
                layer.activations[0] : layer.activations;
            return activations.slice(0, 10).join(','); // Sample first 10 for hash
        }).join('|');
    }

    hashWeightsData(weights) {
        if (!weights || weights.length === 0) return '';
        
        return weights.map((layerWeights, i) => {
            if (!Array.isArray(layerWeights) || !Array.isArray(layerWeights[0])) return '';
            return layerWeights.slice(0, 5).map(nodeWeights => 
                nodeWeights.slice(0, 5).join(',')
            ).join(';');
        }).join('|');
    }

    // Performance monitoring
    getPerformanceStats() {
        return {
            activationCells: this.activationCells.size,
            weightCells: this.weightCells.size,
            maxDisplayNodes: this.maxDisplayNodes,
            maxDisplayLayers: this.maxDisplayLayers
        };
    }

    // Configuration methods
    setDisplayLimits(maxNodes, maxLayers) {
        this.maxDisplayNodes = maxNodes;
        this.maxDisplayLayers = maxLayers;
        
        // Force refresh on next update
        this.lastActivationsData = null;
        this.lastWeightsData = null;
    }

    // Cleanup method
    destroy() {
        this.activationCells.clear();
        this.weightCells.clear();
        this.activationsTable = null;
        this.weightsDiv = null;
        this.lastActivationsData = null;
        this.lastWeightsData = null;
    }
}
