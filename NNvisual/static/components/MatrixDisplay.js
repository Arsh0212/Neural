// components/MatrixDisplay.js
class MatrixDisplay {
    constructor(activationsContainer, weightsContainer) {
        this.activationsContainer = document.getElementById(activationsContainer);
        this.weightsContainer = document.getElementById(weightsContainer);
    }

    updateActivationsMatrix(layers) {
        const table = document.createElement('table');
        table.className = 'matrix-table';

        const headerRow = table.insertRow();
        headerRow.insertCell().textContent = 'Layer';
        headerRow.insertCell().textContent = 'Node';
        headerRow.insertCell().textContent = 'Activation';

        layers.forEach((layer, i) => {
            // If activations is a nested list, use the first element
            const activations = Array.isArray(layer.activations[0]) ? layer.activations[0] : layer.activations;
            activations.forEach((activation, j) => {
                const row = table.insertRow();
                row.insertCell().textContent = i;
                row.insertCell().textContent = j;
                const cell = row.insertCell();
                cell.textContent = typeof activation === "number" ? activation.toFixed(3) : activation;
                cell.className = 'activation-cell';

                const intensity = Math.max(0, Math.min(1, activation));
                cell.style.color = rgb(${255 * intensity}, ${255 * intensity}, ${255 * intensity});
            });
        });

        this.activationsContainer.innerHTML = '';
        this.activationsContainer.appendChild(table);
    }

    updateWeightsMatrix(weights, layers) {
        if (!weights.length) {
            this.weightsContainer.innerHTML = '<div style="color: #666;">No weight data available</div>';
            return;
        }

        const div = document.createElement('div');

        for (let i = 1; i < weights.length && i < layers.length; i++) {
            const layerWeights = weights[i];
            if (!Array.isArray(layerWeights) || !Array.isArray(layerWeights[0])) continue;

            const title = document.createElement('div');
            title.textContent = Layer ${i-1} → ${i};
            title.style.cssText = 'font-weight: bold; margin: 10px 0 5px 0; color: #ccc;';
            div.appendChild(title);

            const table = document.createElement('table');
            table.className = 'matrix-table';
            table.style.marginBottom = '15px';

            layerWeights.forEach((nodeWeights, nodeIndex) => {
                const row = table.insertRow();
                const labelCell = row.insertCell();
                labelCell.textContent = N${nodeIndex};
                labelCell.style.background = '#f1e9e9ff';

                nodeWeights.forEach(weight => {
                    const cell = row.insertCell();
                    cell.textContent = weight.toFixed(2);
                    cell.className = 'weight-cell';

                    if (weight > 0.1) {
                        cell.classList.add('positive');
                    } else if (weight < -0.1) {
                        cell.classList.add('negative');
                    }
                });
            });

            div.appendChild(table);
        }

        this.weightsContainer.innerHTML = '';
        this.weightsContainer.appendChild(div);
    }
}
