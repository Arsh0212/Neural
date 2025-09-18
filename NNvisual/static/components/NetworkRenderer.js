// Simplified NetworkRenderer Component with Grayscale Node Styling
class NetworkRenderer {
    constructor(svgSelector) {
        this.svg = d3.select(svgSelector);
        this.connectionsGroup = this.svg.append('g').attr('class', 'connections-layer');
        this.nodesGroup = this.svg.append('g').attr('class', 'nodes-layer');
        this.labelsGroup = this.svg.append('g').attr('class', 'labels-layer');
        this.controlsGroup = this.svg.append('g').attr('class', 'controls-layer');
        
        this.createGradients();
        this.addStyles();
        this.createControls();
    }

    createGradients() {
        const defs = this.svg.append('defs');
        
        // Simplified grayscale gradients based on activation intensity
        const gradients = [
            ['nodeVeryHigh', '#1f2937', '#111827'],    // Very dark gray (highest activation)
            ['nodeHigh', '#374151', '#1f2937'],        // Dark gray
            ['nodeMedium', '#6b7280', '#4b5563'],      // Medium gray
            ['nodeLow', '#9ca3af', '#6b7280'],         // Light gray
            ['nodeVeryLow', '#e5e7eb', '#d1d5db']      // Very light gray (lowest activation)
        ];

        gradients.forEach(([id, color1, color2]) => {
            const grad = defs.append('radialGradient').attr('id', id)
                .attr('cx', '25%').attr('cy', '25%').attr('r', '75%');
            grad.append('stop').attr('offset', '0%').attr('stop-color', color1).attr('stop-opacity', 1);
            grad.append('stop').attr('offset', '70%').attr('stop-color', color2).attr('stop-opacity', 0.9);
            grad.append('stop').attr('offset', '100%').attr('stop-color', color2).attr('stop-opacity', 1);
        });

        // Connection gradients
        const connGrads = [
            ['connectionPositive', '#3b82f6', '#1e40af'],
            ['connectionNegative', '#ef4444', '#b91c1c']
        ];

        connGrads.forEach(([id, color1, color2]) => {
            const grad = defs.append('linearGradient').attr('id', id);
            grad.append('stop').attr('offset', '0%').attr('stop-color', color1).attr('stop-opacity', 0.8);
            grad.append('stop').attr('offset', '100%').attr('stop-color', color2).attr('stop-opacity', 0.6);
        });

        // Subtle glow filter
        const glow = defs.append('filter').attr('id', 'glow')
            .attr('x', '-50%').attr('y', '-50%').attr('width', '200%').attr('height', '200%');
        glow.append('feGaussianBlur').attr('stdDeviation', '2').attr('result', 'coloredBlur');
        const merge = glow.append('feMerge');
        merge.append('feMergeNode').attr('in', 'coloredBlur');
        merge.append('feMergeNode').attr('in', 'SourceGraphic');

        // Soft shadow filter
        const shadow = defs.append('filter').attr('id', 'softShadow')
            .attr('x', '-50%').attr('y', '-50%').attr('width', '200%').attr('height', '200%');
        shadow.append('feDropShadow')
            .attr('dx', '1').attr('dy', '2')
            .attr('stdDeviation', '1.5')
            .attr('flood-color', '#000000')
            .attr('flood-opacity', '0.15');
    }

    addStyles() {
        if (!document.getElementById('networkStyles')) {
            const style = document.createElement('style');
            style.id = 'networkStyles';
            style.textContent = `
                .connection { transition: all 0.2s ease-out; }
                .node { cursor: pointer; transition: all 0.2s ease-out; }
                .node-circle { 
                    transition: all 0.25s cubic-bezier(0.25, 0.46, 0.45, 0.94);
                    filter: url(#softShadow);
                }
                .node-ring {
                    transition: all 0.3s ease;
                    stroke-dasharray: 3 2;
                    animation: rotate 8s linear infinite;
                }
                .control-button {
                    cursor: pointer;
                    transition: all 0.2s ease;
                }
                .control-button:hover .button-bg {
                    fill: #374151;
                    stroke: #6b7280;
                }
                .control-button:hover .button-text {
                    fill: #f9fafb;
                }
                @keyframes rotate {
                    from { transform: rotate(0deg); }
                    to { transform: rotate(360deg); }
                }
            `;
            document.head.appendChild(style);
        }
    }

    setViewBox(width, height) {
        this.svg.attr('viewBox', `0 0 ${width} ${height}`);
        // Position the button in the bottom left
        this.positionControls(width, height);
    }

    createControls() {
        // Create button group
        const buttonGroup = this.controlsGroup.append('g')
            .attr('class', 'control-button')
            .style('cursor', 'pointer');

        // Button background
        buttonGroup.append('rect')
            .attr('class', 'button-bg')
            .attr('width', 80)
            .attr('height', 30)
            .attr('rx', 6)
            .attr('ry', 6)
            .style('fill', '#4b5563')
            .style('stroke', '#6b7280')
            .style('stroke-width', 1);

        // Button text
        buttonGroup.append('text')
            .attr('class', 'button-text')
            .attr('x', 40)
            .attr('y', 20)
            .attr('text-anchor', 'middle')
            .style('fill', '#e5e7eb')
            .style('font', '12px system-ui')
            .style('pointer-events', 'none')
            .text('Help');

        // Button click handler
        buttonGroup.on('click', () => {
            console.log('Navigating to another page...');
            this.navigateToPage();
        });
    }

    positionControls(width, height) {
        // Position button at bottom left with some margin
        this.controlsGroup.select('.control-button')
            .attr('transform', `translate(20, ${height - 50})`);
    }

    navigateToPage() {
        // Use custom URL if set, otherwise default to '/dashboard'
        const url = this.navigationUrl || '/blog';
        window.location.href = url;
    }

    // Override this method to customize navigation
    setNavigationUrl(url) {
        this.navigationUrl = url;
    }

    getConnectionProps(weight) {
        const abs = Math.abs(weight);
        return {
            color: weight > 0 ? 'url(#connectionPositive)' : 'url(#connectionNegative)',
            width: Math.max(1, Math.min(4, abs * 1)),
            opacity: Math.max(0.3, Math.min(1, abs * 1.5)),
            glow: abs > 0.5
        };
    }

    renderConnections(connections) {
        const lines = this.connectionsGroup
            .selectAll('line.connection')
            .data(connections, d => d.id);

        lines.exit().transition().duration(300).style('opacity', 0).remove();

        const enterLines = lines.enter().append('line')
            .attr('class', 'connection').style('opacity', 0);

        const allLines = enterLines.merge(lines);

        allLines.transition().duration(200)
            .attr('x1', d => d.source.x).attr('y1', d => d.source.y)
            .attr('x2', d => d.target.x).attr('y2', d => d.target.y)
            .style('stroke', d => this.getConnectionProps(d.weight).color)
            .style('stroke-width', d => this.getConnectionProps(d.weight).width)
            .style('opacity', d => this.getConnectionProps(d.weight).opacity)
            .style('filter', d => this.getConnectionProps(d.weight).glow ? 'url(#glow)' : 'none');

        // Hover effects
        allLines
            .on('mouseover', (event, d) => {
                const props = this.getConnectionProps(d.weight);
                d3.select(event.target)
                    .style('stroke-width', props.width + 2)
                    .style('opacity', 1)
                    .style('filter', 'url(#glow)');
            })
            .on('mouseout', (event, d) => {
                const props = this.getConnectionProps(d.weight);
                d3.select(event.target)
                    .style('stroke-width', props.width)
                    .style('opacity', props.opacity)
                    .style('filter', props.glow ? 'url(#glow)' : 'none');
            });
    }

    getNodeProps(activation) {
        const abs = Math.abs(activation);
        let gradient, strokeColor, radius;
        
        // Fixed radius - no more size changes
        radius = 16;
        
        // Simplified grayscale scheme based on activation level
        // Higher activation = darker gray, Lower activation = lighter gray
        if (abs > 0.8) {
            gradient = 'url(#nodeVeryHigh)';
            strokeColor = '#111827';
        } else if (abs > 0.6) {
            gradient = 'url(#nodeHigh)';
            strokeColor = '#1f2937';
        } else if (abs > 0.4) {
            gradient = 'url(#nodeMedium)';
            strokeColor = '#4b5563';
        } else if (abs > 0.2) {
            gradient = 'url(#nodeLow)';
            strokeColor = '#6b7280';
        } else {
            gradient = 'url(#nodeVeryLow)';
            strokeColor = '#9ca3af';
        }

        return {
            fill: gradient,
            stroke: strokeColor,
            radius: radius,
            strokeWidth: 2,
            opacity: Math.max(0.8, Math.min(1, 0.6 + abs * 0.4)),
            glow: abs > 0.7,
            showRing: abs > 0.6
        };
    }

    renderNodes(layers) {
        const allNodes = layers.flatMap(layer => layer.nodes);
        const nodeGroups = this.nodesGroup
            .selectAll('g.node')
            .data(allNodes, d => d.id);

        nodeGroups.exit().transition().duration(300)
            .style('opacity', 0)
            .attr('transform', d => `translate(${d.x}, ${d.y}) scale(0)`)
            .remove();

        const enterGroups = nodeGroups.enter().append('g')
            .attr('class', 'node')
            .style('opacity', 0)
            .attr('transform', d => `translate(${d.x}, ${d.y}) scale(0)`);

        // Add main circle
        enterGroups.append('circle').attr('class', 'node-circle').attr('r', 0);
        
        // Add animated ring for highly active nodes
        enterGroups.append('circle').attr('class', 'node-ring')
            .attr('r', 20).attr('fill', 'none')
            .style('stroke', '#4b5563').style('stroke-width', 1.5)
            .style('opacity', 0);
        
        // Add subtle inner highlight
        enterGroups.append('circle').attr('class', 'node-highlight')
            .attr('r', 6).attr('cx', -4).attr('cy', -4)
            .style('fill', 'rgba(255, 255, 255, 0.3)')
            .style('opacity', 0);
        
        // Add text for hover
        enterGroups.append('text').attr('class', 'node-value')
            .attr('text-anchor', 'middle').attr('dy', '0.35em')
            .style('font', 'bold 10px system-ui')
            .style('fill', 'white').style('opacity', 0)
            .style('pointer-events', 'none')
            .style('text-shadow', '1px 1px 2px rgba(0,0,0,0.7)');

        const allGroups = enterGroups.merge(nodeGroups);

        allGroups.transition().duration(300)
            .style('opacity', 1)
            .attr('transform', d => `translate(${d.x}, ${d.y}) scale(1)`);

        // Update main circles
        allGroups.select('.node-circle').transition().duration(200)
            .attr('r', d => this.getNodeProps(d.activation).radius)
            .style('fill', d => this.getNodeProps(d.activation).fill)
            .style('stroke', d => this.getNodeProps(d.activation).stroke)
            .style('stroke-width', d => this.getNodeProps(d.activation).strokeWidth)
            .style('opacity', d => this.getNodeProps(d.activation).opacity)
            .style('filter', d => this.getNodeProps(d.activation).glow ? 'url(#glow)' : 'url(#softShadow)');

        // Update animated rings with grayscale color
        allGroups.select('.node-ring').transition().duration(200)
            .style('opacity', d => this.getNodeProps(d.activation).showRing ? 0.4 : 0)
            .style('stroke', d => this.getNodeProps(d.activation).stroke);

        // Update highlights
        allGroups.select('.node-highlight').transition().duration(200)
            .style('opacity', d => Math.abs(d.activation) > 0.3 ? 0.5 : 0);

        // Improved hover effects
        allGroups
            .on('mouseover', (event, d) => {
                const group = d3.select(event.currentTarget);
                group.transition().duration(150)
                    .attr('transform', `translate(${d.x}, ${d.y}) scale(1.15)`);
                
                group.select('.node-value')
                    .text(d.activation.toFixed(3))
                    .transition().duration(150).style('opacity', 1);
                
                group.select('.node-circle')
                    .style('filter', 'url(#glow)')
                    .style('stroke-width', 3);
                
                group.select('.node-ring')
                    .style('opacity', 0.6)
                    .style('stroke-width', 2);
                
                this.highlightConnections(d.id);
            })
            .on('mouseout', (event, d) => {
                const group = d3.select(event.currentTarget);
                const props = this.getNodeProps(d.activation);
                
                group.transition().duration(150)
                    .attr('transform', `translate(${d.x}, ${d.y}) scale(1)`);
                
                group.select('.node-value')
                    .transition().duration(150).style('opacity', 0);
                
                group.select('.node-circle')
                    .style('filter', props.glow ? 'url(#glow)' : 'url(#softShadow)')
                    .style('stroke-width', props.strokeWidth);
                
                group.select('.node-ring')
                    .style('opacity', props.showRing ? 0.4 : 0)
                    .style('stroke-width', 1.5);
                
                this.resetConnections();
            });
    }

    highlightConnections(nodeId) {
        this.connectionsGroup.selectAll('.connection')
            .style('opacity', d => d.source.id === nodeId || d.target.id === nodeId ? 1 : 0.2)
            .style('stroke-width', d => {
                const isConnected = d.source.id === nodeId || d.target.id === nodeId;
                return this.getConnectionProps(d.weight).width + (isConnected ? 1 : 0);
            });
    }

    resetConnections() {
        this.connectionsGroup.selectAll('.connection')
            .style('opacity', d => this.getConnectionProps(d.weight).opacity)
            .style('stroke-width', d => this.getConnectionProps(d.weight).width);
    }

    renderLabels(layers) {
        const labels = this.labelsGroup
            .selectAll('text.layer-label')
            .data(layers, d => d.name);

        labels.exit().transition().duration(300).style('opacity', 0).remove();

        const enterLabels = labels.enter().append('text')
            .attr('class', 'layer-label').style('opacity', 0);

        const allLabels = enterLabels.merge(labels);

        allLabels.transition().duration(300)
            .attr('x', d => d.nodes[0]?.x || 0)
            .attr('y', 25)
            .style('opacity', 1)
            .text(d => d.name)
            .style('font', '600 14px system-ui')
            .style('fill', 'var(--text-secondary)')
            .style('text-anchor', 'middle');
    }

    render(layout) {
        this.renderConnections(layout.connections);
        this.renderNodes(layout.layers);
        this.renderLabels(layout.layers);
    }
}