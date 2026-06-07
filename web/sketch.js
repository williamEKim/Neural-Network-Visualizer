// ── Canvas references ──
const drawCanvas = document.getElementById('draw-canvas');
const netCanvas  = document.getElementById('net-canvas');
const drawCtx    = drawCanvas.getContext('2d');
const netCtx     = netCanvas.getContext('2d');

// ── Sidebar toggle ──
const hamburgerBtn = document.getElementById('hamburger-btn');
const sidebar      = document.getElementById('sidebar');
const sidebarClose = document.getElementById('sidebar-close');
const overlay      = document.getElementById('overlay');

function openSidebar()  { sidebar.classList.add('open');  overlay.classList.add('visible'); }
function closeSidebar() { sidebar.classList.remove('open'); overlay.classList.remove('visible'); }

hamburgerBtn.addEventListener('click', openSidebar);
sidebarClose.addEventListener('click', closeSidebar);
overlay.addEventListener('click', closeSidebar);

// ── Digit buttons (0–9) ──
const digitGrid = document.getElementById('digit-grid');
for (let i = 0; i < 10; i++) {
  const btn = document.createElement('button');
  btn.className = 'digit-btn';
  btn.textContent = i;
  btn.addEventListener('click', () => {
    loadSample(i);
    closeSidebar();
  });
  digitGrid.appendChild(btn);
}

// ── Confidence bars (0–9) ──
const confidenceRow = document.getElementById('confidence-row');
const confBars = [];
for (let i = 0; i < 10; i++) {
  const wrap = document.createElement('div');
  wrap.className = 'conf-bar-wrap';
  const bar = document.createElement('div');
  bar.className = 'conf-bar';
  bar.style.height = '2px';
  const label = document.createElement('span');
  label.className = 'conf-digit';
  label.textContent = i;
  wrap.appendChild(bar);
  wrap.appendChild(label);
  confidenceRow.appendChild(wrap);
  confBars.push(bar);
}

// ── State ──
let pixels       = new Float32Array(784).fill(0);
let weights      = null;
let samples      = null;
let isDrawing    = false;
let hoveredNeuron = null; // { layerIdx, neuronIdx } or null
let pixelHistory = []; // stack of Float32Array snapshots

// ── Flag ──
let strokeStarted = false;

// ── Load weights ──
async function loadWeights() {
  const res = await fetch('../model/weights.json');
  weights = await res.json();
  console.log('Weights loaded:', weights.layers.length, 'layers');
  runNetwork();
}

// ── Load samples ──
async function loadSamples() {
  const res = await fetch('../samples.json');
  samples = await res.json();
  console.log('Samples loaded:', samples.length);
}

// ── Load a digit sample ──
function loadSample(digit) {
  if (!samples) return;
  const pool = samples.filter(s => s.label === digit);
  const pick = pool[Math.floor(Math.random() * pool.length)];
  pixels = new Float32Array(pick.pixels);
  renderDrawCanvas();
  runNetwork();
}

// ── Draw canvas rendering ──
const CELL = 10;
function renderDrawCanvas() {
  for (let i = 0; i < 784; i++) {
    const x = (i % 28) * CELL;
    const y = Math.floor(i / 28) * CELL;
    const v = Math.floor(pixels[i] * 255);
    drawCtx.fillStyle = `rgb(${v},${v},${v})`;
    drawCtx.fillRect(x, y, CELL, CELL);
  }
}

// ── Drawing input ──
function getPixelCoords(e) {
  const rect = drawCanvas.getBoundingClientRect();
  const scaleX = drawCanvas.width  / rect.width;
  const scaleY = drawCanvas.height / rect.height;
  const cx = e.touches ? e.touches[0].clientX : e.clientX;
  const cy = e.touches ? e.touches[0].clientY : e.clientY;
  return {
    col: Math.floor((cx - rect.left) * scaleX / CELL),
    row: Math.floor((cy - rect.top)  * scaleY / CELL),
  };
}

function paintAt(col, row) {
    const radius = 1;
    for (let dr = -radius; dr <= radius; dr++) {
        for (let dc = -radius; dc <= radius; dc++) {
            const r = row + dr, c = col + dc;
            if (r < 0 || r >= 28 || c < 0 || c >= 28) continue;
                const dist = Math.sqrt(dr*dr + dc*dc);
                const strength = Math.max(0, 1 - dist / (radius + 0.5));
                const idx = r * 28 + c;
                pixels[idx] = Math.min(1, pixels[idx] + strength * 0.9);
            }
    }
    renderDrawCanvas();
    runNetwork();
}

drawCanvas.addEventListener('mousedown', e => { isDrawing = true; const {col,row} = getPixelCoords(e); pixelHistory.push(pixels.slice()); strokeStarted = true; paintAt(col,row); });
drawCanvas.addEventListener('mousemove', e => { if (!isDrawing) return; const {col,row} = getPixelCoords(e); paintAt(col,row); });
drawCanvas.addEventListener('mouseup',   () => {strokeStarted = false; isDrawing = false;});
drawCanvas.addEventListener('mouseleave',() => {strokeStarted = false; isDrawing = false;});
drawCanvas.addEventListener('touchstart', e => { e.preventDefault(); isDrawing = true; const {col,row} = getPixelCoords(e); paintAt(col,row); }, { passive: false });
drawCanvas.addEventListener('touchmove',  e => { e.preventDefault(); if (!isDrawing) return; const {col,row} = getPixelCoords(e); paintAt(col,row); }, { passive: false });
drawCanvas.addEventListener('touchend',   () => isDrawing = false);

// ── Controls ──
document.getElementById('clear-btn').addEventListener('click', () => {
    pixels.fill(0);
    renderDrawCanvas();
    runNetwork();
});

document.getElementById('random-btn').addEventListener('click', () => {
    const digit = Math.floor(Math.random() * 10);
    loadSample(digit);
    closeSidebar();
});

document.addEventListener('keydown', e => {
    if ((e.metaKey || e.ctrlKey) && e.key === 'z') {
        e.preventDefault();
        if (pixelHistory.length > 0) {
            pixels = pixelHistory.pop();
            renderDrawCanvas();
            runNetwork();
        }
    }
});

// ── Forward pass ──
function sigmoid(z) { return 1 / (1 + Math.exp(-z)); }

function softmax(z) {
    const max = Math.max(...z);
    const exps = z.map(v => Math.exp(v - max));
    const sum  = exps.reduce((a,b) => a+b, 0);
    return exps.map(v => v/sum);
}

function forward(x) {
    if (!weights) return null;
    const activations = [Array.from(x)];
    let a = Array.from(x);
    weights.layers.forEach((layer, idx) => {
        const W = layer.weights;
        const b = layer.biases;
        const z = W.map((row, i) => row.reduce((sum, w, j) => sum + w * a[j], 0) + b[i]);
        const isLast = idx === weights.layers.length - 1;
        a = isLast ? softmax(z) : z.map(sigmoid);
        activations.push(a);
    });
    return activations;
}

// ── Layout helpers ──
function getLayout(W, H) {
    const displaySizes  = [784, 16, 16, 10];  // skip 100 for display
    const activationIdx = [0, 2, 3, 4];
    const xPositions    = [W * 0.18, W * 0.40, W * 0.60, W * 0.80];
    const radii         = [null, 6, 6, 8];
    const cellSize      = 8;
    const gridW         = cellSize * 28;
    const gridH         = cellSize * 28;
    const gridStartX    = xPositions[0] - gridW / 2;
    const gridStartY    = H / 2 - gridH / 2;
    return { displaySizes, activationIdx, xPositions, radii, cellSize, gridW, gridH, gridStartX, gridStartY };
}

// ── Effective weight propagation ──
function getEffectiveWeights(layerIdx, neuronIdx) {
    const wLayers = weights.layers;

    if (layerIdx === 1) {
        const w1 = wLayers[1].weights[neuronIdx]; // length 100
        return wLayers[0].weights[0].map((_, pixelIdx) =>
            w1.reduce((sum, w, j) => sum + w * wLayers[0].weights[j][pixelIdx], 0)
        );
    }

    if (layerIdx === 2) {
        const w2 = wLayers[2].weights[neuronIdx]; // length 16
        const w1eff = wLayers[1].weights[0].map((_, j100) =>
            w2.reduce((sum, w, j16) => sum + w * wLayers[1].weights[j16][j100], 0)
        );
        return wLayers[0].weights[0].map((_, pixelIdx) =>
            w1eff.reduce((sum, w, j) => sum + w * wLayers[0].weights[j][pixelIdx], 0)
        );
    }

    if (layerIdx === 3) {
        const w3 = wLayers[3].weights[neuronIdx]; // length 16
        const w2eff = wLayers[2].weights[0].map((_, j50) =>
            w3.reduce((sum, w, j16) => sum + w * wLayers[2].weights[j16][j50], 0)
        );
        const w1eff = wLayers[1].weights[0].map((_, j100) =>
            w2eff.reduce((sum, w, j16) => sum + w * wLayers[1].weights[j16][j100], 0)
        );
        return wLayers[0].weights[0].map((_, pixelIdx) =>
            w1eff.reduce((sum, w, j) => sum + w * wLayers[0].weights[j][pixelIdx], 0)
        );
    }
}

// ── Draw faint connection lines ──
function drawConnections(H, layout, showWeights) {
    const { displaySizes, xPositions } = layout;
    const pairs = [[0,1], [1,2], [2,3]];

    pairs.forEach(([fromIdx, toIdx]) => {
        const fromSize  = displaySizes[fromIdx];
        const toSize    = displaySizes[toIdx];
        const fromX     = xPositions[fromIdx];
        const toX       = xPositions[toIdx];
        const toSpacing = H / (toSize + 1);

        netCtx.lineWidth = 0.3;

        for (let j = 0; j < toSize; j++) {
        const toY = toSpacing * (j + 1);

        if (fromIdx === 0) {
            netCtx.strokeStyle = 'rgba(255,255,255,0.15)';
            netCtx.beginPath();
            netCtx.moveTo(fromX, H / 2);
            netCtx.lineTo(toX, toY);
            netCtx.stroke();
        } else {
                const fromSpacing = H / (fromSize + 1);
                for (let i = 0; i < fromSize; i++) {
                    const fromY = fromSpacing * (i + 1);
                    
                    const isHoveredConnection = hoveredNeuron && (
                        (hoveredNeuron.layerIdx === toIdx && hoveredNeuron.neuronIdx === j) ||  // incoming
                        (hoveredNeuron.layerIdx === fromIdx && hoveredNeuron.neuronIdx === i)   // outgoing
                    );

                    if (isHoveredConnection && showWeights) {
                        // get weight value
                        const weightLayerIdx = fromIdx; // which weights.layers index?
                        const w = weights.layers[weightLayerIdx].weights[j][i];
                        const mag = Math.min(Math.abs(w) * 3, 1); // scale magnitude
                        const r = w < 0 ? Math.floor(mag * 180) : 0;
                        const g = w > 0 ? Math.floor(mag * 230) : 0;
                        netCtx.strokeStyle = `rgba(${r},${g},50,${0.3 + mag * 0.7})`;
                        netCtx.lineWidth = 1.5;
                    } else if (isHoveredConnection) {
                        netCtx.strokeStyle = 'rgba(255,255,255,0.4)';
                        netCtx.lineWidth = 1.5;
                    }else {
                        netCtx.strokeStyle = 'rgba(255,255,255,0.12)';
                        netCtx.lineWidth = 0.3;
                    }

                    netCtx.beginPath();
                    netCtx.moveTo(fromX, fromY);
                    netCtx.lineTo(toX, toY);
                    netCtx.stroke();
                }
            }
        }
    });
}

// ── Network visualization ──
function renderNetwork(activations) {
    const W = netCanvas.width;
    const H = netCanvas.height;
    netCtx.clearRect(0, 0, W, H);
    if (!activations) return;

    const layout = getLayout(W, H);
    const { displaySizes, activationIdx, xPositions, radii, cellSize, gridStartX, gridStartY } = layout;

    const showWeights = document.getElementById('weights-toggle').checked;
    drawConnections(H, layout, showWeights);

    displaySizes.forEach((size, layerIdx) => {
        const x      = xPositions[layerIdx];
        const actIdx = activationIdx[layerIdx];

        if (layerIdx === 0) {
        // draw base pixel grid
        for (let row = 0; row < 28; row++) {
            for (let col = 0; col < 28; col++) {
            const val = Math.floor(activations[actIdx][row * 28 + col] * 255);
            netCtx.fillStyle = `rgb(${val},${val},${val})`;
            netCtx.fillRect(gridStartX + col * cellSize, gridStartY + row * cellSize, cellSize, cellSize);
            }
        }

        // overlay weight colors on hover
        const showWeights = document.getElementById('weights-toggle').checked;
        if (hoveredNeuron && weights && showWeights) {
            const ew = getEffectiveWeights(hoveredNeuron.layerIdx, hoveredNeuron.neuronIdx);
            const maxW = Math.max(...ew.map(Math.abs));
            for (let row = 0; row < 28; row++) {
            for (let col = 0; col < 28; col++) {
                const idx = row * 28 + col;
                const w = ew[idx] / maxW;
                const r = w < 0 ? Math.floor(-w * 180) : 0;
                const g = w > 0 ? Math.floor(w  * 230) : 0;
                netCtx.fillStyle = `rgba(${r},${g},50,0.75)`;
                netCtx.fillRect(gridStartX + col * cellSize, gridStartY + row * cellSize, cellSize, cellSize);
            }
            }
        }

        } else {
            const spacing = H / (size + 1);
            for (let i = 0; i < size; i++) {
                const y          = spacing * (i + 1);
                const activation = activations[actIdx][i];
                const isHovered  = hoveredNeuron && hoveredNeuron.layerIdx === layerIdx && hoveredNeuron.neuronIdx === i;
                const r = Math.floor((1 - activation) * 180);
                const g = Math.floor(activation * 230);
                const b = 50;

                netCtx.beginPath();
                netCtx.arc(x, y, isHovered ? radii[layerIdx] * 1.8 : radii[layerIdx], 0, Math.PI * 2);
                netCtx.fillStyle = `rgb(${r},${g},${b})`;
                netCtx.fill();

                // white ring on hover
                if (isHovered) {
                    netCtx.beginPath();
                    netCtx.arc(x, y, radii[layerIdx] * 1.8 + 2, 0, Math.PI * 2);
                    netCtx.strokeStyle = 'rgba(255,255,255,0.8)';
                    netCtx.lineWidth = 1.5;
                    netCtx.stroke();
                }

                if (layerIdx === 3) {
                    netCtx.font = '10px Space Mono, monospace';
                    netCtx.fillStyle = '#888';
                    netCtx.textAlign = 'center';
                    netCtx.fillText(i, x, y + radii[3] + 12);
                }

                if (isHovered) {
                    // show hovered neuron's bias
                    const biasLayerIdx = layerIdx; 
                    const bias = weights.layers[biasLayerIdx].biases[i];
                    netCtx.font = '9px Space Mono, monospace';
                    netCtx.fillStyle = '#818cf8'; // indigo blue
                    netCtx.textAlign = 'left';
                    netCtx.fillText(`b:${bias.toFixed(2)}`, x + radii[layerIdx] * 1.8 + 10, y + 3);
                    if (layerIdx < 3) {
                        const nextSize    = displaySizes[layerIdx + 1];
                        const nextX      = xPositions[layerIdx + 1];
                        const nextSpacing = H / (nextSize + 1);
                        for (let j = 0; j < nextSize; j++) {
                            const nextY  = nextSpacing * (j + 1);
                            const nextBias = weights.layers[layerIdx + 1].biases[j];
                            netCtx.font = '10px Space Mono, monospace';
                            netCtx.fillStyle = '#818cf8';
                            netCtx.textAlign = 'left';
                            netCtx.fillText(`b:${nextBias.toFixed(2)}`, nextX + radii[layerIdx + 1] * 1.8 + 6, nextY + 3);
                        }
                    }
                }
            }

        }
    });
}

// ── Hover detection ──
netCanvas.addEventListener('mousemove', e => {
    const rect = netCanvas.getBoundingClientRect();
    const mx = e.clientX - rect.left;
    const my = e.clientY - rect.top;
    const W  = netCanvas.width;
    const H  = netCanvas.height;
    const layout = getLayout(W, H);

    let found = null;
    [1, 2, 3].forEach(layerIdx => {
        const x       = layout.xPositions[layerIdx];
        const size    = layout.displaySizes[layerIdx];
        const spacing = H / (size + 1);
        const r       = layout.radii[layerIdx];
        for (let i = 0; i < size; i++) {
            const y    = spacing * (i + 1);
            const dist = Math.sqrt((mx - x) ** 2 + (my - y) ** 2);
            if (dist < r + 6) found = { layerIdx, neuronIdx: i };
        }
    });

    const changed = JSON.stringify(found) !== JSON.stringify(hoveredNeuron);
    if (changed) {
        hoveredNeuron = found;
        runNetwork();
    }
});

netCanvas.addEventListener('mouseleave', () => {
    hoveredNeuron = null;
    runNetwork();
});

// ── Update prediction bar ──
function updatePrediction(activations) {
    if (!activations) return;
    const out  = activations[activations.length - 1];
    const pred = out.indexOf(Math.max(...out));
    document.getElementById('prediction-digit').textContent = pred;
    const maxH = 36;
    out.forEach((v, i) => {
        confBars[i].style.height = Math.max(2, Math.round(v * maxH)) + 'px';
    });
}

// ── Main run ──
function runNetwork() {
    const activations = forward(pixels);
    renderNetwork(activations);
    updatePrediction(activations);
}

// ── Resize net canvas ──
function resizeNetCanvas() {
    const wrap = netCanvas.parentElement;
    netCanvas.width  = wrap.clientWidth;
    netCanvas.height = wrap.clientHeight;
    runNetwork();
}
window.addEventListener('resize', resizeNetCanvas);

// ── Init ──
renderDrawCanvas();
loadWeights();
loadSamples();
requestAnimationFrame(() => requestAnimationFrame(resizeNetCanvas));