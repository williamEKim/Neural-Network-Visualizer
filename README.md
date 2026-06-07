# Neural Network Visualizer

An interactive neural network visualizer trained on MNIST from scratch — draw a digit, watch it think.

Built as a companion to [3Blue1Brown's neural network series](https://www.youtube.com/watch?v=aircAruvnKk), this project implements a fully connected neural network from the ground up using only NumPy, then visualizes every layer, weight, and bias in the browser.

<img width="1440" height="787" alt="스크린샷 2026-06-07 오전 9 00 01" src="https://github.com/user-attachments/assets/362637b3-7960-4277-ab26-5eb055aa0403" />

---

## Features

- **Draw digits** with your mouse or touchscreen and watch the network predict in real time
- **Weight visualization** — hover any neuron to see which pixels influenced it (green = positive, red = negative)
- **Connection highlighting** — incoming and outgoing connections colored by weight magnitude on hover
- **Bias display** — hover a neuron to see its bias and the biases of all connected neurons in blue
- **Activation colors** — neurons colored from dark red (low) to bright green (high activation)
- **Confidence bars** — output layer probabilities displayed for all 10 digits
- **Load MNIST samples** — pick any digit 0–9 from the sidebar
- **Cmd+Z undo** — undo strokes while drawing
- **Responsive** — works on desktop and mobile

<img width="1117" height="784" alt="image" src="https://github.com/user-attachments/assets/aab64c6a-83e9-4b45-a269-e3a63c31480c" />


---

## Architecture

```
784 → 100(Hidden in Web UI) → 16 → 16 → 10
```

- Input: 28×28 grayscale pixel values (0–1)
- Hidden layers: sigmoid activation, Xavier initialization
- Output: softmax over 10 digit classes
- Loss: cross-entropy
- Training: mini-batch gradient descent (batch size 32)
- Augmentation: random rotation ±15° during training

**Test accuracy: 94.29%**

---

## Project Structure

```
├── model/
│   ├── train.py          # network implementation + training loop
│   ├── export.py         # export sample images to JSON
│   ├── weights.json      # trained weights (generated)
│   └── requirements.txt
├── index.html
├── style.css
├── sketch.js         # forward pass + visualization
├── data/                 # MNIST dataset (downloaded automatically)
├── samples.json          # 100 sample images for the UI
└── run_model.sh          # setup + training script
```

---

## Getting Started

**Train the model:**

```bash
source run_model.sh
```

This will:
1. Create a virtual environment
2. Install dependencies
3. Train the network for 50 epochs
4. Export weights to `model/weights.json`
5. Export 100 sample images to `samples.json`

**Run the visualizer:**

Open `web/index.html` in your browser. No server needed.

---

## Implementation

The entire neural network is implemented from scratch in `model/train.py` using only NumPy:

- `forward()` — sigmoid hidden layers, softmax output
- `backward()` — backpropagation with chain rule
- `train()` — mini-batch gradient descent with shuffling
- `evaluate()` — accuracy on test set
- `save_weights()` — export to JSON for the browser

The browser-side forward pass in `sketch.js` mirrors the Python implementation exactly, so predictions are consistent.

---

## Inspiration

This project is inspired by [3Blue1Brown's](https://www.youtube.com/@3blue1brown) excellent series on neural networks:

- [But what is a neural network?](https://www.youtube.com/watch?v=aircAruvnKk)
- [Gradient descent, how neural networks learn](https://www.youtube.com/watch?v=IHZwWFHWa-w)
- [What is backpropagation really doing?](https://www.youtube.com/watch?v=Ilg3gGewQ5U)

---

## License

MIT © 2026 William Kim
