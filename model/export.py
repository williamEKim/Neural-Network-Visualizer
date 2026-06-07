# model/export.py
import numpy as np
import json
import os
from torchvision import datasets

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, '..', 'data')

test_data = datasets.MNIST(root=DATA_DIR, train=False, download=False)

samples = []
counts = {}

for img, label in test_data:
    label = int(label)
    if counts.get(label, 0) < 10:
        pixels = (np.array(img) / 255.0).flatten().tolist()
        samples.append({"label": label, "pixels": pixels})
        counts[label] = counts.get(label, 0) + 1
    if len(samples) == 100:
        break

out = os.path.join(SCRIPT_DIR, '..', 'samples.json')
with open(out, 'w') as f:
    json.dump(samples, f)

print(f"Exported {len(samples)} samples to samples.json")