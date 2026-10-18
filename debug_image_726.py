
import subprocess
import numpy as np
import torch
from torchvision import datasets

PROJECT = "."
INPUT_SCALE = 45.011730688743526
IMAGE_INDEX = 726

dataset = datasets.MNIST(
    root="./data",
    train=False,
    download=False
)

image, label = dataset[IMAGE_INDEX]

pixels = np.asarray(image, dtype=np.float32) / 255.0
normalized = (pixels - 0.1307) / 0.3081

quantized = np.round(normalized * INPUT_SCALE)
quantized = np.clip(quantized, -128, 127).astype(np.int32)

# Back up the existing input file.
backup_path = "mnist_image.mem.debug_backup"

try:
    with open("mnist_image.mem", "r") as source:
        old_contents = source.read()
except FileNotFoundError:
    old_contents = None

try:
    with open("mnist_image.mem", "w") as f:
        for value in quantized.flatten():
            f.write(f"{int(value) & 0xFF:02X}\n")

    print(f"MNIST index: {IMAGE_INDEX}")
    print(f"Actual label: {label}")
    print(f"Quantized input range: {quantized.min()} to {quantized.max()}")
    print("Running integrated debug simulation...\n")

    result = subprocess.run(
        ["vvp", "cnn_debug_sim"],
        cwd=PROJECT,
        text=True
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"Simulation failed with exit code {result.returncode}"
        )

finally:
    if old_contents is not None:
        with open("mnist_image.mem", "w") as f:
            f.write(old_contents)
    else:
        import os
        os.remove("mnist_image.mem")

print("\nOriginal mnist_image.mem restored.")
