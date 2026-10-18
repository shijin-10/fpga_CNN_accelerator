
import os
import subprocess
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import datasets

PROJECT = os.path.abspath(".")
IMAGE_INDEX = 726
INPUT_SCALE = 45.011730688743526

IMAGE_MEM = os.path.join(PROJECT, "mnist_image.mem")
OUTPUT_MEM = os.path.join(PROJECT, "conv1_pool_output.mem")
SIM_EXE = os.path.join(PROJECT, "conv1_mnist_pool_test")

def read_hex_signed(filename, bits):
    mask = (1 << bits) - 1
    sign_bit = 1 << (bits - 1)
    values = []

    with open(filename) as f:
        for line in f:
            line = line.strip()
            if line:
                value = int(line, 16) & mask
                if value >= sign_bit:
                    value -= 1 << bits
                values.append(value)

    return np.asarray(values, dtype=np.int64)

# Load MNIST image 726 and quantize it exactly as the full test does.
dataset = datasets.MNIST(
    root="./data", train=False, download=False
)
image, label = dataset[IMAGE_INDEX]

pixels = np.asarray(image, dtype=np.float32) / 255.0
normalized = (pixels - 0.1307) / 0.3081
quantized = np.clip(
    np.round(normalized * INPUT_SCALE), -128, 127
).astype(np.int32)

print(f"MNIST index: {IMAGE_INDEX}")
print(f"Actual label: {label}")
print(f"Quantized range: {quantized.min()} to {quantized.max()}")

# Back up files so the existing project data is preserved.
def backup_file(path):
    if os.path.exists(path):
        with open(path, "rb") as f:
            return f.read()
    return None

old_image = backup_file(IMAGE_MEM)
old_output = backup_file(OUTPUT_MEM)

try:
    # Write the input for the RTL testbench.
    with open(IMAGE_MEM, "w") as f:
        for value in quantized.flatten():
            f.write(f"{int(value) & 0xFF:02X}\n")

    # Run the actual Conv1 + ReLU + MaxPool testbench.
    subprocess.run(
        ["vvp", SIM_EXE],
        cwd=PROJECT,
        check=True
    )

    # Calculate the same stage in Python from the exact input and
    # the exported quantized Conv1 parameters.
    image_q = quantized.reshape(1, 1, 28, 28)
    weights_q = read_hex_signed(
        os.path.join(PROJECT, "conv1_weights.mem"), 8
    ).reshape(8, 1, 3, 3)
    biases_q = read_hex_signed(
        os.path.join(PROJECT, "conv1_bias_int32.mem"), 32
    )

    x = torch.tensor(image_q, dtype=torch.float64)
    w = torch.tensor(weights_q, dtype=torch.float64)
    b = torch.tensor(biases_q, dtype=torch.float64)

    conv = F.conv2d(x, w, b, padding=1)
    pool = F.max_pool2d(F.relu(conv), 2, 2)
    python_output = pool.to(torch.int64).numpy().reshape(-1)

    verilog_output = np.loadtxt(
        OUTPUT_MEM, dtype=np.int64
    ).reshape(-1)

    if len(python_output) != 1568 or len(verilog_output) != 1568:
        raise ValueError(
            f"Expected 1568 outputs; Python={len(python_output)}, "
            f"Verilog={len(verilog_output)}"
        )

    diff = np.abs(python_output - verilog_output)
    bad = np.flatnonzero(diff)

    print("\n========== IMAGE 726 CONV1/POOL1 CHECK ==========")
    print("Python outputs:", len(python_output))
    print("Verilog outputs:", len(verilog_output))
    print("Mismatches:", len(bad))
    print("Maximum absolute difference:", int(diff.max()))

    print("\nIndex | Python | Verilog")
    for i in range(min(12, len(python_output))):
        print(f"{i:5d} | {python_output[i]:6d} | {verilog_output[i]:7d}")

    if len(bad):
        print("\nFirst mismatches:")
        for i in bad[:10]:
            print(
                f"{i:5d} | {python_output[i]:6d} | "
                f"{verilog_output[i]:7d} | diff={diff[i]}"
            )
    else:
        print("\nPASS: All Conv1 + ReLU + MaxPool outputs match.")

finally:
    # Restore the original files.
    if old_image is None:
        if os.path.exists(IMAGE_MEM):
            os.remove(IMAGE_MEM)
    else:
        with open(IMAGE_MEM, "wb") as f:
            f.write(old_image)

    if old_output is None:
        if os.path.exists(OUTPUT_MEM):
            os.remove(OUTPUT_MEM)
    else:
        with open(OUTPUT_MEM, "wb") as f:
            f.write(old_output)

    print("\nOriginal memory files restored.")
