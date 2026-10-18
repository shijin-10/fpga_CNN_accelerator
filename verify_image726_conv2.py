
import os
import subprocess
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import datasets

PROJECT = os.path.abspath(".")
IMAGE_INDEX = 726

INPUT_SCALE = 45.011730688743526
CONV1_WEIGHT_SCALE = 142.15939190898197
POOL1_SCALE = 18.25425611513592
CONV2_WEIGHT_SCALE = 241.5322203712151

RQ1_MULTIPLIER = 3063111
RQ1_SHIFT = 30

def read_hex_signed(filename, bits):
    values = []
    mask = (1 << bits) - 1
    sign_bit = 1 << (bits - 1)

    with open(filename) as f:
        for line in f:
            line = line.strip()
            if line:
                value = int(line, 16) & mask
                if value >= sign_bit:
                    value -= 1 << bits
                values.append(value)

    return np.asarray(values, dtype=np.int64)

def save_hex_signed(filename, values, bits):
    mask = (1 << bits) - 1
    with open(filename, "w") as f:
        for value in values:
            f.write(f"{int(value) & mask:0{bits // 4}X}\n")

def backup_file(path):
    if os.path.exists(path):
        with open(path, "rb") as f:
            return f.read()
    return None

def restore_file(path, data):
    if data is None:
        if os.path.exists(path):
            os.remove(path)
    else:
        with open(path, "wb") as f:
            f.write(data)

paths = [
    "mnist_image.mem",
    "conv1_pool_output.mem",
    "conv2_input.mem",
    "conv2_output.mem",
]

backups = {
    name: backup_file(os.path.join(PROJECT, name))
    for name in paths
}

try:
    # 1. Load and quantize MNIST image 726.
    dataset = datasets.MNIST(
        root="./data", train=False, download=False
    )
    image, label = dataset[IMAGE_INDEX]

    pixels = np.asarray(image, dtype=np.float32) / 255.0
    normalized = (pixels - 0.1307) / 0.3081
    q_image = np.clip(
        np.round(normalized * INPUT_SCALE), -128, 127
    ).astype(np.int32)

    save_hex_signed(
        os.path.join(PROJECT, "mnist_image.mem"),
        q_image.flatten(), 8
    )

    print(f"MNIST index: {IMAGE_INDEX}; label: {label}")

    # 2. Generate exact integrated Conv1 + ReLU + MaxPool output.
    subprocess.run(
        ["vvp", "conv1_mnist_pool_test"],
        cwd=PROJECT, check=True
    )

    pool1_path = os.path.join(PROJECT, "conv1_pool_output.mem")
    pool1 = np.loadtxt(pool1_path, dtype=np.int64).reshape(-1)

    if len(pool1) != 1568:
        raise ValueError(f"Expected 1568 pool1 values, got {len(pool1)}")

    # 3. Apply the same fixed-point requantization as requantize_array.v.
    temp = pool1 * RQ1_MULTIPLIER
    temp = temp + (1 << (RQ1_SHIFT - 1))
    q_conv2_input = temp >> RQ1_SHIFT
    q_conv2_input = np.clip(q_conv2_input, 0, 127).astype(np.int64)

    save_hex_signed(
        os.path.join(PROJECT, "conv2_input.mem"),
        q_conv2_input, 8
    )

    # 4. Run the actual Verilog Conv2 testbench.
    subprocess.run(
        ["vvp", "conv2_mnist_test"],
        cwd=PROJECT, check=True
    )

    verilog_output = np.loadtxt(
        os.path.join(PROJECT, "conv2_output.mem"),
        dtype=np.int64
    ).reshape(-1)

    # 5. Python reference using the exact quantized Conv2 inputs,
    # exported weights and exported biases.
    q_weights = read_hex_signed(
        os.path.join(PROJECT, "conv2_weights.mem"), 8
    ).reshape(16, 8, 3, 3)

    q_biases = read_hex_signed(
        os.path.join(PROJECT, "conv2_bias_int32.mem"), 32
    )

    x = torch.tensor(
        q_conv2_input.reshape(1, 8, 14, 14),
        dtype=torch.float64
    )
    w = torch.tensor(q_weights, dtype=torch.float64)
    b = torch.tensor(q_biases, dtype=torch.float64)

    with torch.no_grad():
        python_output = F.relu(
            F.conv2d(x, w, b, padding=1)
        ).to(torch.int64).numpy().reshape(-1)

    if len(verilog_output) != 3136:
        raise ValueError(
            f"Expected 3136 Verilog outputs, got {len(verilog_output)}"
        )

    diff = np.abs(python_output - verilog_output)

    print("\n========== IMAGE 726 CONV2 CHECK ==========")
    print("Python outputs:", len(python_output))
    print("Verilog outputs:", len(verilog_output))
    print("Mismatches:", int(np.count_nonzero(diff)))
    print("Maximum absolute difference:", int(diff.max()))

    print("\nIndex | Python | Verilog")
    for i in range(min(12, len(diff))):
        print(f"{i:5d} | {python_output[i]:6d} | {verilog_output[i]:7d}")

    bad = np.flatnonzero(diff)
    if len(bad):
        print("\nFirst mismatches:")
        for i in bad[:10]:
            print(
                f"{i:5d} | {python_output[i]:6d} | "
                f"{verilog_output[i]:7d} | diff={diff[i]}"
            )
    else:
        print("\nPASS: All Conv2 outputs match.")

finally:
    for name, data in backups.items():
        restore_file(os.path.join(PROJECT, name), data)

    print("\nOriginal memory files restored.")
