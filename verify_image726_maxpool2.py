
import os
import subprocess
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import datasets

PROJECT = os.path.abspath(".")
IMAGE_INDEX = 726
INPUT_SCALE = 45.011730688743526
RQ1_MULTIPLIER = 3063111
RQ1_SHIFT = 30

FILES = [
    "mnist_image.mem",
    "conv1_pool_output.mem",
    "conv2_input.mem",
    "conv2_output.mem",
    "conv2_output_hex.mem",
    "conv2_pool_output.mem",
]

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

def write_hex(filename, values, bits):
    mask = (1 << bits) - 1
    width = bits // 4
    with open(filename, "w") as f:
        for value in values:
            f.write(f"{int(value) & mask:0{width}X}\n")

backups = {
    name: backup_file(os.path.join(PROJECT, name))
    for name in FILES
}

try:
    # Load MNIST test image 726.
    dataset = datasets.MNIST(
        root="./data", train=False, download=False
    )
    image, label = dataset[IMAGE_INDEX]

    pixels = np.asarray(image, dtype=np.float32) / 255.0
    normalized = (pixels - 0.1307) / 0.3081
    q_image = np.clip(
        np.round(normalized * INPUT_SCALE), -128, 127
    ).astype(np.int32)

    print(f"MNIST index: {IMAGE_INDEX}; label: {label}")

    write_hex(
        os.path.join(PROJECT, "mnist_image.mem"),
        q_image.flatten(), 8
    )

    # Generate Conv1 + ReLU + MaxPool1 outputs.
    subprocess.run(
        ["vvp", "conv1_mnist_pool_test"],
        cwd=PROJECT, check=True
    )

    pool1 = np.loadtxt(
        os.path.join(PROJECT, "conv1_pool_output.mem"),
        dtype=np.int64
    ).reshape(-1)

    # Match requantize_array.v: round, arithmetic shift, clamp.
    temp = pool1 * RQ1_MULTIPLIER + (1 << (RQ1_SHIFT - 1))
    q_conv2_input = np.clip(
        temp >> RQ1_SHIFT, 0, 127
    ).astype(np.int64)

    write_hex(
        os.path.join(PROJECT, "conv2_input.mem"),
        q_conv2_input, 8
    )

    # Generate integrated Conv2 outputs.
    subprocess.run(
        ["vvp", "conv2_mnist_test"],
        cwd=PROJECT, check=True
    )

    conv2 = np.loadtxt(
        os.path.join(PROJECT, "conv2_output.mem"),
        dtype=np.int64
    ).reshape(-1)

    if len(conv2) != 3136:
        raise ValueError(f"Expected 3136 Conv2 outputs, got {len(conv2)}")

    # maxpool2_tb.v reads hexadecimal INT32 input.
    write_hex(
        os.path.join(PROJECT, "conv2_output_hex.mem"),
        conv2, 32
    )

    # Run actual Verilog MaxPool2.
    subprocess.run(
        [
            "iverilog", "-g2012", "-o", "maxpool2_image726_sim",
            "maxpool2_tb.v", "maxpool_2x2_conv2.v"
        ],
        cwd=PROJECT, check=True
    )
    subprocess.run(
        ["vvp", "maxpool2_image726_sim"],
        cwd=PROJECT, check=True
    )

    verilog_pool = np.loadtxt(
        os.path.join(PROJECT, "conv2_pool_output.mem"),
        dtype=np.int64
    ).reshape(-1)

    # Python MaxPool2 reference, same Conv2 outputs.
    x = torch.tensor(
        conv2.reshape(1, 16, 14, 14), dtype=torch.int64
    )
    python_pool = F.max_pool2d(x, 2, 2).numpy().reshape(-1)

    diff = np.abs(python_pool - verilog_pool)

    print("\n========== IMAGE 726 MAXPOOL2 CHECK ==========")
    print("Python outputs:", len(python_pool))
    print("Verilog outputs:", len(verilog_pool))
    print("Mismatches:", int(np.count_nonzero(diff)))
    print("Maximum absolute difference:", int(diff.max()))

    print("\nIndex | Python | Verilog")
    for i in range(min(12, len(diff))):
        print(f"{i:5d} | {python_pool[i]:6d} | {verilog_pool[i]:7d}")

    bad = np.flatnonzero(diff)
    if len(bad):
        print("\nFirst mismatches:")
        for i in bad[:10]:
            print(
                f"{i:5d} | {python_pool[i]:6d} | "
                f"{verilog_pool[i]:7d} | diff={diff[i]}"
            )
    else:
        print("\nPASS: All MaxPool2 outputs match.")

finally:
    for name, data in backups.items():
        restore_file(os.path.join(PROJECT, name), data)

    print("\nOriginal memory files restored.")

