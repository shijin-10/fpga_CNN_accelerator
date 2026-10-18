
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
RQ2_MULTIPLIER = 2850408
RQ2_SHIFT = 30

FILES = [
    "mnist_image.mem",
    "conv1_pool_output.mem",
    "conv2_input.mem",
    "conv2_output.mem",
    "conv2_output_hex.mem",
    "conv2_pool_output.mem",
    "fc_input_requantized.mem",
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

def write_hex(filename, values, bits):
    mask = (1 << bits) - 1
    width = bits // 4
    with open(filename, "w") as f:
        for value in values:
            f.write(f"{int(value) & mask:0{width}X}\n")

def rtl_requantize(values, multiplier, shift):
    # Match requantize_array.v: multiply, add rounding offset,
    # arithmetic right shift, then clamp to [0, 127].
    values = np.asarray(values, dtype=np.int64)
    temp = values * multiplier + (1 << (shift - 1))
    scaled = temp >> shift
    return np.clip(scaled, 0, 127).astype(np.int64)

backups = {
    name: backup_file(os.path.join(PROJECT, name))
    for name in FILES
}

try:
    # Load MNIST image 726 and quantize it.
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

    # Conv1 + ReLU + MaxPool1.
    subprocess.run(
        ["vvp", "conv1_mnist_pool_test"],
        cwd=PROJECT, check=True
    )
    pool1 = np.loadtxt(
        os.path.join(PROJECT, "conv1_pool_output.mem"),
        dtype=np.int64
    ).reshape(-1)

    # Requantize Pool1 to Conv2 input.
    conv2_input = rtl_requantize(
        pool1, RQ1_MULTIPLIER, RQ1_SHIFT
    )
    write_hex(
        os.path.join(PROJECT, "conv2_input.mem"),
        conv2_input, 8
    )

    # Conv2.
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

    # MaxPool2.
    write_hex(
        os.path.join(PROJECT, "conv2_output_hex.mem"),
        conv2, 32
    )
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

    pool2 = np.loadtxt(
        os.path.join(PROJECT, "conv2_pool_output.mem"),
        dtype=np.int64
    ).reshape(-1)

    if len(pool2) != 784:
        raise ValueError(f"Expected 784 MaxPool2 values, got {len(pool2)}")

    # Apply the same second requantization as cnn_top.v.
    rtl_fc_input = rtl_requantize(
        pool2, RQ2_MULTIPLIER, RQ2_SHIFT
    )

    # Independent Python reference for MaxPool2 from Conv2 outputs.
    x = torch.tensor(
        conv2.reshape(1, 16, 14, 14), dtype=torch.int64
    )
    python_pool2 = F.max_pool2d(x, 2, 2).numpy().reshape(-1)

    python_fc_input = rtl_requantize(
        python_pool2, RQ2_MULTIPLIER, RQ2_SHIFT
    )

    differences = np.abs(python_fc_input - rtl_fc_input)

    print("\n========== IMAGE 726 REQUANTIZATION 2 CHECK ==========")
    print("Python FC inputs:", len(python_fc_input))
    print("RTL FC inputs:", len(rtl_fc_input))
    print("Mismatches:", int(np.count_nonzero(differences)))
    print("Maximum absolute difference:", int(differences.max()))
    print("\nIndex | Python | RTL")
    for i in range(min(16, len(differences))):
        print(f"{i:5d} | {python_fc_input[i]:6d} | {rtl_fc_input[i]:3d}")

    bad = np.flatnonzero(differences)
    if len(bad):
        print("\nFirst mismatches:")
        for i in bad[:10]:
            print(
                f"{i:5d} | {python_fc_input[i]:6d} | "
                f"{rtl_fc_input[i]:3d}"
            )
    else:
        print("\nPASS: All 784 requantized FC inputs match.")

    # Save decimal values for optional inspection.
    np.savetxt(
        os.path.join(PROJECT, "fc_input_requantized.mem"),
        rtl_fc_input, fmt="%d"
    )

finally:
    for name, data in backups.items():
        restore_file(os.path.join(PROJECT, name), data)

    print("\nOriginal memory files restored.")
