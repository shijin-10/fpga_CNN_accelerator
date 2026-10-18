
import os
import subprocess
import numpy as np
from torchvision import datasets

PROJECT = os.path.abspath(".")
INDEX = 726

INPUT_SCALE = 45.011730688743526
RQ1_MULTIPLIER = 3063111
RQ2_MULTIPLIER = 2850408
SHIFT = 30

FILES = [
    "mnist_image.mem",
    "conv1_pool_output.mem",
    "conv2_input.mem",
    "conv2_output.mem",
    "conv2_output_hex.mem",
    "conv2_pool_output.mem",
    "fc_input.mem",
    "fc_output.mem",
]

def backup(path):
    if os.path.exists(path):
        with open(path, "rb") as f:
            return f.read()
    return None

def restore(path, content):
    if content is None:
        if os.path.exists(path):
            os.remove(path)
    else:
        with open(path, "wb") as f:
            f.write(content)

def write_hex(path, values, bits):
    mask = (1 << bits) - 1
    width = bits // 4
    with open(path, "w") as f:
        for value in values:
            f.write(f"{int(value) & mask:0{width}X}\n")

def read_hex_signed(path, bits):
    values = []
    mask = (1 << bits) - 1
    sign_bit = 1 << (bits - 1)
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                value = int(line, 16) & mask
                if value >= sign_bit:
                    value -= 1 << bits
                values.append(value)
    return np.asarray(values, dtype=np.int64)

def requantize(values, multiplier):
    values = np.asarray(values, dtype=np.int64)
    temp = values * multiplier + (1 << (SHIFT - 1))
    return np.clip(temp >> SHIFT, 0, 127).astype(np.int64)

saved = {
    name: backup(os.path.join(PROJECT, name))
    for name in FILES
}

try:
    # Load MNIST test image 726.
    dataset = datasets.MNIST(
        root="./data", train=False, download=False
    )
    image, label = dataset[INDEX]

    pixels = np.asarray(image, dtype=np.float32) / 255.0
    normalized = (pixels - 0.1307) / 0.3081
    q_image = np.clip(
        np.round(normalized * INPUT_SCALE), -128, 127
    ).astype(np.int64)

    print(f"MNIST index: {INDEX}; actual label: {label}")
    write_hex("mnist_image.mem", q_image.flatten(), 8)

    # Conv1 + ReLU + MaxPool1.
    subprocess.run(
        ["vvp", "conv1_mnist_pool_test"],
        cwd=PROJECT, check=True
    )
    pool1 = np.loadtxt(
        "conv1_pool_output.mem", dtype=np.int64
    ).reshape(-1)

    # Requantize into Conv2.
    conv2_input = requantize(pool1, RQ1_MULTIPLIER)
    write_hex("conv2_input.mem", conv2_input, 8)

    # Conv2.
    subprocess.run(
        ["vvp", "conv2_mnist_test"],
        cwd=PROJECT, check=True
    )
    conv2 = np.loadtxt(
        "conv2_output.mem", dtype=np.int64
    ).reshape(-1)
    if len(conv2) != 3136:
        raise ValueError(f"Expected 3136 Conv2 outputs, got {len(conv2)}")

    # MaxPool2.
    write_hex("conv2_output_hex.mem", conv2, 32)
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
        "conv2_pool_output.mem", dtype=np.int64
    ).reshape(-1)

    # Create exact FC inputs for this image.
    fc_input = requantize(pool2, RQ2_MULTIPLIER)
    write_hex("fc_input.mem", fc_input, 32)

    # Compile and run the actual FC testbench.
    subprocess.run(
        [
            "iverilog", "-g2012", "-o", "fc_image726_sim",
            "fc_layer_tb.v", "fc_layer.v"
        ],
        cwd=PROJECT, check=True
    )
    subprocess.run(
        ["vvp", "fc_image726_sim"],
        cwd=PROJECT, check=True
    )

    verilog_scores = np.loadtxt(
        "fc_output.mem", dtype=np.int64
    ).reshape(-1)

    weights = read_hex_signed("fc_weights.mem", 8).reshape(10, 784)
    biases = read_hex_signed("fc_bias_int32.mem", 32)

    python_scores = np.array([
        sum(int(x) * int(w) for x, w in zip(fc_input, row))
        + int(bias)
        for row, bias in zip(weights, biases)
    ], dtype=np.int64)

    differences = np.abs(python_scores - verilog_scores)

    print("\n========== IMAGE 726 FC CHECK ==========")
    print("Python scores:", python_scores.tolist())
    print("Verilog scores:", verilog_scores.tolist())
    print("Mismatches:", int(np.count_nonzero(differences)))
    print("Maximum absolute difference:", int(differences.max()))
    print("Python predicted digit:", int(np.argmax(python_scores)))
    print("Verilog predicted digit:", int(np.argmax(verilog_scores)))

    if np.count_nonzero(differences) == 0:
        print("\nPASS: All 10 FC scores match.")
    else:
        print("\nFAIL: FC score mismatch.")

finally:
    for name, content in saved.items():
        restore(os.path.join(PROJECT, name), content)

    print("\nOriginal memory files restored.")
