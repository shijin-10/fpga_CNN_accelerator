
import numpy as np
import torch
import torch.nn.functional as F


# --------------------------------------------------
# Read hexadecimal memory files as signed integers
# --------------------------------------------------
def read_hex_signed(filename, bits):
    values = []
    mask = (1 << bits) - 1
    sign_bit = 1 << (bits - 1)

    with open(filename, "r") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()

            if not line:
                continue

            value = int(line, 16)
            value &= mask

            # Convert two's-complement to signed integer
            if value >= sign_bit:
                value -= (1 << bits)

            values.append(value)

    return np.array(values, dtype=np.int32)


# --------------------------------------------------
# Load INT8 image, weights and INT32 biases
# --------------------------------------------------
print("Loading exported memory files...")

q_image = read_hex_signed("mnist_image.mem", 8)
q_weights = read_hex_signed("conv1_weights.mem", 8)
q_bias = read_hex_signed("conv1_bias_int32.mem", 32)

# Verify file sizes
assert len(q_image) == 784, (
    f"Expected 784 image pixels, got {len(q_image)}"
)
assert len(q_weights) == 72, (
    f"Expected 72 Conv1 weights, got {len(q_weights)}"
)
assert len(q_bias) == 8, (
    f"Expected 8 Conv1 biases, got {len(q_bias)}"
)

print("Image values:", len(q_image))
print("Conv1 weights:", len(q_weights))
print("Conv1 biases:", len(q_bias))


# --------------------------------------------------
# Prepare tensors for Python reference calculation
# --------------------------------------------------
q_image = q_image.reshape(1, 1, 28, 28)
q_weights = q_weights.reshape(8, 1, 3, 3)

# INT32 accumulation using integer-valued tensors
x = torch.tensor(q_image, dtype=torch.int32)
w = torch.tensor(q_weights, dtype=torch.int32)
b = torch.tensor(q_bias, dtype=torch.int32)


# --------------------------------------------------
# Conv1 + Bias + ReLU
# --------------------------------------------------
print("\nCalculating Python reference outputs...")

# Conv2d requires compatible numeric types.
# These values are integers, represented exactly as float64
# for the convolution reference calculation.
conv = F.conv2d(
    x.to(torch.float64),
    w.to(torch.float64),
    b.to(torch.float64),
    padding=1
)

relu = F.relu(conv)

# MaxPool 2x2, stride 2
pool = F.max_pool2d(relu, kernel_size=2, stride=2)

# Hardware uses filter-major order:
# filter 0's 196 values, then filter 1's 196 values, etc.
python_output = pool.to(torch.int32).numpy().reshape(-1)


# --------------------------------------------------
# Load Verilog output
# --------------------------------------------------
print("Loading Verilog outputs...")

verilog_output = np.loadtxt(
    "conv1_pool_output.mem",
    dtype=np.int32
).reshape(-1)


# --------------------------------------------------
# Compare all outputs
# --------------------------------------------------
print("\n========== COMPARISON RESULTS ==========")

print("Python output count:", len(python_output))
print("Verilog output count:", len(verilog_output))

if len(python_output) != len(verilog_output):
    raise ValueError(
        "Output counts differ. Check the Verilog testbench."
    )

differences = np.abs(
    python_output.astype(np.int64)
    - verilog_output.astype(np.int64)
)

mismatches = np.count_nonzero(differences)

print("Mismatches:", mismatches)
print("Maximum absolute difference:", differences.max())

if mismatches == 0:
    print("\nSUCCESS: All 1568 outputs match!")
    print("Conv1 + ReLU + MaxPool verification PASSED.")
else:
    print("\nVerification FAILED.")

    indices = np.where(differences != 0)[0][:10]

    print("\nFirst mismatches:")
    print("Index | Python | Verilog | Difference")

    for i in indices:
        print(
            f"{i:5d} | "
            f"{python_output[i]:6d} | "
            f"{verilog_output[i]:7d} | "
            f"{differences[i]:10d}"
        )

    print(
        "\nDo not proceed to Conv2 until we investigate "
        "the mismatches."
    )

