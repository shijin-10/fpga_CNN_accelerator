
import numpy as np
import torch
import torch.nn.functional as F


def read_hex_signed(filename, bits):
    values = []
    mask = (1 << bits) - 1
    sign_bit = 1 << (bits - 1)

    with open(filename, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            value = int(line, 16) & mask

            if value >= sign_bit:
                value -= (1 << bits)

            values.append(value)

    return np.array(values, dtype=np.int64)


print("Loading Conv2 memory files...")

# Load signed hexadecimal memory files
q_input = read_hex_signed("conv2_input.mem", 8)
q_weights = read_hex_signed("conv2_weights.mem", 8)
q_bias = read_hex_signed("conv2_bias_int32.mem", 32)

# Validate expected sizes
assert len(q_input) == 1568, f"Input count: {len(q_input)}"
assert len(q_weights) == 1152, f"Weight count: {len(q_weights)}"
assert len(q_bias) == 16, f"Bias count: {len(q_bias)}"

# Input layout: channel-major, 8 x 14 x 14
x = torch.tensor(
    q_input.reshape(1, 8, 14, 14),
    dtype=torch.float64
)

# Weight layout: filter-major, 16 x 8 x 3 x 3
w = torch.tensor(
    q_weights.reshape(16, 8, 3, 3),
    dtype=torch.float64
)

b = torch.tensor(q_bias, dtype=torch.float64)

print("Calculating Python Conv2 reference...")

# Integer-valued convolution with padding, bias and ReLU
conv = F.conv2d(x, w, b, stride=1, padding=1)
relu = F.relu(conv)

# Preserve filter-major order: 16 filters x 14 x 14
python_output = relu.to(torch.int64).numpy().reshape(-1)

# Load decimal outputs written by the Verilog testbench
verilog_output = np.loadtxt(
    "conv2_output.mem",
    dtype=np.int64
).reshape(-1)

print("\n========== CONV2 VERIFICATION ==========")
print("Python outputs:", len(python_output))
print("Verilog outputs:", len(verilog_output))

if len(python_output) != len(verilog_output):
    raise ValueError("Output counts do not match!")

differences = np.abs(python_output - verilog_output)
mismatches = np.count_nonzero(differences)

print("Mismatches:", mismatches)
print("Maximum absolute difference:", differences.max())
print("Python first output:", python_output[0])
print("Verilog first output:", verilog_output[0])

if mismatches == 0:
    print("\nSUCCESS: All Conv2 outputs match!")
    print("Conv2 arithmetic verification PASSED.")
else:
    print("\nVerification FAILED.")
    print("\nFirst 10 mismatches:")
    print("Index | Python | Verilog | Difference")

    indices = np.where(differences != 0)[0][:10]

    for i in indices:
        print(
            f"{i:5d} | {python_output[i]:6d} | "
            f"{verilog_output[i]:7d} | {differences[i]:10d}"
        )
