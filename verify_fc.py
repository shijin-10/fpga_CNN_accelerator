
import numpy as np


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
                value -= 1 << bits

            values.append(value)

    return np.array(values, dtype=np.int64)


# Load FC inputs, weights and biases
x = read_hex_signed("fc_input.mem", 32)
w = read_hex_signed("fc_weights.mem", 8)
b = read_hex_signed("fc_bias_int32.mem", 32)

assert len(x) == 784
assert len(w) == 7840
assert len(b) == 10

# FC weights are output-major: 10 x 784
w = w.reshape(10, 784)

# Calculate reference scores using int64 arithmetic
python_scores = np.array(
    [sum(int(xi) * int(wi) for xi, wi in zip(x, row)) + int(bias)
     for row, bias in zip(w, b)],
    dtype=np.int64
)

# Load Verilog scores, written in decimal
verilog_scores = np.loadtxt(
    "fc_output.mem",
    dtype=np.int64
).reshape(-1)

differences = np.abs(python_scores - verilog_scores)

print("========== FC VERIFICATION ==========")
print("Python scores:", python_scores.tolist())
print("Verilog scores:", verilog_scores.tolist())
print("Mismatches:", np.count_nonzero(differences))
print("Maximum absolute difference:", int(differences.max()))

print("Python predicted digit:", int(np.argmax(python_scores)))
print("Verilog predicted digit:", int(np.argmax(verilog_scores)))

if np.count_nonzero(differences) == 0:
    print("\nSUCCESS: All 10 FC scores match!")
else:
    print("\nVerification FAILED.")
