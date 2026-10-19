
import numpy as np

# FC input reference is decimal, 784 values.
x = np.loadtxt("fc_input_requantized.mem", dtype=np.int64)
assert len(x) == 784, f"Expected 784 inputs, got {len(x)}"

# FC weights are hexadecimal signed INT8 values.
raw_weights = np.loadtxt("fc_weights.mem", dtype=str)
assert len(raw_weights) == 7840, f"Expected 7840 weights, got {len(raw_weights)}"

def signed_value(value, bits):
    value = int(value, 16)
    if value >= (1 << (bits - 1)):
        value -= 1 << bits
    return value

weights = np.array(
    [signed_value(v, 8) for v in raw_weights],
    dtype=np.int64
).reshape(10, 784)

# FC biases are hexadecimal signed INT32 values.
raw_biases = np.loadtxt("fc_bias_int32.mem", dtype=str)
assert len(raw_biases) == 10, f"Expected 10 biases, got {len(raw_biases)}"

biases = np.array(
    [signed_value(v, 32) for v in raw_biases],
    dtype=np.int64
)

# Match RTL: sum of 784 signed products plus signed bias.
scores = np.sum(weights * x[np.newaxis, :], axis=1) + biases

# Testbench reads decimal scores into signed 64-bit registers.
np.savetxt("fc_output.mem", scores, fmt="%d")

print("Created fresh fc_output.mem")
print("Scores:", scores.tolist())
print("Predicted digit:", int(np.argmax(scores)))
