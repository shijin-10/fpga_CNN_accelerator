
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
                value -= (1 << bits)

            values.append(value)

    return np.array(values, dtype=np.int32)


# Load verified Conv1 + ReLU + MaxPool INT32 outputs
pooled = np.loadtxt("conv1_pool_output.mem", dtype=np.int32)

assert len(pooled) == 1568, (
    f"Expected 1568 pooled values, got {len(pooled)}"
)

# Quantize pooled activations to signed INT8.
# This is a temporary per-image scale for RTL testing.
max_abs = np.max(np.abs(pooled))
scale = max_abs / 127.0 if max_abs != 0 else 1.0

q_pooled = np.clip(
    np.round(pooled / scale),
    -128,
    127
).astype(np.int8)

# Export as two-digit hexadecimal, compatible with $readmemh
with open("conv2_input.mem", "w") as f:
    for value in q_pooled:
        f.write(f"{int(value) & 0xFF:02X}\n")

print("Exported conv2_input.mem")
print("Values:", len(q_pooled))
print("Scale:", scale)
print("INT8 range:", int(q_pooled.min()), "to", int(q_pooled.max()))
print("First 10 INT8 values:", q_pooled[:10].tolist())

