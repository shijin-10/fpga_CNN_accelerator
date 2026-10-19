import numpy as np
import torch
import torch.nn.functional as F

def read_hex_signed(filename, bits):
    mask = (1 << bits) - 1
    sign_bit = 1 << (bits - 1)
    values = []

    with open(filename) as f:
        for line in f:
            line = line.strip()
            if line:
                v = int(line, 16) & mask
                if v >= sign_bit:
                    v -= 1 << bits
                values.append(v)

    return np.array(values, dtype=np.int64)

x = read_hex_signed("conv2_input.mem", 8)
w = read_hex_signed("conv2_weights.mem", 8)
b = read_hex_signed("conv2_bias_int32.mem", 32)

assert len(x) == 1568, f"Input count: {len(x)}"
assert len(w) == 1152, f"Weight count: {len(w)}"
assert len(b) == 16, f"Bias count: {len(b)}"

with torch.no_grad():
    output = F.conv2d(
        torch.tensor(x.reshape(1, 8, 14, 14), dtype=torch.int64).to(torch.float64),
        torch.tensor(w.reshape(16, 8, 3, 3), dtype=torch.float64),
        torch.tensor(b, dtype=torch.float64),
        padding=1,
    )
    reference = F.relu(output).to(torch.int64).numpy().reshape(-1)

np.savetxt("conv2_reference_fresh.mem", reference, fmt="%d")

print("Fresh reference created: conv2_reference_fresh.mem")
print("Output count:", len(reference))
print("First 30 values:", reference[:30].tolist())
print("Input range:", int(x.min()), "to", int(x.max()))
print("Weight range:", int(w.min()), "to", int(w.max()))
print("Bias range:", int(b.min()), "to", int(b.max()))
