
import numpy as np

MULTIPLIER = 2658134
SHIFT = 30

pool2 = np.loadtxt("conv2_pool_output.mem", dtype=np.int64)
assert len(pool2) == 784, f"Expected 784 Pool2 values, got {len(pool2)}"

# Match RTL: (input * multiplier + 2^(SHIFT-1)) >>> SHIFT
temp = pool2 * MULTIPLIER + (1 << (SHIFT - 1))
scaled = temp >> SHIFT

# Match RTL saturation to the range 0..127.
fc_input = np.clip(scaled, 0, 127).astype(np.int32)

# Testbench reads decimal integers.
np.savetxt("fc_input_requantized.mem", fc_input, fmt="%d")

print("Created fc_input_requantized.mem")
print("Values:", len(fc_input))
print("Range:", int(fc_input.min()), "to", int(fc_input.max()))
print("First 20:", fc_input[:20].tolist())
