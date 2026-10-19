
import numpy as np

# Fresh Conv2 reference contains 16 channels × 14 × 14 values.
conv2 = np.loadtxt("conv2_reference.mem", dtype=np.int64)
assert len(conv2) == 3136, f"Expected 3136 Conv2 values, got {len(conv2)}"

conv2 = conv2.reshape(16, 14, 14)

# 2×2 max pooling, stride 2.
pool2 = conv2.reshape(16, 7, 2, 7, 2).max(axis=(2, 4))
pool2 = pool2.reshape(-1)

assert len(pool2) == 784

# Testbench reads this file as decimal integers.
np.savetxt("conv2_pool_output.mem", pool2, fmt="%d")

print("Created conv2_pool_output.mem")
print("Values:", len(pool2))
print("Range:", int(pool2.min()), "to", int(pool2.max()))
print("First 10:", pool2[:10].tolist())
