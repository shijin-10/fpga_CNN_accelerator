
import numpy as np

values = np.loadtxt("conv2_pool_output.mem", dtype=np.int64)
assert len(values) == 784

with open("fc_input.mem", "w") as f:
    for value in values:
        f.write(f"{int(value) & 0xFFFFFFFF:08X}\n")

print("Created fc_input.mem")
print("Values:", len(values))
print("Range:", int(values.min()), "to", int(values.max()))
