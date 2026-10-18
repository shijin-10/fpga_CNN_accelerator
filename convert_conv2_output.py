
import numpy as np

values = np.loadtxt("conv2_output.mem", dtype=np.int64)

assert len(values) == 3136, f"Expected 3136 values, got {len(values)}"

with open("conv2_output_hex.mem", "w") as f:
    for value in values:
        f.write(f"{int(value) & 0xFFFFFFFF:08X}\n")

print("Created conv2_output_hex.mem")
print("Values:", len(values))
print("First 5 values:", values[:5].tolist())

