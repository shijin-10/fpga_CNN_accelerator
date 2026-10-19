
import numpy as np

# Fixed multipliers from calibration
INPUT_SCALE = 45.011730688743526
CONV1_WEIGHT_SCALE = 177.25626334908588
POOL1_SCALE = 18.25425611513592

# Load verified Conv1 + ReLU + MaxPool outputs.
# This file contains decimal INT32 accumulations.
pooled = np.loadtxt("conv1_pool_output.mem", dtype=np.int64)

assert len(pooled) == 1568, (
    f"Expected 1568 values, got {len(pooled)}"
)

# Convert accumulator values to the calibrated INT8 activation scale
real_values = pooled / (INPUT_SCALE * CONV1_WEIGHT_SCALE)

q_pool1 = np.clip(
    np.rint(real_values * POOL1_SCALE),
    0,
    127
).astype(np.int8)

# Export two-digit hexadecimal for Verilog $readmemh
with open("conv2_input.mem", "w") as f:
    for value in q_pool1:
        f.write(f"{int(value) & 0xFF:02X}\n")

print("Fixed-scale Conv2 input exported")
print("Values:", len(q_pool1))
print("INT8 range:", int(q_pool1.min()), "to", int(q_pool1.max()))
print("First 10 values:", q_pool1[:10].tolist())
print("Pool1 multiplier:", POOL1_SCALE)
