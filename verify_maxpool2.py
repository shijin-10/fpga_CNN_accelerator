
import numpy as np
import torch
import torch.nn.functional as F

# Read decimal Conv2 outputs
conv2 = np.loadtxt("conv2_output.mem", dtype=np.int64)

assert len(conv2) == 3136, f"Expected 3136 values, got {len(conv2)}"

# Reshape to [batch, channels, height, width]
conv2 = conv2.reshape(1, 16, 14, 14)

# Python reference max pooling
x = torch.tensor(conv2, dtype=torch.int64)
python_output = F.max_pool2d(x, kernel_size=2, stride=2)
python_output = python_output.numpy().reshape(-1)

# Load Verilog output
verilog_output = np.loadtxt(
    "conv2_pool_output.mem",
    dtype=np.int64
).reshape(-1)

assert len(verilog_output) == 784, (
    f"Expected 784 Verilog outputs, got {len(verilog_output)}"
)

# Compare
differences = np.abs(python_output - verilog_output)
mismatches = np.count_nonzero(differences)

print("========== MAXPOOL2 VERIFICATION ==========")
print("Python outputs:", len(python_output))
print("Verilog outputs:", len(verilog_output))
print("Mismatches:", mismatches)
print("Maximum absolute difference:", differences.max())
print("Python first output:", python_output[0])
print("Verilog first output:", verilog_output[0])

if mismatches == 0:
    print("\nSUCCESS: All 784 MaxPool2 outputs match!")
else:
    print("\nVerification FAILED.")
    for i in np.where(differences != 0)[0][:10]:
        print(
            f"Index {i}: Python={python_output[i]}, "
            f"Verilog={verilog_output[i]}"
        )
