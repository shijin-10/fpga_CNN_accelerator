
import torch
from model import CNN

model = CNN()
model.load_state_dict(
    torch.load("cnn_mnist.pth", map_location="cpu", weights_only=True)
)
model.eval()

# Multipliers used to quantize the Conv1 input and weights
conv1_input_multiplier = 45.0117301940918
conv1_weight_multiplier = 142.15939331054688

# Read the actual Conv1 pooled INT32 output
import numpy as np

pooled = np.loadtxt("conv1_pool_output.mem", dtype=np.int32)
assert len(pooled) == 1568

# Recover approximate real-valued activations
pooled_real = (
    pooled.astype(np.float64)
    / (conv1_input_multiplier * conv1_weight_multiplier)
)

max_real = np.max(np.abs(pooled_real))

# INT8 activation multiplier for Conv2 input
conv2_input_multiplier = 127.0 / max_real if max_real else 1.0

# Conv2 weight multiplier
conv2_weight = model.conv2.weight.detach()
conv2_weight_multiplier = 127.0 / conv2_weight.abs().max().item()

# Convert floating-point bias to the Conv2 accumulator scale
bias_multiplier = (
    conv2_input_multiplier * conv2_weight_multiplier
)

bias_int32 = torch.round(
    model.conv2.bias.detach() * bias_multiplier
).to(torch.int32)

print("Maximum recovered activation:", max_real)
print("Conv2 input multiplier:", conv2_input_multiplier)
print("Conv2 weight multiplier:", conv2_weight_multiplier)
print("Bias multiplier:", bias_multiplier)
print("Conv2 INT32 biases:", bias_int32.tolist())

with open("conv2_bias_int32.mem", "w") as f:
    for value in bias_int32.tolist():
        f.write(f"{int(value) & 0xFFFFFFFF:08X}\n")

print("Saved conv2_bias_int32.mem")
print("Bias count:", len(bias_int32))

