
import numpy as np
import torch
from model import CNN

# Fixed calibration multipliers
POOL1_SCALE = 18.25425611513592
CONV2_WEIGHT_SCALE = 241.5322203712151
POOL2_SCALE = 11.704324914902063

# Load trained model
model = CNN()
model.load_state_dict(
    torch.load(
        "cnn_mnist.pth",
        map_location="cpu",
        weights_only=True
    )
)
model.eval()

# FC weight multiplier
fc_weight = model.fc.weight.detach()
FC_WEIGHT_SCALE = 127.0 / fc_weight.abs().max().item()

# Load MaxPool2 INT32 accumulations
pooled = np.loadtxt(
    "conv2_pool_output.mem",
    dtype=np.int64
)

assert len(pooled) == 784, (
    f"Expected 784 values, got {len(pooled)}"
)

# Convert Conv2 accumulator units to real activation values,
# then quantize with the fixed Pool2 multiplier.
real_activations = pooled / (
    POOL1_SCALE * CONV2_WEIGHT_SCALE
)

q_fc_input = np.clip(
    np.rint(real_activations * POOL2_SCALE),
    0,
    127
).astype(np.int32)

# Export as 32-bit hexadecimal because fc_layer.v
# reads input_data as signed 32-bit values.
with open("fc_input.mem", "w") as f:
    for value in q_fc_input:
        f.write(f"{int(value) & 0xFFFFFFFF:08X}\n")

# Export FC biases in matching accumulator units
fc_bias = model.fc.bias.detach()

q_fc_bias = torch.round(
    fc_bias * POOL2_SCALE * FC_WEIGHT_SCALE
).to(torch.int32)

with open("fc_bias_int32.mem", "w") as f:
    for value in q_fc_bias.tolist():
        f.write(f"{int(value) & 0xFFFFFFFF:08X}\n")

print("FC input values:", len(q_fc_input))
print("FC input range:", int(q_fc_input.min()),
      "to", int(q_fc_input.max()))
print("FC weight multiplier:", FC_WEIGHT_SCALE)
print("FC bias multiplier:", POOL2_SCALE * FC_WEIGHT_SCALE)
print("FC INT32 biases:", q_fc_bias.tolist())
print("Saved fc_input.mem and fc_bias_int32.mem")
