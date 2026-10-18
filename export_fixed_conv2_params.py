
import torch
from model import CNN

model = CNN()
model.load_state_dict(
    torch.load("cnn_mnist.pth", map_location="cpu", weights_only=True)
)
model.eval()

# Fixed calibration multipliers
POOL1_SCALE = 18.25425611513592

# Symmetric INT8 weight multiplier for Conv2
conv2_weight = model.conv2.weight.detach()
CONV2_WEIGHT_SCALE = 127.0 / conv2_weight.abs().max().item()

# Quantize Conv2 weights to signed INT8
q_weights = torch.round(
    conv2_weight * CONV2_WEIGHT_SCALE
).clamp(-127, 127).to(torch.int8)

# Bias is stored in accumulator units:
# input multiplier * weight multiplier
bias_multiplier = POOL1_SCALE * CONV2_WEIGHT_SCALE

q_bias = torch.round(
    model.conv2.bias.detach() * bias_multiplier
).to(torch.int32)

print("Pool1 multiplier:", POOL1_SCALE)
print("Conv2 weight multiplier:", CONV2_WEIGHT_SCALE)
print("Conv2 bias multiplier:", bias_multiplier)
print("Conv2 INT32 biases:", q_bias.tolist())

# Export weights as two-digit hexadecimal
with open("conv2_weights.mem", "w") as f:
    for value in q_weights.reshape(-1).tolist():
        f.write(f"{int(value) & 0xFF:02X}\n")

# Export biases as eight-digit hexadecimal
with open("conv2_bias_int32.mem", "w") as f:
    for value in q_bias.tolist():
        f.write(f"{int(value) & 0xFFFFFFFF:08X}\n")

print("Saved conv2_weights.mem")
print("Saved conv2_bias_int32.mem")
print("Weight count:", q_weights.numel())
print("Bias count:", q_bias.numel())
