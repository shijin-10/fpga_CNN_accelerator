
import torch
from model import CNN

# Load trained CNN
model = CNN()
model.load_state_dict(
    torch.load(
        "cnn_mnist.pth",
        map_location="cpu",
        weights_only=True
    )
)
model.eval()

# Must match validate_fixed_pipeline.py
POOL2_SCALE = 11.704324914902063

# FC weights and biases from trained model
fc_weights = model.fc.weight.detach()
fc_bias = model.fc.bias.detach()

# Symmetric INT8 weight quantization
fc_weight_scale = 127.0 / fc_weights.abs().max().item()

q_fc_weights = torch.round(
    fc_weights * fc_weight_scale
).clamp(-127, 127).to(torch.int8)

# Bias must use the FC input scale and FC weight scale
bias_scale = POOL2_SCALE * fc_weight_scale

q_fc_bias = torch.round(
    fc_bias * bias_scale
).to(torch.int32)

print("FC input scale:", POOL2_SCALE)
print("FC weight scale:", fc_weight_scale)
print("FC bias scale:", bias_scale)
print("FC weight shape:", tuple(q_fc_weights.shape))
print("FC bias count:", len(q_fc_bias))
print("FC bias values:", q_fc_bias.tolist())

# Export signed INT8 weights as hexadecimal
with open("fc_weights.mem", "w") as f:
    for value in q_fc_weights.reshape(-1).tolist():
        f.write(f"{int(value) & 0xFF:02X}\n")

# Export signed INT32 biases as hexadecimal
with open("fc_bias_int32.mem", "w") as f:
    for value in q_fc_bias.tolist():
        f.write(f"{int(value) & 0xFFFFFFFF:08X}\n")

print("\nSaved fc_weights.mem")
print("Saved fc_bias_int32.mem")
