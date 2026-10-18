
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

# Current quantization multipliers from Conv2
conv2_input_multiplier = 19.55748730611678
conv2_weight_multiplier = 241.5322203712151

# Conv2 output accumulator multiplier relative to real values
fc_input_multiplier = (
    conv2_input_multiplier * conv2_weight_multiplier
)

# FC weights: [10, 784]
fc_weights = model.fc.weight.detach()
fc_bias = model.fc.bias.detach()

# Symmetric INT8 quantization of FC weights
fc_weight_multiplier = 127.0 / fc_weights.abs().max().item()

q_fc_weights = torch.round(
    fc_weights * fc_weight_multiplier
).clamp(-127, 127).to(torch.int8)

# Bias scale matches FC input and FC weight scales
bias_multiplier = fc_input_multiplier * fc_weight_multiplier

q_fc_bias = torch.round(
    fc_bias * bias_multiplier
).to(torch.int32)

print("FC input multiplier:", fc_input_multiplier)
print("FC weight multiplier:", fc_weight_multiplier)
print("FC bias multiplier:", bias_multiplier)
print("FC weight shape:", tuple(q_fc_weights.shape))
print("FC bias count:", len(q_fc_bias))
print("FC bias values:", q_fc_bias.tolist())

# Export weights as signed INT8 hexadecimal
with open("fc_weights.mem", "w") as f:
    for value in q_fc_weights.reshape(-1).tolist():
        f.write(f"{int(value) & 0xFF:02X}\n")

# Export biases as signed INT32 hexadecimal
with open("fc_bias_int32.mem", "w") as f:
    for value in q_fc_bias.tolist():
        f.write(f"{int(value) & 0xFFFFFFFF:08X}\n")

print("\nSaved fc_weights.mem")
print("Saved fc_bias_int32.mem")

