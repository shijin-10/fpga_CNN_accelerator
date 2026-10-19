
import torch
from model import CNN

INPUT_SCALE = 45.011730688743526

model = CNN()
model.load_state_dict(
    torch.load("cnn_mnist.pth", map_location="cpu", weights_only=True)
)
model.eval()

weights = model.conv1.weight.detach()
bias = model.conv1.bias.detach()

weight_scale = 127.0 / weights.abs().max().item()

q_weights = torch.round(
    weights * weight_scale
).clamp(-127, 127).to(torch.int8)

q_bias = torch.round(
    bias * INPUT_SCALE * weight_scale
).to(torch.int32)

with open("conv1_weights.mem", "w") as f:
    for value in q_weights.reshape(-1).tolist():
        f.write(f"{int(value) & 0xFF:02X}\n")

with open("conv1_bias_int32.mem", "w") as f:
    for value in q_bias.tolist():
        f.write(f"{int(value) & 0xFFFFFFFF:08X}\n")

print("Fixed-scale Conv1 parameters exported")
print("Input scale:", INPUT_SCALE)
print("Weight scale:", weight_scale)
print("Weight count:", q_weights.numel())
print("Bias values:", q_bias.tolist())
