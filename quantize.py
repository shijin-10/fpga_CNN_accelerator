import torch
import numpy as np
from model import CNN
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


# --------------------------------
# Load trained model
# --------------------------------

model = CNN()

model.load_state_dict(
    torch.load(
        "cnn_mnist.pth",
        map_location="cpu"
    )
)

model.eval()


# --------------------------------
# INT8 quantization function
# --------------------------------

def quantize_tensor(tensor):

    max_value = torch.max(torch.abs(tensor))

    scale = 127.0 / max_value

    quantized = torch.round(
        tensor * scale
    ).clamp(-128, 127)

    return quantized.to(torch.int8), scale


# --------------------------------
# Quantize each layer
# --------------------------------

print("\n===== INT8 QUANTIZATION =====\n")

quantized_weights = {}

for name, parameter in model.named_parameters():

    q_tensor, scale = quantize_tensor(
        parameter.data
    )

    quantized_weights[name] = q_tensor

    print(name)

    print(
        "Original range:",
        parameter.min().item(),
        "to",
        parameter.max().item()
    )

    print(
        "INT8 range:",
        q_tensor.min().item(),
        "to",
        q_tensor.max().item()
    )

    print(
        "Scale:",
        scale
    )

    print()


# --------------------------------
# Save quantized weights
# --------------------------------

torch.save(
    quantized_weights,
    "cnn_int8_weights.pth"
)

print("INT8 weights saved as cnn_int8_weights.pth")
