import torch
from train import CNN


# Load the trained model
model = CNN()

model.load_state_dict(
    torch.load(
        "cnn_mnist.pth",
        map_location="cpu"
    )
)

model.eval()


# Print the layers
print("\n===== MODEL WEIGHTS =====\n")

for name, parameter in model.named_parameters():

    print(f"{name}")
    print(f"Shape: {parameter.shape}")
    print(f"Minimum: {parameter.min().item():.6f}")
    print(f"Maximum: {parameter.max().item():.6f}")
    print(f"Mean: {parameter.mean().item():.6f}")
    print()

