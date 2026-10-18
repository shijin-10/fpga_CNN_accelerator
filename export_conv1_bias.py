import torch
from model import CNN

model = CNN()
model.load_state_dict(
    torch.load("cnn_mnist.pth", map_location="cpu")
)
model.eval()

# Match the input quantization used in quantized_pipeline.py
from torchvision import datasets, transforms

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

dataset = datasets.MNIST(
    "./data",
    train=False,
    download=True,
    transform=transform
)

image, label = dataset[0]

input_scale = 127.0 / torch.max(torch.abs(image))

weights = model.conv1.weight.data
weight_scale = 127.0 / torch.max(torch.abs(weights))

# Bias in INT32 accumulator units
bias = model.conv1.bias.data
q_bias = torch.round(bias * input_scale * weight_scale).to(torch.int32)

with open("conv1_bias_int32.mem", "w") as f:
    for value in q_bias:
        n = int(value.item()) & 0xFFFFFFFF
        f.write(f"{n:08X}\n")

print("MNIST label:", label)
print("Input scale:", input_scale.item())
print("Weight scale:", weight_scale.item())
print("INT32 biases:", q_bias.tolist())
print("Saved: conv1_bias_int32.mem")
