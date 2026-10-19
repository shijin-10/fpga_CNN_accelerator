
import torch
from torchvision import datasets, transforms

INPUT_SCALE = 45.011730688743526

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

q_image = torch.round(
    image.to(torch.float64) * INPUT_SCALE
).clamp(-127, 127).to(torch.int8)

with open("mnist_image.mem", "w") as f:
    for value in q_image.flatten().tolist():
        f.write(f"{int(value) & 0xFF:02X}\n")

print("Label:", label)
print("Fixed input scale:", INPUT_SCALE)
print("Saved: mnist_image.mem")
print("Pixel count:", q_image.numel())
print("Quantized range:", int(q_image.min()), "to", int(q_image.max()))
