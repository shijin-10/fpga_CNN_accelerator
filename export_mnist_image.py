import torch
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

input_max = torch.max(torch.abs(image))
input_scale = 127.0 / input_max

q_image = torch.round(image * input_scale)
q_image = q_image.clamp(-128, 127).to(torch.int8)

with open("mnist_image.mem", "w") as f:
    for value in q_image.flatten():
        f.write(format(int(value) & 0xFF, "02X") + "\n")

print("Label:", label)
print("Saved: mnist_image.mem")
print("Pixel count:", q_image.numel())
