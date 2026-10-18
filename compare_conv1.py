import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from model import CNN

# Load trained model
model = CNN()
model.load_state_dict(
    torch.load("cnn_mnist.pth", map_location="cpu")
)
model.eval()

# Load the same MNIST image
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
image = image.unsqueeze(0)

# Match the INT8 quantization used by Verilog
input_scale = 127.0 / torch.max(torch.abs(image))
q_image = torch.round(image * input_scale)
q_image = q_image.clamp(-128, 127).to(torch.int8)

w = model.conv1.weight.data
weight_scale = 127.0 / torch.max(torch.abs(w))

q_weights = torch.round(w * weight_scale)
q_weights = q_weights.clamp(-128, 127).to(torch.int8)

# INT32 bias in accumulator units
q_bias = torch.round(
    model.conv1.bias.data * input_scale * weight_scale
).to(torch.int32)

# Calculate integer-domain convolution
conv = F.conv2d(
    q_image.float(),
    q_weights.float(),
    bias=None,
    padding=1
)

conv = conv + q_bias.float().view(1, 8, 1, 1)
relu = torch.relu(conv)

# MaxPool reference
pool = F.max_pool2d(relu, 2, 2)

print("MNIST label:", label)
print("\nPython Conv1 + Bias + ReLU — Output[0]:")

for f in range(8):
    print(f"Filter {f}: {int(relu[0, f, 0, 0].item())}")

print("\nPython MaxPool — Output[0]:")

for f in range(8):
    print(f"Filter {f}: {int(pool[0, f, 0, 0].item())}")
