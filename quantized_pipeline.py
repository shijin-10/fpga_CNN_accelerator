import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from model import CNN

# --------------------------------------------------
# Load trained model
# --------------------------------------------------
model = CNN()
model.load_state_dict(
    torch.load("cnn_mnist.pth", map_location="cpu")
)
model.eval()

# --------------------------------------------------
# Load ONE real MNIST image
# Same normalization used during training
# --------------------------------------------------
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

print("MNIST label:", label)

# image shape = [1, 28, 28]
image = image.unsqueeze(0)

# --------------------------------------------------
# Input quantization
# Symmetric INT8
# --------------------------------------------------
input_max = torch.max(torch.abs(image))
input_scale = 127.0 / input_max

q_image = torch.round(image * input_scale)
q_image = q_image.clamp(-128, 127).to(torch.int8)

print("Input scale:", input_scale.item())
print(
    "INT8 input range:",
    q_image.min().item(),
    "to",
    q_image.max().item()
)

# --------------------------------------------------
# Conv1 weight quantization
# --------------------------------------------------
w1 = model.conv1.weight.data

w1_max = torch.max(torch.abs(w1))
w1_scale = 127.0 / w1_max

q_w1 = torch.round(w1 * w1_scale)
q_w1 = q_w1.clamp(-128, 127).to(torch.int8)

print("Conv1 weight scale:", w1_scale.item())

# --------------------------------------------------
# Integer Conv1
# --------------------------------------------------
conv1_int = F.conv2d(
    q_image.float(),
    q_w1.float(),
    bias=None,
    padding=1
)

# --------------------------------------------------
# Correct INT32 bias
# --------------------------------------------------
bias1 = model.conv1.bias.data

q_bias1 = torch.round(
    bias1 * input_scale * w1_scale
)

print("Conv1 INT32 biases:")
print(q_bias1.to(torch.int32))

# Add bias
conv1_int = (
    conv1_int +
    q_bias1.reshape(1, 8, 1, 1)
)

# --------------------------------------------------
# ReLU
# --------------------------------------------------
relu1 = torch.relu(conv1_int)

# --------------------------------------------------
# MaxPool
# --------------------------------------------------
pool1 = F.max_pool2d(
    relu1,
    kernel_size=2,
    stride=2
)

print("Pool shape:", pool1.shape)

print(
    "Conv1+ReLU+Pool INT32 range:",
    pool1.min().item(),
    "to",
    pool1.max().item()
)

# --------------------------------------------------
# Determine scale for Conv2 input
# --------------------------------------------------
pool_max = torch.max(torch.abs(pool1))

pool_scale = 127.0 / pool_max

q_pool = torch.round(
    pool1 * pool_scale
)

q_pool = q_pool.clamp(-128, 127).to(torch.int8)

print("Conv2 input scale:", pool_scale.item())

print(
    "Conv2 INT8 input range:",
    q_pool.min().item(),
    "to",
    q_pool.max().item()
)

# --------------------------------------------------
# Save quantized Conv2 input
# --------------------------------------------------
with open("conv2_input.mem", "w") as f:

    for value in q_pool.flatten():
        value = int(value)

        f.write(
            format(value & 0xFF, "02X") + "\n"
        )

print("Saved: conv2_input.mem")

print("Expected values:", q_pool.numel())
