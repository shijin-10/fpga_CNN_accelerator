import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from model import CNN


# -------------------------
# Load original model
# -------------------------

model = CNN()

model.load_state_dict(
    torch.load(
        "cnn_mnist.pth",
        map_location="cpu"
    )
)

model.eval()


# -------------------------
# Load INT8 weights
# -------------------------

int8_weights = torch.load(
    "cnn_int8_weights.pth",
    map_location="cpu"
)


# -------------------------
# Dequantize weights
# -------------------------

scales = {}

for name, parameter in model.named_parameters():

    max_value = torch.max(
        torch.abs(parameter.data)
    )

    scales[name] = 127.0 / max_value


# Replace FP32 weights
# with dequantized INT8 weights

with torch.no_grad():

    for name, parameter in model.named_parameters():

        q = int8_weights[name].float()

        parameter.data = (
            q / scales[name]
        )


# -------------------------
# MNIST test dataset
# -------------------------

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        (0.1307,),
        (0.3081,)
    )
])

test_dataset = datasets.MNIST(
    root="./data",
    train=False,
    download=True,
    transform=transform
)

test_loader = DataLoader(
    test_dataset,
    batch_size=1000,
    shuffle=False
)


# -------------------------
# Test INT8 model
# -------------------------

correct = 0
total = 0

with torch.no_grad():

    for images, labels in test_loader:

        outputs = model(images)

        _, predicted = torch.max(
            outputs,
            1
        )

        total += labels.size(0)

        correct += (
            predicted == labels
        ).sum().item()


accuracy = 100 * correct / total

print()
print("==============================")
print("INT8 MODEL TEST")
print("==============================")
print(f"INT8 Accuracy: {accuracy:.2f}%")
print("==============================")
