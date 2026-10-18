
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model import CNN

# Fixed multipliers from calibration
INPUT_SCALE = 45.011730688743526
POOL1_SCALE = 18.25425611513592
POOL2_SCALE = 11.704324914902063

BATCH_SIZE = 128
device = "cpu"

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

test_dataset = datasets.MNIST(
    root="./data",
    train=False,
    download=True,
    transform=transform
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

model = CNN()
model.load_state_dict(
    torch.load(
        "cnn_mnist.pth",
        map_location=device,
        weights_only=True
    )
)
model.eval()


def quantize_weight(weight):
    scale = 127.0 / weight.abs().max().item()
    q = torch.round(weight * scale).clamp(-127, 127)
    return q.to(torch.int64), scale


def quantize_bias(bias, input_scale, weight_scale):
    return torch.round(
        bias * input_scale * weight_scale
    ).to(torch.int64)


# Quantized weights and integer biases
w1, w1_scale = quantize_weight(model.conv1.weight)
w2, w2_scale = quantize_weight(model.conv2.weight)
wf, wf_scale = quantize_weight(model.fc.weight)

b1 = quantize_bias(model.conv1.bias, INPUT_SCALE, w1_scale)
b2 = quantize_bias(model.conv2.bias, POOL1_SCALE, w2_scale)
bf = quantize_bias(model.fc.bias, POOL2_SCALE, wf_scale)

w1 = w1.to(torch.float64)
w2 = w2.to(torch.float64)
wf = wf.to(torch.float64)

b1 = b1.to(torch.float64)
b2 = b2.to(torch.float64)
bf = bf.to(torch.float64)


def requantize(accumulator, input_scale, weight_scale, output_scale):
    real_values = accumulator / (input_scale * weight_scale)

    return torch.round(
        real_values * output_scale
    ).clamp(0, 127)


float_correct = 0
fixed_correct = 0
prediction_disagreements = 0
total = 0

print("Evaluating fixed-scale CNN reference...")

with torch.no_grad():
    for images, labels in test_loader:

        # Floating-point reference
        float_scores = model(images)
        float_predictions = float_scores.argmax(dim=1)

        # Quantize normalized input to signed INT8
        x = torch.round(
            images.to(torch.float64) * INPUT_SCALE
        ).clamp(-127, 127)

        # Conv1 + bias + ReLU + MaxPool
        x = F.conv2d(x, w1, b1, padding=1)
        x = F.relu(x)
        x = F.max_pool2d(x, 2, 2)

        # Requantize to fixed Pool1 INT8 activations
        x = requantize(
            x, INPUT_SCALE, w1_scale, POOL1_SCALE
        )

        # Conv2 + bias + ReLU + MaxPool
        x = F.conv2d(x, w2, b2, padding=1)
        x = F.relu(x)
        x = F.max_pool2d(x, 2, 2)

        # Requantize to fixed Pool2 INT8 activations
        x = requantize(
            x, POOL1_SCALE, w2_scale, POOL2_SCALE
        )

        # Fully connected layer
        x = x.flatten(1)
        scores = F.linear(x, wf, bf)

        fixed_predictions = scores.argmax(dim=1)

        float_correct += (
            float_predictions == labels
        ).sum().item()

        fixed_correct += (
            fixed_predictions == labels
        ).sum().item()

        prediction_disagreements += (
            float_predictions != fixed_predictions
        ).sum().item()

        total += len(labels)

print("\n========== FIXED-PIPELINE RESULTS ==========")
print("Test images:", total)
print(f"Float accuracy: {100 * float_correct / total:.2f}%")
print(f"Fixed-scale reference accuracy: {100 * fixed_correct / total:.2f}%")
print(
    "Prediction agreement: "
    f"{100 * (total - prediction_disagreements) / total:.2f}%"
)
print("Prediction disagreements:", prediction_disagreements)
