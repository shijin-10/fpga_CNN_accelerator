
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model import CNN

# --------------------------------------------------
# Configuration
# --------------------------------------------------
torch.manual_seed(42)
device = torch.device("cpu")
BATCH_SIZE = 128
CALIBRATION_IMAGES = 1000

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

test_dataset = datasets.MNIST(
    root="./data",
    train=False,
    download=True,
    transform=transform
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

model = CNN().to(device)
model.load_state_dict(
    torch.load(
        "cnn_mnist.pth",
        map_location=device,
        weights_only=True
    )
)
model.eval()


# --------------------------------------------------
# Floating-point reference
# --------------------------------------------------
def forward_float(x):
    x = model.conv1(x)
    x = F.relu(x)
    x = model.pool(x)

    x = model.conv2(x)
    x = F.relu(x)
    x = model.pool(x)

    x = x.flatten(1)
    return model.fc(x)


# --------------------------------------------------
# Fixed symmetric INT8 quantization helpers
# --------------------------------------------------
def make_scale(max_abs):
    return 127.0 / max(float(max_abs), 1e-12)


def quantize_dequantize(x, scale, limit=127):
    q = torch.round(x * scale).clamp(-limit, limit)
    return q / scale


def quantized_bias(bias, input_scale, weight_scale):
    accumulator_scale = input_scale * weight_scale

    q_bias = torch.round(
        bias * accumulator_scale
    ).clamp(-(2**31), 2**31 - 1)

    return q_bias / accumulator_scale


def weight_scale_for(weight):
    return make_scale(weight.detach().abs().max().item())


# --------------------------------------------------
# Step A: Calibrate fixed activation scales
# --------------------------------------------------
print("Calibrating activation scales...")

max_input = 0.0
max_pool1 = 0.0
max_pool2 = 0.0
seen = 0

with torch.no_grad():
    for images, _ in train_loader:
        remaining = CALIBRATION_IMAGES - seen

        if remaining <= 0:
            break

        images = images[:remaining].to(device)
        seen += len(images)

        max_input = max(
            max_input,
            images.abs().max().item()
        )

        x = model.conv1(images)
        x = F.relu(x)
        x = model.pool(x)

        max_pool1 = max(
            max_pool1,
            x.abs().max().item()
        )

        x = model.conv2(x)
        x = F.relu(x)
        x = model.pool(x)

        max_pool2 = max(
            max_pool2,
            x.abs().max().item()
        )

input_scale = make_scale(max_input)
pool1_scale = make_scale(max_pool1)
pool2_scale = make_scale(max_pool2)

print("Calibration images:", seen)
print("Input INT8 scale:", input_scale)
print("Pool1 INT8 scale:", pool1_scale)
print("Pool2 INT8 scale:", pool2_scale)

# Fixed weight scales for the trained model
w1_scale = weight_scale_for(model.conv1.weight)
w2_scale = weight_scale_for(model.conv2.weight)
wf_scale = weight_scale_for(model.fc.weight)

print("Conv1 weight scale:", w1_scale)
print("Conv2 weight scale:", w2_scale)
print("FC weight scale:", wf_scale)


# --------------------------------------------------
# Step B: Quantized inference reference
# --------------------------------------------------
def forward_quantized_reference(images):
    # Quantize and dequantize input
    x = quantize_dequantize(images, input_scale)

    # Conv1: quantized weights and INT32-scaled bias
    w1 = quantize_dequantize(
        model.conv1.weight, w1_scale
    )
    b1 = quantized_bias(
        model.conv1.bias, input_scale, w1_scale
    )

    x = F.conv2d(x, w1, b1, padding=1)
    x = F.relu(x)
    x = model.pool(x)

    # Requantize activation using fixed calibration scale
    x = quantize_dequantize(x, pool1_scale)

    # Conv2
    w2 = quantize_dequantize(
        model.conv2.weight, w2_scale
    )
    b2 = quantized_bias(
        model.conv2.bias, pool1_scale, w2_scale
    )

    x = F.conv2d(x, w2, b2, padding=1)
    x = F.relu(x)
    x = model.pool(x)

    x = quantize_dequantize(x, pool2_scale)

    # Fully connected layer
    x = x.flatten(1)

    wf = quantize_dequantize(
        model.fc.weight, wf_scale
    )
    bf = quantized_bias(
        model.fc.bias, pool2_scale, wf_scale
    )

    return F.linear(x, wf, bf)


# --------------------------------------------------
# Step C: Evaluate both models on the test set
# --------------------------------------------------
print("\nEvaluating MNIST test dataset...")

float_correct = 0
quant_correct = 0
total = 0
prediction_mismatches = 0

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        float_scores = forward_float(images)
        quant_scores = forward_quantized_reference(images)

        float_predictions = float_scores.argmax(dim=1)
        quant_predictions = quant_scores.argmax(dim=1)

        float_correct += (
            float_predictions == labels
        ).sum().item()

        quant_correct += (
            quant_predictions == labels
        ).sum().item()

        prediction_mismatches += (
            float_predictions != quant_predictions
        ).sum().item()

        total += labels.size(0)

float_accuracy = 100.0 * float_correct / total
quant_accuracy = 100.0 * quant_correct / total
prediction_agreement = (
    100.0 * (total - prediction_mismatches) / total
)

print("\n========== RESULTS ==========")
print("Test images:", total)
print(f"Float model accuracy: {float_accuracy:.2f}%")
print(f"Quantized-reference accuracy: {quant_accuracy:.2f}%")
print(f"Float/quantized prediction agreement: {prediction_agreement:.2f}%")
print("Prediction disagreements:", prediction_mismatches)
