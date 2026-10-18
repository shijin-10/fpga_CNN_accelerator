import torch
from model import CNN


# -------------------------
# Load trained model
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
# Quantization function
# -------------------------

def quantize_tensor(tensor):

    max_value = torch.max(
        torch.abs(tensor)
    )

    scale = 127.0 / max_value

    quantized = torch.round(
        tensor * scale
    ).clamp(-128, 127)

    return quantized.to(torch.int8), scale


# -------------------------
# Export Conv1 weights
# -------------------------

weights = model.conv1.weight.data

q_weights, weight_scale = quantize_tensor(
    weights
)


with open("conv1_weights.mem", "w") as f:

    for value in q_weights.flatten():

        value_int = int(value.item())

        if value_int < 0:
            value_int += 256

        f.write(f"{value_int:02X}\n")


print("Conv1 weights exported.")
print("Weight scale:", weight_scale.item())


# -------------------------
# Export Conv1 biases
# -------------------------

bias = model.conv1.bias.data

q_bias, bias_scale = quantize_tensor(
    bias
)


with open("conv1_bias.mem", "w") as f:

    for value in q_bias.flatten():

        value_int = int(value.item())

        if value_int < 0:
            value_int += 256

        f.write(f"{value_int:02X}\n")


print("Conv1 biases exported.")
print("Bias scale:", bias_scale.item())


# -------------------------
# Create test pixels
# -------------------------

test_pixels = torch.tensor([
    10, 20, 30,
    40, 50, 60,
    70, 80, 90
], dtype=torch.int8)


with open("test_pixels.mem", "w") as f:

    for value in test_pixels:

        value_int = int(value.item())

        if value_int < 0:
            value_int += 256

        f.write(f"{value_int:02X}\n")


print("Test pixels exported.")

print("\nFiles created:")
print("  conv1_weights.mem")
print("  conv1_bias.mem")
print("  test_pixels.mem")
