import torch
from model import CNN

# Load trained model
model = CNN()
model.load_state_dict(torch.load("cnn_mnist.pth", map_location="cpu"))
model.eval()

# -----------------------------------------
# Quantization function
# -----------------------------------------
def quantize_tensor(tensor):
    max_value = torch.max(torch.abs(tensor))
    scale = 127.0 / max_value

    quantized = torch.round(tensor * scale)
    quantized = quantized.clamp(-128, 127)

    return quantized.to(torch.int8), scale


# -----------------------------------------
# Conv2 weights
# Shape = [16, 8, 3, 3]
# -----------------------------------------
conv2_weights = model.conv2.weight.data

q_conv2, conv2_scale = quantize_tensor(conv2_weights)


# -----------------------------------------
# Export Conv2 weights
# -----------------------------------------
with open("conv2_weights.mem", "w") as f:

    for value in q_conv2.flatten():
        value = int(value)

        # Convert signed INT8 to hexadecimal
        hex_value = format(value & 0xFF, "02X")

        f.write(hex_value + "\n")


# -----------------------------------------
# Display information
# -----------------------------------------
print("Conv2 weight shape:", conv2_weights.shape)

print("Conv2 INT8 range:",
      q_conv2.min().item(),
      "to",
      q_conv2.max().item())

print("Conv2 scale:", conv2_scale)

print("Total Conv2 weights:",
      q_conv2.numel())

print("Saved: conv2_weights.mem")
