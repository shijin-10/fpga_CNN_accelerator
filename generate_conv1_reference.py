import torch
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
# Create the same 28x28 test image used in Verilog
# --------------------------------------------------
image = torch.tensor(
    [(i % 128) for i in range(784)],
    dtype=torch.float32
).reshape(1, 1, 28, 28)

# --------------------------------------------------
# Get trained Conv1 weights
# --------------------------------------------------
conv1_weight = model.conv1.weight.data

# Quantize Conv1 weights using our existing scale
max_value = torch.max(torch.abs(conv1_weight))
scale = 127.0 / max_value

q_weight = torch.round(conv1_weight * scale)
q_weight = q_weight.clamp(-128, 127)

# --------------------------------------------------
# Convert image to INT8
# --------------------------------------------------
q_image = image.to(torch.int8)

# Convert back to float for PyTorch convolution
# while preserving the integer values.
q_image_float = q_image.float()
q_weight_float = q_weight.float()

# --------------------------------------------------
# Conv1
# padding=1
# --------------------------------------------------
conv_output = torch.nn.functional.conv2d(
    q_image_float,
    q_weight_float,
    bias=None,
    padding=1
)

# --------------------------------------------------
# ReLU
# --------------------------------------------------
relu_output = torch.relu(conv_output)

# --------------------------------------------------
# MaxPool
# --------------------------------------------------
pool_output = torch.nn.functional.max_pool2d(
    relu_output,
    kernel_size=2,
    stride=2
)

# --------------------------------------------------
# Print dimensions
# --------------------------------------------------
print("Conv output shape:", conv_output.shape)
print("Pool output shape:", pool_output.shape)

# --------------------------------------------------
# Print first 3x3 pooled values of each filter
# --------------------------------------------------
for f in range(8):

    print()
    print("Filter", f)

    for row in range(3):
        values = []

        for col in range(3):
            values.append(
                int(pool_output[0, f, row, col])
            )

        print(values)

# --------------------------------------------------
# Save reference output
# --------------------------------------------------
with open("conv1_pool_reference.txt", "w") as f:

    for value in pool_output.flatten():
        f.write(str(int(value)) + "\n")

print()
print("Saved: conv1_pool_reference.txt")
