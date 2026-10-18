
import os
import re
import subprocess
import numpy as np
import torch
from torchvision import datasets
from model import CNN

PROJECT = os.path.expanduser("~/fpga-cnn")
SCALE = 45.011730688743526
MEM = os.path.join(PROJECT, "mnist_image.mem")
SIM = os.path.join(PROJECT, "cnn_sim_full_test")

model = CNN()
model.load_state_dict(torch.load(
    os.path.join(PROJECT, "cnn_mnist.pth"),
    map_location="cpu",
    weights_only=True
))
model.eval()

dataset = datasets.MNIST(
    root=os.path.join(PROJECT, "data"),
    train=False,
    download=False
)

old_memory = None
if os.path.exists(MEM):
    with open(MEM, "rb") as f:
        old_memory = f.read()

try:
    for index in [726]:
        image, label = dataset[index]

        pixels = np.asarray(image, dtype=np.float32) / 255.0
        normalized = (pixels - 0.1307) / 0.3081

        tensor = torch.from_numpy(
            normalized.copy()
        ).unsqueeze(0).unsqueeze(0)

        with torch.no_grad():
            pt_scores = model(tensor)[0].numpy()

        quantized = np.clip(
            np.round(normalized * SCALE), -128, 127
        ).astype(np.int32)

        with open(MEM, "w") as f:
            for value in quantized.flatten():
                f.write(f"{int(value) & 0xFF:02x}\n")

        result = subprocess.run(
            ["vvp", SIM],
            cwd=PROJECT,
            capture_output=True,
            text=True,
            check=True
        )

        output = result.stdout + "\n" + result.stderr
        matches = re.findall(
            r"Digit\s+(\d+):\s*(-?\d+)", output
        )
        rtl_scores = {int(d): int(s) for d, s in matches}

        print("\n" + "=" * 45)
        print(f"MNIST index: {index}, actual label: {label}")
        print("PyTorch prediction:", int(np.argmax(pt_scores)))
        print("Verilog prediction:", re.search(
            r"Predicted digit:\s*(\d+)", output
        ).group(1))

        print("\nDigit | PyTorch score | Verilog score")
        print("-" * 45)

        for digit in range(10):
            print(
                f"{digit:5} | {pt_scores[digit]:13.5f} | "
                f"{rtl_scores.get(digit, 'MISSING')}"
            )

finally:
    if old_memory is not None:
        with open(MEM, "wb") as f:
            f.write(old_memory)
    elif os.path.exists(MEM):
        os.remove(MEM)

