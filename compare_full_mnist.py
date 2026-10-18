
import os
import re
import csv
import argparse
import subprocess
import numpy as np
import torch
from torchvision import datasets
from model import CNN

# ==================================================
# CONFIGURATION
# ==================================================

PROJECT = os.path.expanduser("~/fpga-cnn")

MODEL_PATH = os.path.join(PROJECT, "cnn_mnist.pth")
MEM_PATH = os.path.join(PROJECT, "mnist_image.mem")
SIM_PATH = os.path.join(PROJECT, "cnn_sim_full_test")
CSV_PATH = os.path.join(PROJECT, "full_mnist_results.csv")

INPUT_SCALE = 45.011730688743526

SOURCES = [
    "cnn_top_tb.v",
    "cnn_top.v",
    "conv1_8filters_bias.v",
    "maxpool_2x2.v",
    "requantize_array.v",
    "conv2_16filters.v",
    "maxpool_2x2_conv2.v",
    "fc_layer.v",
]

# ==================================================
# PYTORCH MODEL
# ==================================================

def load_model():
    model = CNN()

    try:
        weights = torch.load(
            MODEL_PATH,
            map_location="cpu",
            weights_only=True
        )
    except TypeError:
        weights = torch.load(
            MODEL_PATH,
            map_location="cpu"
        )

    model.load_state_dict(weights)
    model.eval()

    return model


# ==================================================
# COMPILE VERILOG ONCE
# ==================================================

def compile_verilog():
    paths = [
        os.path.join(PROJECT, source)
        for source in SOURCES
    ]

    for path in paths:
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"Missing Verilog source: {path}"
            )

    result = subprocess.run(
        ["iverilog", "-g2012", "-o", SIM_PATH, *paths],
        cwd=PROJECT,
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        print(result.stdout)
        print(result.stderr)
        raise RuntimeError("Verilog compilation failed.")

    print("Verilog compilation successful.")


# ==================================================
# PREPARE THE SAME MNIST IMAGE FOR BOTH MODELS
# ==================================================

def prepare_image(pil_image):
    # MNIST images are grayscale, 28 x 28.
    pixels = np.asarray(
        pil_image, dtype=np.float32
    ) / 255.0

    # Same normalization used during model training.
    normalized = (pixels - 0.1307) / 0.3081

    # PyTorch receives the normalized floating-point image.
    tensor = torch.from_numpy(
        normalized.copy()
    ).unsqueeze(0).unsqueeze(0)

    # Verilog receives its quantized INT8 representation.
    quantized = np.round(
        normalized * INPUT_SCALE
    ).astype(np.int32)

    quantized = np.clip(
        quantized, -128, 127
    )

    return tensor, quantized


def write_memory(quantized):
    with open(MEM_PATH, "w") as file:
        for value in quantized.flatten():
            file.write(f"{int(value) & 0xFF:02x}\n")


# ==================================================
# PREDICTION FUNCTIONS
# ==================================================

def predict_pytorch(model, tensor):
    with torch.no_grad():
        scores = model(tensor)
        return int(scores.argmax(dim=1).item())


def predict_verilog():
    result = subprocess.run(
        ["vvp", SIM_PATH],
        cwd=PROJECT,
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        raise RuntimeError(
            result.stderr or "Verilog simulation failed."
        )

    output = result.stdout + "\n" + result.stderr

    match = re.search(
        r"Predicted digit:\s*(\d+)",
        output
    )

    if not match:
        raise RuntimeError(
            "Prediction not found in Verilog output."
        )

    return int(match.group(1))


# ==================================================
# FULL TEST-SET EVALUATION
# ==================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--limit",
        type=int,
        default=10000,
        help="Number of images to test (default: 10000)"
    )
    args = parser.parse_args()

    if not 1 <= args.limit <= 10000:
        raise ValueError("--limit must be between 1 and 10000")

    if not os.path.isfile(MODEL_PATH):
        raise FileNotFoundError(MODEL_PATH)

    model = load_model()

    print("Loading official MNIST test set...")

    dataset = datasets.MNIST(
        root=os.path.join(PROJECT, "data"),
        train=False,
        download=False
    )

    total = min(args.limit, len(dataset))

    print("MNIST test images available:", len(dataset))
    print("Images selected for this run:", total)

    compile_verilog()

    # Preserve the existing input memory file.
    old_memory = None

    if os.path.exists(MEM_PATH):
        with open(MEM_PATH, "rb") as file:
            old_memory = file.read()

    float_correct = 0
    verilog_correct = 0
    agreement = 0
    completed = 0
    errors = 0

    # Write the header and save each result as it completes.
    with open(CSV_PATH, "w", newline="") as file:
        fields = [
            "index",
            "actual_label",
            "pytorch_prediction",
            "verilog_prediction",
            "predictions_match",
            "pytorch_correct",
            "verilog_correct",
        ]

        writer = csv.DictWriter(
            file, fieldnames=fields
        )
        writer.writeheader()
        file.flush()

        try:
            for index in range(total):
                image, label = dataset[index]

                try:
                    tensor, quantized = prepare_image(image)

                    pytorch_pred = predict_pytorch(
                        model, tensor
                    )

                    write_memory(quantized)

                    verilog_pred = predict_verilog()

                    match = pytorch_pred == verilog_pred
                    p_correct = pytorch_pred == label
                    v_correct = verilog_pred == label

                    float_correct += int(p_correct)
                    verilog_correct += int(v_correct)
                    agreement += int(match)
                    completed += 1

                    writer.writerow({
                        "index": index,
                        "actual_label": label,
                        "pytorch_prediction": pytorch_pred,
                        "verilog_prediction": verilog_pred,
                        "predictions_match": match,
                        "pytorch_correct": p_correct,
                        "verilog_correct": v_correct,
                    })

                    # Keep results if the process is interrupted.
                    file.flush()

                except Exception as error:
                    errors += 1
                    print(
                        f"\nERROR at image {index}: {error}"
                    )

                if (index + 1) % 100 == 0 or index + 1 == total:
                    print(
                        f"Progress: {index + 1}/{total} | "
                        f"Completed: {completed} | "
                        f"Errors: {errors}"
                    )

        finally:
            if old_memory is not None:
                with open(MEM_PATH, "wb") as backup:
                    backup.write(old_memory)
            elif os.path.exists(MEM_PATH):
                os.remove(MEM_PATH)

    print("\n========== FINAL RESULTS ==========")
    print("Images requested:", total)
    print("Images completed:", completed)
    print("Errors:", errors)

    if completed:
        print(
            f"PyTorch accuracy: "
            f"{100 * float_correct / completed:.2f}%"
        )
        print(
            f"Verilog accuracy: "
            f"{100 * verilog_correct / completed:.2f}%"
        )
        print(
            f"Prediction agreement: "
            f"{100 * agreement / completed:.2f}%"
        )

    print("CSV results:", CSV_PATH)


if __name__ == "__main__":
    main()
