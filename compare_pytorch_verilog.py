
import os
import re
import csv
import subprocess
import cv2
import numpy as np
import torch
from PIL import Image
from model import CNN

# ==================================================
# PROJECT CONFIGURATION
# ==================================================

PROJECT = os.path.expanduser("~/fpga-cnn")

# Change this if your test images are stored elsewhere.
IMAGE_DIR = os.path.join(PROJECT, "test_images")

MODEL_PATH = os.path.join(PROJECT, "cnn_mnist.pth")
INPUT_SCALE = 45.011730688743526

MEM_PATH = os.path.join(PROJECT, "mnist_image.mem")
PROCESSED_PATH = os.path.join(PROJECT, "processed_input.png")
RESULTS_PATH = os.path.join(PROJECT, "comparison_results.csv")
SIM_PATH = os.path.join(PROJECT, "cnn_sim")

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
# LOAD PYTORCH MODEL
# ==================================================

def load_pytorch_model():
    if not os.path.isfile(MODEL_PATH):
        raise FileNotFoundError(
            f"Model weights not found: {MODEL_PATH}"
        )

    model = CNN()

    try:
        state_dict = torch.load(
            MODEL_PATH,
            map_location="cpu",
            weights_only=True
        )
    except TypeError:
        # Compatibility with older PyTorch versions.
        state_dict = torch.load(
            MODEL_PATH,
            map_location="cpu"
        )

    model.load_state_dict(state_dict)
    model.eval()

    print("PyTorch model loaded successfully.")
    print("PyTorch version:", torch.__version__)

    return model


# ==================================================
# IMAGE PREPROCESSING
# Same preprocessing for both implementations
# ==================================================

def preprocess_image(path):
    image = cv2.imread(path)

    if image is None:
        raise ValueError(f"Cannot read image: {path}")

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    lower_blue = np.array([90, 35, 20])
    upper_blue = np.array([145, 255, 255])

    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    kernel_open = np.ones((3, 3), np.uint8)
    kernel_close = np.ones((5, 5), np.uint8)

    mask = cv2.morphologyEx(
        mask, cv2.MORPH_OPEN, kernel_open, iterations=1
    )
    mask = cv2.morphologyEx(
        mask, cv2.MORPH_CLOSE, kernel_close, iterations=2
    )

    count, labels, stats, centroids = (
        cv2.connectedComponentsWithStats(mask, 8)
    )

    if count <= 1:
        raise ValueError(
            "No blue digit detected. Check the image and lighting."
        )

    # Select the largest non-background component.
    largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])

    x = stats[largest, cv2.CC_STAT_LEFT]
    y = stats[largest, cv2.CC_STAT_TOP]
    w = stats[largest, cv2.CC_STAT_WIDTH]
    h = stats[largest, cv2.CC_STAT_HEIGHT]

    margin = max(12, int(0.20 * max(w, h)))

    x1 = max(0, x - margin)
    y1 = max(0, y - margin)
    x2 = min(mask.shape[1], x + w + margin)
    y2 = min(mask.shape[0], y + h + margin)

    crop = mask[y1:y2, x1:x2]

    # Recalculate bounds inside the crop.
    ys, xs = np.where(crop > 0)

    if len(xs) == 0:
        raise ValueError("Digit disappeared during preprocessing.")

    crop = crop[
        ys.min():ys.max() + 1,
        xs.min():xs.max() + 1
    ]

    height, width = crop.shape

    # Resize digit to fit inside a 20x20 region.
    scale = 20.0 / max(height, width)

    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))

    resized = cv2.resize(
        crop,
        (new_width, new_height),
        interpolation=cv2.INTER_AREA
    )

    # Centre digit on a 28x28 black canvas.
    canvas = np.zeros((28, 28), dtype=np.uint8)

    x_offset = (28 - new_width) // 2
    y_offset = (28 - new_height) // 2

    canvas[
        y_offset:y_offset + new_height,
        x_offset:x_offset + new_width
    ] = resized

    return canvas


# ==================================================
# SAVE INPUT FOR VERILOG
# ==================================================

def write_verilog_memory(canvas):
    normalized = (
        (canvas.astype(np.float32) / 255.0) - 0.1307
    ) / 0.3081

    quantized = np.round(
        normalized * INPUT_SCALE
    ).astype(np.int32)

    quantized = np.clip(quantized, -128, 127)

    with open(MEM_PATH, "w") as file:
        for value in quantized.flatten():
            # Write each signed INT8 value as 2-digit hex.
            file.write(f"{int(value) & 0xFF:02x}\n")


# ==================================================
# PYTORCH PREDICTION
# ==================================================

def predict_pytorch(model, canvas):
    pixels = canvas.astype(np.float32) / 255.0

    # Shape: batch x channel x height x width
    tensor = torch.from_numpy(pixels).unsqueeze(0).unsqueeze(0)

    # Same normalization used for MNIST training.
    tensor = (tensor - 0.1307) / 0.3081

    with torch.no_grad():
        scores = model(tensor)
        prediction = int(torch.argmax(scores, dim=1).item())

    return prediction, scores[0].cpu().numpy()


# ==================================================
# COMPILE VERILOG
# ==================================================

def compile_verilog():
    source_paths = [
        os.path.join(PROJECT, source)
        for source in SOURCES
    ]

    for source in source_paths:
        if not os.path.isfile(source):
            raise FileNotFoundError(
                f"Verilog source not found: {source}"
            )

    command = [
        "iverilog",
        "-g2012",
        "-o",
        SIM_PATH,
        *source_paths
    ]

    print("\nCompiling Verilog design...")

    result = subprocess.run(
        command,
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
# VERILOG PREDICTION
# ==================================================

def predict_verilog():
    result = subprocess.run(
        ["vvp", SIM_PATH],
        cwd=PROJECT,
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        print(result.stdout)
        print(result.stderr)
        raise RuntimeError("Verilog simulation failed.")

    output = result.stdout + "\n" + result.stderr

    match = re.search(
        r"Predicted digit:\s*(\d+)",
        output,
        re.IGNORECASE
    )

    if not match:
        print(output)
        raise ValueError(
            "Could not find 'Predicted digit:' in simulation output."
        )

    return int(match.group(1))


# ==================================================
# MAIN COMPARISON
# ==================================================

def main():
    if not os.path.isdir(IMAGE_DIR):
        raise FileNotFoundError(
            f"Image folder not found: {IMAGE_DIR}\n"
            "Edit IMAGE_DIR near the top of this script."
        )

    model = load_pytorch_model()
    compile_verilog()

    image_files = sorted([
        name for name in os.listdir(IMAGE_DIR)
        if name.lower().endswith(
            (".png", ".jpg", ".jpeg", ".bmp")
        )
    ])

    if not image_files:
        raise RuntimeError(
            f"No images found in {IMAGE_DIR}"
        )

    results = []

    print("\n" + "=" * 76)
    print("PYTORCH vs VERILOG CNN COMPARISON")
    print("=" * 76)

    for filename in image_files:
        path = os.path.join(IMAGE_DIR, filename)

        # Extract actual digit from filenames such as 7.jpg or 7.jpg.jpeg.
        label_match = re.match(r"\s*(\d)", filename)

        if not label_match:
            print(f"Skipping {filename}: filename must start with its label.")
            continue

        actual = int(label_match.group(1))

        try:
            # Process each original image only once.
            canvas = preprocess_image(path)

            # Save the exact processed image for inspection.
            cv2.imwrite(PROCESSED_PATH, canvas)

            # Run PyTorch on this processed image.
            pytorch_pred, scores = predict_pytorch(model, canvas)

            # Write the quantized version of that same image for RTL.
            write_verilog_memory(canvas)

            # Run Verilog simulation.
            verilog_pred = predict_verilog()

            match = pytorch_pred == verilog_pred
            pytorch_correct = pytorch_pred == actual
            verilog_correct = verilog_pred == actual

            print(
                f"{filename:18} | Actual: {actual} | "
                f"PyTorch: {pytorch_pred} | "
                f"Verilog: {verilog_pred} | "
                f"Match: {'YES' if match else 'NO'}"
            )

            results.append({
                "filename": filename,
                "actual_label": actual,
                "pytorch_prediction": pytorch_pred,
                "verilog_prediction": verilog_pred,
                "predictions_match": match,
                "pytorch_correct": pytorch_correct,
                "verilog_correct": verilog_correct,
            })

        except Exception as error:
            print(f"ERROR processing {filename}: {error}")

    if not results:
        raise RuntimeError(
            "No images were successfully tested."
        )

    with open(RESULTS_PATH, "w", newline="") as file:
        fieldnames = list(results[0].keys())
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    total = len(results)
    matches = sum(r["predictions_match"] for r in results)
    pytorch_correct = sum(r["pytorch_correct"] for r in results)
    verilog_correct = sum(r["verilog_correct"] for r in results)

    print("\n" + "=" * 76)
    print("FINAL RESULTS")
    print("=" * 76)
    print(f"Images tested:               {total}")
    print(
        f"PyTorch vs Verilog matches:  {matches}/{total} "
        f"({100 * matches / total:.2f}%)"
    )
    print(
        f"PyTorch accuracy:            {pytorch_correct}/{total} "
        f"({100 * pytorch_correct / total:.2f}%)"
    )
    print(
        f"Verilog accuracy:            {verilog_correct}/{total} "
        f"({100 * verilog_correct / total:.2f}%)"
    )
    print(f"\nDetailed results saved to: {RESULTS_PATH}")


if __name__ == "__main__":
    main()

