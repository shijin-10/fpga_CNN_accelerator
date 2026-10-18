
import os
import re
import subprocess
import csv

import cv2
import numpy as np
from PIL import Image

# ==================================================
# PROJECT CONFIGURATION
# ==================================================

PROJECT = os.path.expanduser("~/fpga-cnn")
IMAGE_DIR = os.path.join(PROJECT, "test_images")

INPUT_SCALE = 45.011730688743526

PROCESSED_PATH = os.path.join(PROJECT, "processed_input.png")
MEM_PATH = os.path.join(PROJECT, "mnist_image.mem")
RESULTS_PATH = os.path.join(PROJECT, "test_results.csv")

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
# IMAGE PREPROCESSING
# Same method as image_demo.py
# ==================================================

def preprocess_image(path):
    image = cv2.imread(path)

    if image is None:
        raise ValueError(f"Cannot read image: {path}")

    # Detect blue pen ink.
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    lower_blue = np.array([90, 35, 20], dtype=np.uint8)
    upper_blue = np.array([145, 255, 255], dtype=np.uint8)

    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    # Remove isolated noise.
    kernel = np.ones((3, 3), dtype=np.uint8)
    mask = cv2.morphologyEx(
        mask, cv2.MORPH_OPEN, kernel, iterations=1
    )

    # Join nearby pieces of handwriting.
    join_kernel = np.ones((5, 5), dtype=np.uint8)
    joined = cv2.morphologyEx(
        mask, cv2.MORPH_CLOSE, join_kernel, iterations=2
    )

    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        joined, connectivity=8
    )

    if count <= 1:
        raise ValueError("No blue handwriting detected")

    # Find the largest connected component.
    areas = stats[1:, cv2.CC_STAT_AREA]
    largest_label = 1 + int(np.argmax(areas))

    x = int(stats[largest_label, cv2.CC_STAT_LEFT])
    y = int(stats[largest_label, cv2.CC_STAT_TOP])
    w = int(stats[largest_label, cv2.CC_STAT_WIDTH])
    h = int(stats[largest_label, cv2.CC_STAT_HEIGHT])

    # Include a margin around the digit.
    margin = max(12, int(0.20 * max(w, h)))

    x1 = max(0, x - margin)
    y1 = max(0, y - margin)
    x2 = min(joined.shape[1], x + w + margin)
    y2 = min(joined.shape[0], y + h + margin)

    # Preserve the blue pixels inside the expanded crop.
    crop = mask[y1:y2, x1:x2]

    points = cv2.findNonZero(crop)

    if points is None:
        raise ValueError("Could not locate digit strokes")

    bx, by, bw, bh = cv2.boundingRect(points)
    digit = crop[by:by + bh, bx:bx + bw]

    dh, dw = digit.shape

    if dw == 0 or dh == 0:
        raise ValueError("Detected digit has invalid dimensions")

    # Fit the digit inside 20x20 while preserving aspect ratio.
    scale = min(20.0 / dw, 20.0 / dh)

    new_w = max(1, int(round(dw * scale)))
    new_h = max(1, int(round(dh * scale)))

    resized = cv2.resize(
        digit,
        (new_w, new_h),
        interpolation=cv2.INTER_AREA
    )

    # Centre the digit on a black 28x28 canvas.
    canvas = np.zeros((28, 28), dtype=np.uint8)

    offset_x = (28 - new_w) // 2
    offset_y = (28 - new_h) // 2

    canvas[
        offset_y:offset_y + new_h,
        offset_x:offset_x + new_w
    ] = resized

    # Save for visual inspection and PyTorch comparison.
    Image.fromarray(canvas).save(PROCESSED_PATH)

    # Normalize and quantize exactly as in image_demo.py.
    values = []

    for pixel in canvas.flatten():
        normalized = (
            (int(pixel) / 255.0) - 0.1307
        ) / 0.3081

        quantized = round(normalized * INPUT_SCALE)
        quantized = max(-128, min(127, quantized))

        values.append(quantized)

    if len(values) != 784:
        raise ValueError(
            f"Expected 784 pixels, got {len(values)}"
        )

    # Write signed INT8 values as hexadecimal bytes.
    with open(MEM_PATH, "w") as file:
        for value in values:
            file.write(f"{value & 0xFF:02x}\n")


# ==================================================
# COMPILE VERILOG
# ==================================================

def compile_verilog():
    print("Compiling Verilog CNN...")

    result = subprocess.run(
        ["iverilog", "-g2012", "-o", "cnn_sim"] + SOURCES,
        cwd=PROJECT,
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        raise RuntimeError(
            "Verilog compilation failed:\n" + result.stderr
        )

    print("Compilation successful.\n")


# ==================================================
# RUN ONE IMAGE
# ==================================================

def predict_image(path):
    # Every image must be preprocessed immediately before
    # running simulation because the testbench reads mnist_image.mem.
    preprocess_image(path)

    result = subprocess.run(
        ["vvp", "cnn_sim"],
        cwd=PROJECT,
        capture_output=True,
        text=True,
        timeout=120
    )

    if result.returncode != 0:
        raise RuntimeError(
            "Verilog simulation failed:\n" + result.stderr
        )

    match = re.search(
        r"Predicted digit:\s*(\d+)",
        result.stdout
    )

    if not match:
        raise RuntimeError(
            "Prediction not found in output:\n" + result.stdout
        )

    return int(match.group(1))


# ==================================================
# MAIN BATCH TEST
# ==================================================

def main():
    if not os.path.isdir(IMAGE_DIR):
        print("Image directory not found:", IMAGE_DIR)
        return

    # The filename must begin with the actual digit label.
    supported_extensions = (".png", ".jpg", ".jpeg", ".bmp")
    images = []

    for filename in sorted(os.listdir(IMAGE_DIR)):
        if not filename.lower().endswith(supported_extensions):
            continue

        label_match = re.match(r"([0-9])", filename)

        if label_match is None:
            print("Skipping unlabelled file:", filename)
            continue

        actual = int(label_match.group(1))
        path = os.path.join(IMAGE_DIR, filename)

        images.append((filename, path, actual))

    if not images:
        print("No labelled images found in:", IMAGE_DIR)
        return

    print("Images found:", len(images))
    print("Preprocessing: blue-ink extraction")
    print("Input dimensions: 28 x 28")
    print()

    try:
        compile_verilog()
    except Exception as error:
        print(error)
        return

    results = []

    for filename, path, actual in images:
        try:
            predicted = predict_image(path)
            correct = predicted == actual

            results.append({
                "filename": filename,
                "actual": actual,
                "predicted": predicted,
                "correct": correct
            })

            status = "CORRECT" if correct else "WRONG"

            print(
                f"{filename:20s} "
                f"Actual={actual} "
                f"Predicted={predicted} "
                f"{status}"
            )

        except Exception as error:
            print(f"{filename}: ERROR - {error}")

    # Save CSV results.
    with open(RESULTS_PATH, "w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "filename",
                "actual",
                "predicted",
                "correct"
            ]
        )

        writer.writeheader()
        writer.writerows(results)

    if not results:
        print("\nNo images were successfully tested.")
        return

    correct_count = sum(row["correct"] for row in results)
    total = len(results)
    incorrect_count = total - correct_count
    accuracy = 100.0 * correct_count / total

    print("\n========== TEST SUMMARY ==========")
    print("Images tested:", total)
    print("Correct:", correct_count)
    print("Incorrect:", incorrect_count)
    print(f"Accuracy: {accuracy:.2f}%")
    print("==================================")
    print("Results saved to:", RESULTS_PATH)


if __name__ == "__main__":
    main()
