
import os
import re
import subprocess
import tkinter as tk
from tkinter import filedialog, messagebox

import cv2
import numpy as np
from PIL import Image, ImageTk

# ==================================================
# PROJECT CONFIGURATION
# ==================================================

PROJECT = os.path.expanduser("~/fpga-cnn")
INPUT_SCALE = 45.011730688743526

PROCESSED_PATH = os.path.join(PROJECT, "processed_input.png")
MEM_PATH = os.path.join(PROJECT, "mnist_image.mem")

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
# ==================================================

def preprocess_image(path):
    """
    Extract blue handwriting, crop around the digit,
    preserve nearby strokes, and create a 28x28 image.

    Output:
        PIL grayscale image, white digit on black background.
    """

    image = cv2.imread(path)

    if image is None:
        raise ValueError(f"Cannot read image:\n{path}")

    # Detect blue pen ink.
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    lower_blue = np.array([90, 35, 20], dtype=np.uint8)
    upper_blue = np.array([145, 255, 255], dtype=np.uint8)

    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    # Remove isolated noise while retaining the main strokes.
    kernel = np.ones((3, 3), dtype=np.uint8)
    mask = cv2.morphologyEx(
        mask, cv2.MORPH_OPEN, kernel, iterations=1
    )

    # Join nearby pieces of the same handwritten digit.
    # This can help digits whose strokes are slightly disconnected.
    join_kernel = np.ones((5, 5), dtype=np.uint8)
    joined = cv2.morphologyEx(
        mask, cv2.MORPH_CLOSE, join_kernel, iterations=2
    )

    # Locate connected components.
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        joined, connectivity=8
    )

    if count <= 1:
        raise ValueError(
            "No blue handwriting detected.\n"
            "Try a clearer image with darker blue ink."
        )

    # Choose the largest component as the main digit.
    areas = stats[1:, cv2.CC_STAT_AREA]
    largest_label = 1 + int(np.argmax(areas))

    x = stats[largest_label, cv2.CC_STAT_LEFT]
    y = stats[largest_label, cv2.CC_STAT_TOP]
    w = stats[largest_label, cv2.CC_STAT_WIDTH]
    h = stats[largest_label, cv2.CC_STAT_HEIGHT]

    # Include nearby components within the digit's bounding box
    # expanded by a margin. This helps preserve detached strokes.
    margin = max(12, int(0.20 * max(w, h)))

    x1 = max(0, x - margin)
    y1 = max(0, y - margin)
    x2 = min(joined.shape[1], x + w + margin)
    y2 = min(joined.shape[0], y + h + margin)

    crop = mask[y1:y2, x1:x2]

    # Recompute the ink bounds in the crop.
    points = cv2.findNonZero(crop)

    if points is None:
        raise ValueError("Could not locate the digit strokes.")

    bx, by, bw, bh = cv2.boundingRect(points)

    # Crop more tightly around the detected ink.
    digit = crop[by:by + bh, bx:bx + bw]

    # Keep the original aspect ratio; fit inside a 20x20 box.
    dh, dw = digit.shape

    if dw == 0 or dh == 0:
        raise ValueError("Detected digit has an invalid size.")

    scale = min(20.0 / dw, 20.0 / dh)

    new_w = max(1, int(round(dw * scale)))
    new_h = max(1, int(round(dh * scale)))

    resized = cv2.resize(
        digit,
        (new_w, new_h),
        interpolation=cv2.INTER_AREA
    )

    # Centre the digit on a 28x28 black canvas.
    canvas = np.zeros((28, 28), dtype=np.uint8)

    offset_x = (28 - new_w) // 2
    offset_y = (28 - new_h) // 2

    canvas[
        offset_y:offset_y + new_h,
        offset_x:offset_x + new_w
    ] = resized

    # Save the exact 28x28 image used by Verilog.
    processed = Image.fromarray(canvas)
    processed.save(PROCESSED_PATH)

    # Convert pixels to normalized, quantized INT8 values.
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
            f"Expected 784 pixels, got {len(values)}."
        )

    # Write each signed INT8 pixel as a hexadecimal byte.
    with open(MEM_PATH, "w") as file:
        for value in values:
            file.write(f"{value & 0xFF:02x}\n")

    print("Processed image:", PROCESSED_PATH)
    print("Input memory file:", MEM_PATH)
    print("Pixel count:", len(values))
    print("Ink pixels:", int(np.count_nonzero(canvas)))

    return processed


# ==================================================
# RUN VERILOG CNN
# ==================================================

def run_cnn():
    path = filedialog.askopenfilename(
        title="Select a handwritten digit image",
        filetypes=[
            ("Image files", "*.png *.jpg *.jpeg *.bmp"),
            ("All files", "*.*")
        ]
    )

    if not path:
        return

    try:
        # Preprocess and generate the Verilog input.
        processed = preprocess_image(path)

        # Compile the complete Verilog CNN.
        compile_result = subprocess.run(
            ["iverilog", "-g2012", "-o", "cnn_sim"] + SOURCES,
            cwd=PROJECT,
            capture_output=True,
            text=True
        )

        if compile_result.returncode != 0:
            raise RuntimeError(
                "Verilog compilation failed:\n"
                + compile_result.stderr
            )

        # Execute the simulation.
        result = subprocess.run(
            ["vvp", "cnn_sim"],
            cwd=PROJECT,
            capture_output=True,
            text=True,
            timeout=120
        )

        if result.returncode != 0:
            raise RuntimeError(
                "Verilog simulation failed:\n"
                + result.stderr
            )

        output = result.stdout

        # Parse predicted digit.
        match = re.search(
            r"Predicted digit:\s*(\d+)",
            output
        )

        if not match:
            raise RuntimeError(
                "Could not find the predicted digit.\n\n"
                + output
            )

        prediction = int(match.group(1))

        # Parse the ten output scores.
        scores = re.findall(
            r"Digit (\d+): (-?\d+)",
            output
        )

        # Load the original image for display.
        original = Image.open(path).convert("RGB")
        original.thumbnail((220, 220))

        original_photo = ImageTk.PhotoImage(original)

        # Enlarge the 28x28 input without smoothing.
        processed_preview = processed.resize(
            (224, 224),
            Image.Resampling.NEAREST
        )

        processed_photo = ImageTk.PhotoImage(
            processed_preview
        )

        # Display the results.
        result_window = tk.Toplevel(root)
        result_window.title("CNN Digit Recognition")
        result_window.geometry("600x560")

        tk.Label(
            result_window,
            text=f"Predicted digit: {prediction}",
            font=("Arial", 22, "bold")
        ).pack(pady=12)

        images_frame = tk.Frame(result_window)
        images_frame.pack(pady=5)

        left = tk.Frame(images_frame)
        left.pack(side="left", padx=15)

        tk.Label(
            left,
            text="Original image"
        ).pack()

        tk.Label(
            left,
            image=original_photo
        ).pack()

        right = tk.Frame(images_frame)
        right.pack(side="left", padx=15)

        tk.Label(
            right,
            text="28 x 28 CNN input"
        ).pack()

        tk.Label(
            right,
            image=processed_photo
        ).pack()

        # Keep image references alive.
        result_window.original_photo = original_photo
        result_window.processed_photo = processed_photo

        tk.Label(
            result_window,
            text="Output scores (digit: score)",
            font=("Arial", 11, "bold")
        ).pack(pady=(12, 4))

        score_lines = [
            f"Digit {digit}: {score}"
            for digit, score in scores
        ]

        tk.Label(
            result_window,
            text="\n".join(score_lines),
            font=("Courier", 10),
            justify="left"
        ).pack()

        # Show the complete simulation output in the terminal.
        print("\n========== VERILOG OUTPUT ==========")
        print(output)

    except Exception as error:
        messagebox.showerror(
            "CNN Demo Error",
            str(error)
        )


# ==================================================
# MAIN APPLICATION
# ==================================================

root = tk.Tk()

root.title("FPGA CNN Accelerator - Digit Recognition")
root.geometry("460x250")

tk.Label(
    root,
    text="Handwritten Digit Recognition",
    font=("Arial", 18, "bold")
).pack(pady=(28, 12))

tk.Label(
    root,
    text=(
        "Select a handwritten digit image.\n"
        "The Verilog CNN will calculate the prediction."
    ),
    justify="center"
).pack(pady=5)

tk.Button(
    root,
    text="Choose Image and Predict",
    command=run_cnn,
    font=("Arial", 12),
    padx=14,
    pady=9
).pack(pady=22)

tk.Label(
    root,
    text="Icarus Verilog simulation",
    font=("Arial", 9)
).pack()

root.mainloop()
