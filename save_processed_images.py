
import os
import cv2
import numpy as np
from PIL import Image, ImageDraw

PROJECT = os.path.expanduser("~/fpga-cnn")
INPUT_DIR = os.path.join(PROJECT, "test_images")
OUTPUT_DIR = os.path.join(PROJECT, "processed_images")
os.makedirs(OUTPUT_DIR, exist_ok=True)

EXTENSIONS = (".jpg", ".jpeg", ".png")


def preprocess_image(path):
    image = cv2.imread(path)

    if image is None:
        raise ValueError(f"Cannot read image: {path}")

    # Detect blue ink in HSV colour space.
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # OpenCV hue range is 0-179.
    # These limits target blue pen ink and reject much of the paper texture.
    lower_blue = np.array([90, 35, 20], dtype=np.uint8)
    upper_blue = np.array([145, 255, 255], dtype=np.uint8)

    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    # Remove tiny isolated specks without aggressively eroding strokes.
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask, connectivity=8
    )

    if count <= 1:
        raise ValueError(
            "No blue ink detected. The colour thresholds may need adjustment."
        )

    # Select the largest substantial connected component.
    # If the digit is broken into several components, this may need adjustment.
    areas = stats[1:, cv2.CC_STAT_AREA]
    largest = 1 + int(np.argmax(areas))

    digit_mask = np.zeros_like(mask)
    digit_mask[labels == largest] = 255

    # Find the digit bounds.
    points = cv2.findNonZero(digit_mask)

    if points is None:
        raise ValueError("Could not locate the digit.")

    x, y, w, h = cv2.boundingRect(points)

    # Add a small margin around the detected strokes.
    margin = max(5, int(0.08 * max(w, h)))
    x1 = max(0, x - margin)
    y1 = max(0, y - margin)
    x2 = min(digit_mask.shape[1], x + w + margin)
    y2 = min(digit_mask.shape[0], y + h + margin)

    digit = digit_mask[y1:y2, x1:x2]

    # Fit the digit inside a 20x20 region, preserving proportions.
    dh, dw = digit.shape
    scale = min(20.0 / dw, 20.0 / dh)

    new_w = max(1, round(dw * scale))
    new_h = max(1, round(dh * scale))

    digit = cv2.resize(
        digit,
        (new_w, new_h),
        interpolation=cv2.INTER_AREA
    )

    # MNIST-style polarity: black background, bright digit.
    canvas = np.zeros((28, 28), dtype=np.uint8)

    x0 = (28 - new_w) // 2
    y0 = (28 - new_h) // 2
    canvas[y0:y0 + new_h, x0:x0 + new_w] = digit

    return canvas


def main():
    if not os.path.isdir(INPUT_DIR):
        print("Input folder not found:", INPUT_DIR)
        return

    files = sorted(
        f for f in os.listdir(INPUT_DIR)
        if f.lower().endswith(EXTENSIONS)
    )

    if not files:
        print("No images found in:", INPUT_DIR)
        return

    processed = []

    for filename in files:
        source = os.path.join(INPUT_DIR, filename)
        output_name = os.path.splitext(filename)[0] + ".png"
        output_path = os.path.join(OUTPUT_DIR, output_name)

        try:
            result = preprocess_image(source)
            cv2.imwrite(output_path, result)
            processed.append(output_name)
            print("Processed:", filename, "->", output_name)

        except Exception as error:
            print("FAILED:", filename, "-", error)

    # Create an enlarged comparison sheet.
    sheet = Image.new(
        "RGB",
        (420, max(1, len(processed)) * 100),
        "white"
    )
    draw = ImageDraw.Draw(sheet)

    for index, filename in enumerate(processed):
        path = os.path.join(OUTPUT_DIR, filename)
        img = Image.open(path).convert("L")
        enlarged = img.resize(
            (84, 84),
            Image.Resampling.NEAREST
        )

        sheet.paste(enlarged.convert("RGB"), (5, index * 100 + 5))
        draw.text(
            (100, index * 100 + 35),
            filename,
            fill="black"
        )

    comparison = os.path.join(PROJECT, "processed_comparison.png")
    sheet.save(comparison)

    print("\nPreprocessing finished.")
    print("Successfully processed:", len(processed), "of", len(files))
    print("Output folder:", OUTPUT_DIR)
    print("Comparison sheet:", comparison)


if __name__ == "__main__":
    main()
