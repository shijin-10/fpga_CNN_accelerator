
from PIL import Image, ImageOps, ImageDraw
import os

folder = "test_images"
files = sorted(
    f for f in os.listdir(folder)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
)

thumbs = []

for name in files:
    img = Image.open(os.path.join(folder, name)).convert("L")

    # Show the original photograph resized to fit.
    original = ImageOps.contain(img, (180, 220))

    # Current preprocessing: resize the entire photograph.
    direct = img.resize((28, 28), Image.Resampling.BILINEAR)
    direct = direct.resize((180, 180), Image.Resampling.NEAREST)

    panel = Image.new("RGB", (390, 270), "white")
    panel.paste(original.convert("RGB"), (10, 35))
    panel.paste(direct.convert("RGB"), (200, 35))

    draw = ImageDraw.Draw(panel)
    draw.text((10, 10), f"{name} - Original", fill="black")
    draw.text((200, 10), "Direct 28x28 resize", fill="black")
    thumbs.append(panel)

columns = 2
rows = (len(thumbs) + columns - 1) // columns
sheet = Image.new("RGB", (columns * 390, rows * 270), "#cccccc")

for i, panel in enumerate(thumbs):
    sheet.paste(panel, ((i % columns) * 390, (i // columns) * 270))

sheet.save("image_inspection.png")
print("Saved image_inspection.png")
