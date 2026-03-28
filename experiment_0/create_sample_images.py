"""
Generate synthetic handwritten-style sample images for testing the pipeline.

Uses PIL to render text with a handwriting-style layout, adding slight
noise and rotation to simulate real scanned documents.
"""

from pathlib import Path
import random
import numpy as np
from PIL import Image, ImageDraw, ImageFont

SAMPLE_TEXTS = [
    {
        "filename": "sample_letter.png",
        "lines": [
            "Dear Professor Smith,",
            "",
            "I am writing to request an extension",
            "on the research paper that is due",
            "next Friday. I have been working",
            "diligently on the topic of machine",
            "learning applications in healthcare,",
            "but I need additional time to complete",
            "the literature review section.",
            "",
            "I would greatly appreciate your",
            "consideration of this request.",
            "",
            "Sincerely,",
            "Jane Doe",
        ],
    },
    {
        "filename": "sample_notes.png",
        "lines": [
            "Chapter 3: Neural Networks",
            "",
            "Key Concepts:",
            "- Perceptron model",
            "- Activation functions (ReLU, sigmoid)",
            "- Backpropagation algorithm",
            "- Gradient descent optimization",
            "",
            "The basic unit of a neural network",
            "is the neuron, which computes a",
            "weighted sum of its inputs and",
            "applies a nonlinear activation",
            "function to produce an output.",
            "",
            "Loss = 1/N sum(y - y_hat)^2",
        ],
    },
    {
        "filename": "sample_recipe.png",
        "lines": [
            "Grandma's Chocolate Cake Recipe",
            "",
            "Ingredients:",
            "- 2 cups all-purpose flour",
            "- 1.5 cups sugar",
            "- 3/4 cup cocoa powder",
            "- 2 eggs",
            "- 1 cup buttermilk",
            "- 1/2 cup vegetable oil",
            "- 2 tsp vanilla extract",
            "",
            "Instructions:",
            "1. Preheat oven to 350 degrees F",
            "2. Mix dry ingredients together",
            "3. Add wet ingredients and stir",
            "4. Pour into greased 9x13 pan",
            "5. Bake for 30-35 minutes",
        ],
    },
]


def create_handwritten_image(
    lines: list[str],
    width: int = 800,
    line_height: int = 40,
    margin: int = 60,
    font_size: int = 24,
) -> Image.Image:
    height = margin * 2 + len(lines) * line_height + 40
    img = Image.new("RGB", (width, height), color=(255, 252, 240))
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype("/System/Library/Fonts/Noteworthy.ttc", font_size)
    except (OSError, IOError):
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size)
        except (OSError, IOError):
            font = ImageFont.load_default()

    y = margin
    for line in lines:
        x_offset = random.randint(-3, 3)
        y_offset = random.randint(-2, 2)
        color_var = random.randint(0, 40)
        ink = (20 + color_var, 20 + color_var, 80 + color_var)
        draw.text((margin + x_offset, y + y_offset), line, fill=ink, font=font)
        y += line_height

    arr = np.array(img)
    noise = np.random.normal(0, 3, arr.shape).astype(np.int16)
    arr = np.clip(arr.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    img = Image.fromarray(arr)

    angle = random.uniform(-1.5, 1.5)
    img = img.rotate(angle, fillcolor=(255, 252, 240), expand=False)

    return img


def main():
    output_dir = Path(__file__).resolve().parent / "sample_images"
    output_dir.mkdir(parents=True, exist_ok=True)

    for sample in SAMPLE_TEXTS:
        img = create_handwritten_image(sample["lines"])
        path = output_dir / sample["filename"]
        img.save(path, "PNG")
        print(f"Created {path}  ({img.width}x{img.height})")

    print(f"\nGenerated {len(SAMPLE_TEXTS)} sample images in {output_dir}")


if __name__ == "__main__":
    main()
