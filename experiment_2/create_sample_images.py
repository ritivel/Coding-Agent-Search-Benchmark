#!/usr/bin/env python3
"""Generate synthetic handwritten-style test images for pipeline testing."""

from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
import random
import math


def create_handwriting_image(
    text_lines: list,
    output_path: str,
    width: int = 1200,
    line_height: int = 45,
    margin: int = 60,
    bg_color: int = 245,
    ink_color: int = 25,
):
    height = margin * 2 + line_height * len(text_lines) + 40
    img = Image.new("L", (width, height), bg_color)
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype("/System/Library/Fonts/Noteworthy.ttc", 28)
    except OSError:
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 24)
        except OSError:
            font = ImageFont.load_default()

    random.seed(42)
    for i, line in enumerate(text_lines):
        x = margin + random.randint(-3, 3)
        y = margin + i * line_height + random.randint(-2, 2)
        color = ink_color + random.randint(-10, 15)
        draw.text((x, y), line, fill=color, font=font)

    for _ in range(5):
        x = random.randint(0, width)
        y = random.randint(0, height)
        r = random.randint(1, 3)
        shade = bg_color - random.randint(5, 20)
        draw.ellipse([x - r, y - r, x + r, y + r], fill=shade)

    angle = random.uniform(-1.5, 1.5)
    img = img.rotate(angle, fillcolor=bg_color, expand=False)

    img.save(output_path)
    print(f"Created: {output_path}")


def create_all_samples():
    output_dir = Path(__file__).parent / "sample_images"
    output_dir.mkdir(exist_ok=True)

    letter_lines = [
        "Dear Professor Johnson,",
        "",
        "I am writing to express my gratitude for the",
        "wonderful lecture on neural networks yesterday.",
        "The explanation of backpropagation was particularly",
        "clear and helped me understand gradient descent",
        "in a way that textbooks have not managed to convey.",
        "",
        "I have a few questions about the assignment:",
        "1. Should we implement batch normalization?",
        "2. What learning rate do you recommend?",
        "3. Is dropout required for the final model?",
        "",
        "Thank you for your time and guidance.",
        "Best regards,",
        "Alex Thompson",
    ]
    create_handwriting_image(letter_lines, str(output_dir / "sample_letter.png"))

    notes_lines = [
        "Meeting Notes - March 28, 2026",
        "",
        "Attendees: Sarah, James, Maria, David",
        "",
        "Agenda Items:",
        "- Q1 performance review results",
        "- New product launch timeline",
        "- Budget allocation for Q2",
        "- Team restructuring proposal",
        "",
        "Key Decisions:",
        "- Launch date moved to April 15th",
        "- Marketing budget increased by 20%",
        "- Hire two additional engineers",
        "- Weekly standup moved to Tuesdays",
        "",
        "Action Items:",
        "- Sarah: finalize marketing plan by April 1",
        "- James: prepare technical documentation",
        "- Maria: schedule client meetings",
        "- David: update project timeline in Jira",
    ]
    create_handwriting_image(notes_lines, str(output_dir / "sample_meeting_notes.png"))

    science_lines = [
        "Introduction to Deep Learning",
        "",
        "Neural networks are computational models inspired",
        "by the biological neural networks in human brains.",
        "They consist of layers of interconnected nodes that",
        "process information using mathematical functions.",
        "",
        "Key Concepts:",
        "- Neurons: basic computational units",
        "- Weights: learnable parameters",
        "- Activation functions: ReLU, sigmoid, tanh",
        "- Loss functions: MSE, cross-entropy",
        "- Backpropagation: gradient-based optimization",
        "",
        "Applications include computer vision, natural",
        "language processing, speech recognition, and",
        "autonomous driving systems.",
    ]
    create_handwriting_image(science_lines, str(output_dir / "sample_science_notes.png"))


if __name__ == "__main__":
    create_all_samples()
