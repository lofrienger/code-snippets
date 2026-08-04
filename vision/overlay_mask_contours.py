#!/usr/bin/env python3
"""Overlay optic-disc and optic-cup mask contours on a fundus image."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

Color = tuple[int, int, int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path, help="source image")
    parser.add_argument("ground_truth_mask", type=Path, help="ground-truth label mask")
    parser.add_argument("output", type=Path, help="output image")
    parser.add_argument(
        "--prediction-mask",
        type=Path,
        help="optional prediction mask to draw with separate colors",
    )
    parser.add_argument(
        "--size",
        type=int,
        nargs=2,
        metavar=("WIDTH", "HEIGHT"),
        help="resize the image and masks before drawing",
    )
    parser.add_argument("--background-label", type=int, default=0)
    parser.add_argument("--cup-label", type=int, default=255)
    parser.add_argument("--thickness", type=int, default=2)
    return parser.parse_args()


def read_image(path: Path, mode: int) -> np.ndarray:
    image = cv2.imread(str(path), mode)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return image


def resize_image(image: np.ndarray, size: tuple[int, int], mask: bool) -> np.ndarray:
    interpolation = cv2.INTER_NEAREST if mask else cv2.INTER_AREA
    return cv2.resize(image, size, interpolation=interpolation)


def contour_count_and_draw(
    image: np.ndarray,
    binary_mask: np.ndarray,
    color: Color,
    thickness: int,
) -> int:
    contours, _ = cv2.findContours(
        binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if contours:
        cv2.drawContours(image, contours, -1, color, thickness)
    return len(contours)


def overlay_label_mask(
    image: np.ndarray,
    mask: np.ndarray,
    background_label: int,
    cup_label: int,
    disc_color: Color,
    cup_color: Color,
    thickness: int,
) -> tuple[int, int]:
    disc = np.where(mask != background_label, 255, 0).astype(np.uint8)
    cup = np.where(mask == cup_label, 255, 0).astype(np.uint8)
    disc_count = contour_count_and_draw(image, disc, disc_color, thickness)
    cup_count = contour_count_and_draw(image, cup, cup_color, thickness)
    return disc_count, cup_count


def main() -> int:
    args = parse_args()
    if args.thickness < 1:
        raise ValueError("--thickness must be at least 1")
    if args.size and any(value < 1 for value in args.size):
        raise ValueError("--size values must be positive")
    if not 0 <= args.background_label <= 255 or not 0 <= args.cup_label <= 255:
        raise ValueError("mask labels must be between 0 and 255")

    image = read_image(args.image, cv2.IMREAD_COLOR)
    ground_truth = read_image(args.ground_truth_mask, cv2.IMREAD_GRAYSCALE)
    prediction = None
    if args.prediction_mask:
        prediction = read_image(args.prediction_mask, cv2.IMREAD_GRAYSCALE)

    if args.size:
        size = (args.size[0], args.size[1])
        image = resize_image(image, size, mask=False)
    else:
        size = (image.shape[1], image.shape[0])
    ground_truth = resize_image(ground_truth, size, mask=True)
    if prediction is not None:
        prediction = resize_image(prediction, size, mask=True)

    gt_counts = overlay_label_mask(
        image,
        ground_truth,
        args.background_label,
        args.cup_label,
        disc_color=(0, 0, 255),
        cup_color=(0, 255, 255),
        thickness=args.thickness,
    )
    prediction_counts = None
    if prediction is not None:
        prediction_counts = overlay_label_mask(
            image,
            prediction,
            args.background_label,
            args.cup_label,
            disc_color=(255, 0, 0),
            cup_color=(0, 255, 0),
            thickness=args.thickness,
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(args.output), image):
        raise OSError(f"Could not write output image: {args.output}")

    print(f"Saved overlay: {args.output}")
    print(f"Ground truth contours: disc={gt_counts[0]}, cup={gt_counts[1]}")
    if prediction_counts is not None:
        print(
            "Prediction contours: "
            f"disc={prediction_counts[0]}, cup={prediction_counts[1]}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
