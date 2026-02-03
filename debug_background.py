#!/usr/bin/env python3
"""Background subtraction debugger.

Given a video and frame index, compute the median background,
show raw vs background vs diff, and test multiple thresholds.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import cv2
import numpy as np
from skimage.filters import threshold_otsu

from src.mask_utils import clean_binary_mask


def compute_median_frame(
    video_path: str,
    sample_count: int,
    max_frames: int | None,
) -> np.ndarray:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    if max_frames is not None:
        total_frames = min(total_frames, max_frames)

    if total_frames <= 0:
        cap.release()
        raise ValueError("Video contains no frames")

    num_samples = min(sample_count, total_frames)
    indices = np.linspace(0, total_frames - 1, num_samples).astype(int)
    samples: List[np.ndarray] = []

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if not ret:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        samples.append(gray.astype(np.float32))

    cap.release()

    if not samples:
        raise ValueError("Unable to compute median background")

    median_frame = np.median(np.stack(samples, axis=0), axis=0).astype(np.uint8)
    return median_frame


def read_frame(video_path: str, frame_idx: int) -> np.ndarray:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise ValueError(f"Could not read frame {frame_idx}")
    return frame


def normalize_diff(diff: np.ndarray) -> np.ndarray:
    norm = np.zeros_like(diff, dtype=np.uint8)
    cv2.normalize(diff, dst=norm, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    return norm


def make_visuals(
    frame_bgr: np.ndarray,
    background_gray: np.ndarray,
    diff_gray: np.ndarray,
    thresholds: List[int],
    apply_clean: bool,
) -> List[np.ndarray]:
    visuals: List[np.ndarray] = []

    background_bgr = cv2.cvtColor(background_gray, cv2.COLOR_GRAY2BGR)
    diff_norm = normalize_diff(diff_gray)
    diff_color = cv2.applyColorMap(diff_norm, cv2.COLORMAP_TURBO)

    visuals.append(frame_bgr)
    visuals.append(background_bgr)
    visuals.append(diff_color)

    for thresh in thresholds:
        _, mask = cv2.threshold(diff_gray, thresh, 255, cv2.THRESH_BINARY)
        if apply_clean:
            mask = clean_binary_mask(mask, min_area=5)
        mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        label = f"thr={thresh}"
        cv2.putText(mask_bgr, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        visuals.append(mask_bgr)

    return visuals


def tile_images(images: List[np.ndarray], per_row: int = 3) -> np.ndarray:
    if not images:
        raise ValueError("No images to tile")
    h, w = images[0].shape[:2]
    resized = [cv2.resize(img, (w, h)) for img in images]
    rows = []
    for i in range(0, len(resized), per_row):
        row_imgs = resized[i : i + per_row]
        if len(row_imgs) < per_row:
            pad_count = per_row - len(row_imgs)
            row_imgs.extend([np.zeros_like(resized[0]) for _ in range(pad_count)])
        rows.append(np.hstack(row_imgs))
    montage = np.vstack(rows)
    return montage


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Debug background subtraction")
    parser.add_argument("--video", required=True, help="Path to video file")
    parser.add_argument("--frame", type=int, default=0, help="Frame index to inspect")
    parser.add_argument(
        "--thresholds",
        type=int,
        nargs="+",
        default=[10, 20, 30],
        help="Threshold values to test",
    )
    parser.add_argument(
        "--background-samples",
        type=int,
        default=10,
        help="Number of frames for median background",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional cap on frames considered for median background",
    )
    parser.add_argument(
        "--clean-mask",
        action="store_true",
        help="Apply morphological cleanup to binary masks",
    )
    parser.add_argument(
        "--save-prefix",
        type=str,
        default=None,
        help="If provided, save the montage image with this prefix",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the montage in a window",
    )
    parser.add_argument(
        "--auto-threshold",
        action="store_true",
        help="Compute a per-video Otsu threshold and display it",
    )
    parser.add_argument(
        "--per-frame-otsu",
        action="store_true",
        help="Compute Otsu threshold on the selected frame diff",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    video_path = args.video

    print(f"Computing median background using {args.background_samples} samples...")
    background = compute_median_frame(video_path, args.background_samples, args.max_frames)
    frame_bgr = read_frame(video_path, args.frame)
    gray_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(gray_frame, background)

    print(
        f"Frame stats: mean={gray_frame.mean():.2f}, median={np.median(gray_frame):.2f}, min={gray_frame.min()}, max={gray_frame.max()}"
    )
    print(
        f"Background stats: mean={background.mean():.2f}, median={np.median(background):.2f}, min={background.min()}, max={background.max()}"
    )
    print(
        f"Diff stats: mean={diff.mean():.2f}, median={np.median(diff):.2f}, min={diff.min()}, max={diff.max()}"
    )

    auto_threshold = None
    if args.auto_threshold:
        diff_values = diff.flatten()
        diff_values = diff_values[diff_values > 2]
        if diff_values.size:
            auto_threshold = max(int(round(threshold_otsu(diff_values))), 5)
            print(f"Auto threshold estimate (per video): {auto_threshold}")
        else:
            print("Auto threshold estimate unavailable (diff near zero)")

    if args.per_frame_otsu:
        try:
            frame_thresh = threshold_otsu(diff)
            print(f"Per-frame Otsu threshold: {frame_thresh:.2f}")
        except ValueError:
            print("Per-frame Otsu threshold unavailable")

    visuals = make_visuals(
        frame_bgr,
        background,
        diff,
        thresholds=args.thresholds,
        apply_clean=args.clean_mask,
    )
    montage = tile_images(visuals, per_row=3)

    if args.save_prefix:
        output_path = Path(f"{args.save_prefix}_frame{args.frame}.png")
        cv2.imwrite(str(output_path), montage)
        print(f"Saved montage to {output_path}")

    if args.show or not args.save_prefix:
        cv2.imshow("Background Debug", montage)
        print("Press any key to close window...")
        cv2.waitKey(0)
        cv2.destroyWindow("Background Debug")


if __name__ == "__main__":
    main()
