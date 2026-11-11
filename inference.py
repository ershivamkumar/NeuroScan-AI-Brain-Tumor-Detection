#!/usr/bin/env python3
"""NeuroScan AI - Command Line Inference Tool.

Run single-image or batch brain MRI classification using trained Keras models.
Supports MobileNetV2 transfer learning and CNN-LSTM models.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageOps

CLASS_NAMES = {0: "Brain Tumor", 1: "Healthy / No Tumor"}
IMG_SIZE = 150


def load_inference_model(model_path: str | Path):
    """Load the Keras model from disk."""
    import tensorflow as tf

    path = Path(model_path)
    if not path.is_file():
        raise FileNotFoundError(f"Model file not found at: {path.resolve()}")

    print(f"[NeuroScan AI] Loading model: {path.name}")
    return tf.keras.models.load_model(path)


def preprocess_image(image_path: str | Path) -> np.ndarray:
    """Preprocess MRI image: grayscale -> 150x150 -> 3-channel duplicate -> MobileNetV2 preprocess_input."""
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

    path = Path(image_path)
    if not path.is_file():
        raise FileNotFoundError(f"Image not found at: {path.resolve()}")

    with Image.open(path) as raw:
        img = ImageOps.exif_transpose(raw)
        gray = ImageOps.grayscale(img).resize((IMG_SIZE, IMG_SIZE), Image.Resampling.LANCZOS)
        arr = np.asarray(gray, dtype=np.float32)
        rgb = np.repeat(arr[..., np.newaxis], 3, axis=-1)
        return np.expand_dims(preprocess_input(rgb), axis=0)


def predict_single(model, image_path: str | Path) -> dict[str, Any]:
    """Perform prediction on a single MRI image."""
    x = preprocess_image(image_path)
    out = model(x, training=False)
    probs = np.asarray(out.numpy() if hasattr(out, "numpy") else out, dtype=float)[0]

    cls = int(np.argmax(probs))
    return {
        "file": str(Path(image_path).name),
        "predicted_class": cls,
        "label": CLASS_NAMES[cls],
        "confidence": float(probs[cls] * 100),
        "tumor_score": float(probs[0] * 100),
        "healthy_score": float(probs[1] * 100),
    }


def run_batch(model, folder_path: str | Path, output_csv: str | None = None) -> list[dict[str, Any]]:
    """Perform batch prediction on all images in a folder."""
    import pandas as pd

    folder = Path(folder_path)
    valid_exts = {".jpg", ".jpeg", ".png"}
    images = sorted([f for f in folder.iterdir() if f.is_file() and f.suffix.lower() in valid_exts])

    if not images:
        print(f"[NeuroScan AI] No valid images found in folder: {folder}")
        return []

    print(f"[NeuroScan AI] Running batch inference on {len(images)} images...")
    results = []
    for img_path in images:
        try:
            res = predict_single(model, img_path)
            results.append(res)
        except Exception as e:
            print(f"  [ERROR] Failed to process {img_path.name}: {e}")

    if output_csv:
        df = pd.DataFrame(results)
        df.to_csv(output_csv, index=False)
        print(f"[NeuroScan AI] Results saved to: {output_csv}")

    return results


def main() -> int:
    parser = argparse.ArgumentParser(
        description="NeuroScan AI: Brain MRI Tumor Detection CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--model",
        type=str,
        default="outputs/mobilenetv2_best.keras",
        help="Path to trained .keras model (default: outputs/mobilenetv2_best.keras)",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--image", type=str, help="Path to single MRI scan image")
    group.add_argument("--folder", type=str, help="Path to directory of MRI images for batch evaluation")
    parser.add_argument("--output", type=str, default=None, help="Path to output CSV file for batch results")

    args = parser.parse_args()

    # Fallback search for model file if default relative path doesn't exist
    model_path = Path(args.model)
    if not model_path.exists():
        script_dir = Path(__file__).resolve().parent
        alt_paths = [script_dir / "outputs" / "mobilenetv2_best.keras", script_dir / "mobilenetv2_best.keras"]
        for alt in alt_paths:
            if alt.exists():
                model_path = alt
                break

    try:
        model = load_inference_model(model_path)
    except Exception as exc:
        print(f"[ERROR] Could not load model: {exc}", file=sys.stderr)
        return 1

    if args.image:
        try:
            res = predict_single(model, args.image)
            print("\n" + "=" * 45)
            print("         NEUROSCAN AI INFERENCE REPORT        ")
            print("=" * 45)
            print(f" File:             {res['file']}")
            print(f" Predicted Class:  {res['label']} (Class {res['predicted_class']})")
            print(f" Confidence:       {res['confidence']:.2f}%")
            print("-" * 45)
            print(f" Tumor Score:      {res['tumor_score']:.2f}%")
            print(f" Healthy Score:    {res['healthy_score']:.2f}%")
            print("=" * 45)
            print("[Notice] Research prototype. Not a clinical diagnosis.\n")
            return 0
        except Exception as exc:
            print(f"[ERROR] Inference failed on {args.image}: {exc}", file=sys.stderr)
            return 1

    if args.folder:
        try:
            results = run_batch(model, args.folder, output_csv=args.output)
            print(f"\n[NeuroScan AI] Completed inference on {len(results)} images.")
            return 0
        except Exception as exc:
            print(f"[ERROR] Batch inference failed: {exc}", file=sys.stderr)
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
