"""
Run once before training to pre-crop faces from the dataset.
Saves cropped faces to a mirrored folder structure under a new root.

Usage:
    python3 preprocess_faces.py --src ./data/TuftsFaceDatabase --dst ./data/TuftsFaceCropped
"""

import os
import argparse
import cv2
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from utils import detect_and_crop_face


def process_image(args):
    src_path, dst_path = args
    if os.path.exists(dst_path):
        return
    img_bgr = cv2.imread(src_path)
    if img_bgr is None:
        return
    cropped, _ = detect_and_crop_face(img_bgr, padding=0.15)
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    cv2.imwrite(dst_path, cropped)


def preprocess(src_root: str, dst_root: str):
    supported = ('.jpg', '.jpeg', '.png', '.bmp')
    pairs = []

    for dirpath, _, filenames in os.walk(src_root):
        for fname in filenames:
            if fname.lower().endswith(supported):
                src_path = os.path.join(dirpath, fname)
                rel_path = os.path.relpath(src_path, src_root)
                dst_path = os.path.join(dst_root, rel_path)
                pairs.append((src_path, dst_path))

    workers = cpu_count()
    total = len(pairs)
    print(f"Found {total} images. Cropping with {workers} workers...")

    completed = 0
    with Pool(workers) as pool:
        for _ in tqdm(pool.imap(process_image, pairs), total=total, unit="img"):
            completed += 1
            if completed % 500 == 0:
                print(f"  Progress: {completed}/{total} images done ({completed*100//total}%)")

    print(f"Done. All {total} images processed. Cropped images saved to: {dst_root}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", default="./data/TuftsFaceDatabase")
    parser.add_argument("--dst", default="./data/TuftsFaceCropped")
    args = parser.parse_args()
    preprocess(args.src, args.dst)
