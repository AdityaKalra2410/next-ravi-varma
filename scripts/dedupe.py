"""Flag near-duplicate images (same painting, different scans) using perceptual hashing.

Usage: python scripts/dedupe.py --dir data/raw --threshold 8
Prints groups of likely duplicates; keep the best scan, mark the rest keep=n in metadata.csv.
"""
import argparse, os
from PIL import Image
import imagehash

ap = argparse.ArgumentParser()
ap.add_argument("--dir", default="data/raw")
ap.add_argument("--threshold", type=int, default=8, help="max hamming distance to count as duplicate")
args = ap.parse_args()

hashes = []
for f in sorted(os.listdir(args.dir)):
    if f.lower().endswith((".jpg", ".jpeg", ".png")):
        hashes.append((f, imagehash.phash(Image.open(os.path.join(args.dir, f)).convert("RGB"))))

used = set()
for i, (f1, h1) in enumerate(hashes):
    if f1 in used:
        continue
    group = [f2 for f2, h2 in hashes[i + 1:] if f2 not in used and h1 - h2 <= args.threshold]
    if group:
        used.update(group)
        print("DUPLICATES:", f1, *group)
