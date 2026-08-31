#!/usr/bin/env python3
"""Split a checkpoint into GUI-uploadable parts, or rejoin them.

GitHub's web uploader rejects any single file over 25 MB, and this model's
token embedding alone is one 103 MB tensor -- so splitting per tensor cannot
get under the limit. The file is split at the BYTE level instead, which is
size-agnostic, and rejoined (with a checksum) before loading.

    python split_weights.py split weights/pca_lm_63m.pt
    python split_weights.py join  weights/pca_lm_63m.manifest.json

You do not normally need : infer.py and eval_val.py call it for you the
first time they load, then cache the joined file.
"""
import argparse

from pca_lm.shards import split, join


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["split", "join"])
    ap.add_argument("path")
    ap.add_argument("--part-mb", type=int, default=20)
    a = ap.parse_args()
    if a.action == "split":
        split(a.path, a.part_mb)
    else:
        print("  joined ->", join(a.path))


if __name__ == "__main__":
    main()
