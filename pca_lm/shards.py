#!/usr/bin/env python3
"""Byte-level split/join for checkpoints too large to upload whole.

GitHub's web uploader rejects any single file over 25 MB, and this
checkpoint's token embedding is one 103 MB tensor -- so splitting per tensor
cannot get under the limit. The file is therefore split at the BYTE level and
reassembled before loading, which is size-agnostic.

Used by load_pretrained(); also runnable via ../split_weights.py.
"""

import argparse
import hashlib
import json
import os

PART_MB = 20            # under GitHub's 25 MB web-upload cap, with margin
CHUNK = 1 << 20


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(CHUNK), b""):
            h.update(b)
    return h.hexdigest()


def split(src, part_mb=PART_MB):
    size = part_mb * 1024 * 1024
    base = os.path.splitext(src)[0]
    parts, idx = [], 0
    with open(src, "rb") as f:
        while True:
            buf = f.read(size)
            if not buf:
                break
            name = f"{os.path.basename(base)}.part{idx:03d}"
            p = os.path.join(os.path.dirname(src) or ".", name)
            with open(p, "wb") as out:
                out.write(buf)
            parts.append({"name": name, "bytes": len(buf), "sha256": _sha256(p)})
            idx += 1
    man = {"file": os.path.basename(src),
           "total_bytes": os.path.getsize(src),
           "sha256": _sha256(src),
           "part_bytes": size,
           "parts": parts}
    mpath = base + ".manifest.json"
    with open(mpath, "w") as f:
        json.dump(man, f, indent=2)
    print(f"  {len(parts)} parts, {man['total_bytes']/1e6:.0f} MB total "
          f"-> {os.path.basename(mpath)}")
    return mpath


def join(manifest, out=None, verify=True):
    """Reassemble; returns the path to the joined file."""
    d = os.path.dirname(os.path.abspath(manifest))
    man = json.load(open(manifest))
    out = out or os.path.join(d, man["file"])
    if os.path.exists(out) and verify and _sha256(out) == man["sha256"]:
        return out                                  # already joined
    with open(out, "wb") as w:
        for part in man["parts"]:
            p = os.path.join(d, part["name"])
            if not os.path.exists(p):
                raise FileNotFoundError(
                    f"missing {part['name']} -- all {len(man['parts'])} parts "
                    "must be present")
            with open(p, "rb") as r:
                while True:
                    b = r.read(CHUNK)
                    if not b:
                        break
                    w.write(b)
    if verify:
        got = _sha256(out)
        if got != man["sha256"]:
            raise ValueError(f"checksum mismatch: {got} != {man['sha256']}")
    return out


