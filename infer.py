#!/usr/bin/env python3
"""Sample from the released PCA-LM checkpoint.

    python infer.py --prompt "The key idea behind" --tokens 120
"""
import argparse
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from pca_lm import load_pretrained


@torch.no_grad()
def generate(model, cfg, tok, prompt, n_tokens, temperature, top_p, device):
    ids = tok(prompt, return_tensors="pt").input_ids.to(device)
    for _ in range(n_tokens):
        # The recurrence is not cached across calls, so the window is re-read
        # each step; keep it inside the trained context.
        window = ids[:, -cfg.max_seq_len:]
        out = model(window)
        logits = (out[0] if isinstance(out, (tuple, list)) else out)[:, -1]
        logits = logits.float() / max(temperature, 1e-5)
        probs = F.softmax(logits, dim=-1)
        if 0 < top_p < 1:
            sp, si = probs.sort(descending=True)
            keep = (sp.cumsum(-1) - sp) < top_p
            sp = sp * keep
            sp = sp / sp.sum(-1, keepdim=True)
            nxt = si.gather(-1, torch.multinomial(sp, 1))
        else:
            nxt = torch.multinomial(probs, 1)
        ids = torch.cat([ids, nxt], dim=1)
    return tok.decode(ids[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="weights",
                    help="a .pt, a .manifest.json, or the weights/ directory")
    ap.add_argument("--prompt", default="The key idea behind")
    ap.add_argument("--tokens", type=int, default=120)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()

    model, cfg = load_pretrained(a.ckpt, a.device)
    tok = AutoTokenizer.from_pretrained("gpt2")
    print(generate(model, cfg, tok, a.prompt, a.tokens,
                   a.temperature, a.top_p, a.device))


if __name__ == "__main__":
    main()
