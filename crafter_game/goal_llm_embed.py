"""
crafter_game/goal_llm_embed.py — Compute and cache Qwen3 text embeddings
for Crafter's 22 achievement goals, ONCE. Run this on a machine with
internet access (your local PC, or a cluster's frontend node) — training
jobs then just load the cached .npy file, no internet needed at that
point (same pattern as installing pip packages once on Adastra/Jean
Zay's frontend before a job runs on a compute node with no internet).

These embeddings are meant to be used FROZEN (not fine-tuned further
during RL training) as the goal representation — see architecture.py's
GoalRepr(frozen_embeddings_path=...).

Usage:
    python -m crafter_game.goal_llm_embed --out crafter_goal_embeddings.npy
"""

import argparse
import numpy as np

from crafter_game.env import ACHIEVEMENTS


def achievement_to_text(name: str) -> str:
    """snake_case achievement id -> natural-language phrase, e.g.
    'make_iron_pickaxe' -> 'make iron pickaxe'. Deliberately simple/rule-
    based rather than 22 hand-written descriptions — good enough for a
    semantic embedding model, and easy to edit later if specific phrasing
    turns out to matter."""
    return name.replace("_", " ")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-Embedding-0.6B",
                   help="Any sentence-transformers-compatible model id. "
                        "0.6B is the smallest Qwen3 embedding size — "
                        "plenty for embedding 22 short phrases.")
    p.add_argument("--out", default="crafter_goal_embeddings.npy")
    args = p.parse_args()

    from sentence_transformers import SentenceTransformer
    print(f"Loading {args.model} (downloads on first use — needs internet)...")
    model = SentenceTransformer(args.model)

    texts = [achievement_to_text(a) for a in ACHIEVEMENTS]
    print("Embedding goals:")
    for a, t in zip(ACHIEVEMENTS, texts):
        print(f"  {a:<20} -> \"{t}\"")

    embeddings = model.encode(texts, normalize_embeddings=True)
    print(f"Embeddings shape: {embeddings.shape}")

    np.save(args.out, embeddings.astype(np.float32))
    print(f"Saved to {args.out}")
    print("Row i = embedding for crafter_game.env.ACHIEVEMENTS[i] — order "
         "matters, don't reorder ACHIEVEMENTS without recomputing this file.")


if __name__ == "__main__":
    main()