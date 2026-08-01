#!/usr/bin/env python3
"""Draw the README diagram explaining LeWM rollouts and CEM planning."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


def box(ax, xy, width, height, text, color, *, fontsize=10):
    patch = FancyBboxPatch(xy, width, height, boxstyle="round,pad=0.02", facecolor=color, edgecolor="#334155", linewidth=1.2)
    ax.add_patch(patch)
    ax.text(xy[0] + width / 2, xy[1] + height / 2, text, ha="center", va="center", fontsize=fontsize, wrap=True)


def arrow(ax, start, end, label="", *, color="#334155"):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="->", mutation_scale=14, linewidth=1.3, color=color))
    if label:
        ax.text((start[0] + end[0]) / 2, (start[1] + end[1]) / 2 + 0.025, label, ha="center", va="bottom", fontsize=8, color=color)


def render(output: Path) -> None:
    figure, ax = plt.subplots(figsize=(14, 8), dpi=130)
    ax.set(xlim=(0, 14), ylim=(0, 8))
    ax.axis("off")
    figure.patch.set_facecolor("white")

    ax.text(7, 7.72, "LeWM + CEM : imaginer des futurs, puis choisir des actions", ha="center", fontsize=17, fontweight="bold")
    ax.text(7, 7.37, "Les images décodées rendent les embeddings lisibles ; le prédicteur, lui, réinjecte directement ses embeddings prédits.", ha="center", fontsize=9.5, color="#475569")

    box(ax, (0.45, 5.35), 1.65, 1.05, "Image réelle\nde départ", "#dbeafe")
    box(ax, (2.72, 5.35), 1.8, 1.05, "Vision encoder\nimage → embedding z₀", "#bfdbfe")
    box(ax, (5.18, 5.35), 1.75, 1.05, "Predictor\nP(z₀, actions₁…₅)", "#fde68a")
    box(ax, (7.58, 5.35), 1.7, 1.05, "Embedding prédit\nz₁", "#fef3c7")
    box(ax, (9.92, 5.35), 1.75, 1.05, "Vision decoder\nz₁ → image prédite", "#ddd6fe")
    box(ax, (12.18, 5.35), 1.25, 1.05, "Frame\nprédite", "#ede9fe")
    arrow(ax, (2.1, 5.88), (2.72, 5.88))
    arrow(ax, (4.52, 5.88), (5.18, 5.88), "z₀")
    arrow(ax, (6.93, 5.88), (7.58, 5.88))
    arrow(ax, (9.28, 5.88), (9.92, 5.88))
    arrow(ax, (11.67, 5.88), (12.18, 5.88))
    ax.add_patch(FancyArrowPatch((8.42, 5.35), (6.04, 5.35), connectionstyle="arc3,rad=-0.58", arrowstyle="->", mutation_scale=14, linewidth=1.5, color="#ea580c"))
    ax.text(7.15, 4.52, "z₁ est réinjecté avec le bloc d'actions suivant", ha="center", fontsize=8.7, color="#ea580c")
    ax.text(0.45, 6.72, "1. Rollout autorégressif : z₀ → z₁ → z₂ → …", fontsize=12, fontweight="bold", color="#1e3a8a")

    box(ax, (0.7, 2.15), 1.55, 1.0, "Image de départ\nencodée", "#dbeafe")
    box(ax, (0.7, 0.72), 1.55, 1.0, "Image objectif\nencodée", "#dcfce7")
    box(ax, (3.05, 2.15), 2.2, 1.0, "300 séquences candidates\n5 blocs × 5 actions", "#e0e7ff")
    box(ax, (6.05, 2.15), 2.0, 1.0, "LeWM prédit\n300 futurs z₂₅", "#fde68a")
    box(ax, (8.9, 2.15), 2.0, 1.0, "coût = distance\n‖z₂₅ − zobjectif‖", "#fecaca")
    box(ax, (11.65, 2.15), 1.65, 1.0, "30 élites\n(coût le plus bas)", "#fed7aa")
    arrow(ax, (2.25, 2.65), (3.05, 2.65))
    arrow(ax, (5.25, 2.65), (6.05, 2.65))
    arrow(ax, (8.05, 2.65), (8.9, 2.65))
    arrow(ax, (10.9, 2.65), (11.65, 2.65))
    arrow(ax, (2.25, 1.22), (8.9, 2.15), "zobjectif", color="#16a34a")
    ax.add_patch(FancyArrowPatch((12.45, 2.15), (4.15, 2.15), connectionstyle="arc3,rad=-0.28", arrowstyle="->", mutation_scale=14, linewidth=1.5, color="#ea580c"))
    ax.text(8.35, 1.36, "met à jour la distribution, puis on recommence", ha="center", fontsize=8.7, color="#ea580c")
    ax.text(0.45, 3.55, "2. Planning CEM : 300 candidats → 30 élites → distribution plus concentrée", fontsize=12, fontweight="bold", color="#9a3412")
    ax.text(7, 0.18, "Les trajectoires colorées du README sont des futurs prédits/décodés. Seul le plan final est envoyé à PushT et produit une trajectoire réelle.", ha="center", fontsize=9.5, color="#475569")
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    render(parser.parse_args().output)


if __name__ == "__main__":
    main()
