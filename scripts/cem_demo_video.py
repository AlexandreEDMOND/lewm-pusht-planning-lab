"""Deterministic rendering of the end-to-end CEM demo animations.

The GIF frame convention is documented and differs from the raw recording
convention ``observation initiale + observation après chaque action``:

- for each of the two decisions (action offsets 0 and 25), three "CEM search"
  frames show the recorded search at iterations 10, 20 and 30 of the 30;
- then the recorded real observations are shown, one per executed action,
  from the decision offset up to the next replanning offset (25 frames for
  decision 0, 26 for decision 1, i.e. observations t=0..24 then t=25..50).

The full sequence has 3 + 25 + 3 + 26 = 57 frames.  Every number displayed in
the panels comes from the compact traces or from the raw execution arrays.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import imageio.v2 as imageio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrow, Polygon as MplPolygon

from cem_demo import ACTION_BLOCK, BUDGET_ACTIONS, HORIZON, ITERATIONS


WORLD_SIZE = 512.0
FRAME_SIZE = 224
SCALE = FRAME_SIZE / WORLD_SIZE
AGENT_RADIUS_WORLD = 15.0

# PushT "T" block geometry (stable-worldmodel 0.0.6, add_tee), body frame.
T_BLOCK_VERTICES = np.array(
    [
        [-60.0, 30.0], [60.0, 30.0], [60.0, 0.0], [-60.0, 0.0],
        [-15.0, 30.0], [-15.0, 120.0], [15.0, 120.0], [15.0, 30.0],
    ],
    dtype=np.float64,
)

SEARCH_ITERATIONS = (10, 20, 30)

COLOR_REAL = "#111827"
COLOR_PREDICTED = "#f97316"
COLOR_GOAL = "#16a34a"
COLOR_AGENT = "#1d4ed8"
COLOR_DECISION_0 = "#2563eb"
COLOR_DECISION_1 = "#ea580c"
COLOR_SIGMA = "#9333ea"
COLOR_ELITE = "#f59e0b"


@dataclass
class DecisionRender:
    """All numbers needed to draw one recorded CEM decision."""

    offset: int
    mean_after: np.ndarray = field(repr=False)
    std_after: np.ndarray = field(repr=False)
    cost_mean: np.ndarray = field(repr=False)
    cost_median: np.ndarray = field(repr=False)
    cost_min: np.ndarray = field(repr=False)
    sigma_mean: np.ndarray = field(repr=False)
    plan_actions: np.ndarray = field(repr=False)
    plan_poses: np.ndarray = field(repr=False)
    block_errors_px: np.ndarray = field(repr=False)
    block_latent_mse: np.ndarray = field(repr=False)
    block_angle_errors_deg: np.ndarray = field(repr=False)


@dataclass
class EpisodeRender:
    """One episode, fully prepared for rendering."""

    episode: int
    start_step: int
    success: bool
    observations: np.ndarray = field(repr=False)
    states: np.ndarray = field(repr=False)
    actions_postprocessed: np.ndarray = field(repr=False)
    goal_state: np.ndarray = field(repr=False)
    final_block_position_error_px: float = 0.0
    final_block_angle_error_deg: float = 0.0
    decisions: list[DecisionRender] = field(default_factory=list)


def block_polygon_image(pose: np.ndarray) -> np.ndarray:
    """Closed polygon of the T block in image pixels (224x224 frame)."""
    x, y, angle = float(pose[0]), float(pose[1]), float(pose[2])
    cosine, sine = np.cos(angle), np.sin(angle)
    rotation = np.array([[cosine, -sine], [sine, cosine]])
    points = T_BLOCK_VERTICES @ rotation.T + np.array([x, y])
    points = np.concatenate([points, points[:1]], axis=0)
    return points * SCALE


def add_block_outline(ax, pose: np.ndarray, color: str, linewidth: float, dashed: bool = False) -> None:
    style = (0, (5, 3)) if dashed else "solid"
    ax.add_patch(
        MplPolygon(
            block_polygon_image(pose),
            closed=True,
            fill=False,
            edgecolor=color,
            linewidth=linewidth,
            linestyle=style,
            zorder=6,
        )
    )


def add_agent_outline(ax, xy: np.ndarray, color: str, linewidth: float) -> None:
    ax.add_patch(
        Circle(
            (float(xy[0]) * SCALE, float(xy[1]) * SCALE),
            AGENT_RADIUS_WORLD * SCALE,
            fill=False,
            edgecolor=color,
            linewidth=linewidth,
            zorder=6,
        )
    )


def draw_scene(ax, frame: np.ndarray, states: np.ndarray, goal_state: np.ndarray) -> None:
    ax.imshow(frame)
    ax.set_xlim(-1, FRAME_SIZE)
    ax.set_ylim(FRAME_SIZE, -1)
    ax.set_xticks([])
    ax.set_yticks([])
    real_pose = states[2:5]
    agent_xy = states[:2]
    goal_pose = goal_state[2:5]
    add_block_outline(ax, real_pose, COLOR_REAL, 1.6)
    add_block_outline(ax, goal_pose, COLOR_GOAL, 1.4, dashed=True)
    add_agent_outline(ax, agent_xy, COLOR_AGENT, 1.4)


def draw_plan_imagined(ax, decision: DecisionRender, current_block: int) -> None:
    """Draw the selected plan's imagined block trajectory."""
    poses = decision.plan_poses
    points = np.stack([block_polygon_image(pose)[0] for pose in poses], axis=0)
    ax.plot(points[:, 0], points[:, 1], "o-", color=COLOR_PREDICTED, markersize=2.2, linewidth=0.8, alpha=0.85, zorder=5)
    for index, pose in enumerate(poses):
        if index <= current_block:
            add_block_outline(ax, pose, COLOR_PREDICTED, 1.0, dashed=True)


def draw_timeline(ax, t: int, active_decision: int) -> None:
    for offset in (0, 25):
        ax.axvspan(offset, offset + 25, color=COLOR_DECISION_0 if offset == 0 else COLOR_DECISION_1, alpha=0.18)
    for offset in (0, 25):
        ax.annotate(
            f"replan d={offset // 25}",
            xy=(offset, 0.0), xytext=(offset, 0.62),
            fontsize=7, ha="center", color="#374151",
            arrowprops=dict(arrowstyle="->", color="#374151", lw=0.8),
        )
    ax.axvline(t, color="#111827", linewidth=1.6)
    ax.set_xlim(-1, 50)
    ax.set_ylim(0, 1)
    ax.set_xticks([0, 10, 20, 25, 30, 40, 49])
    ax.set_yticks([])
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.set_xlabel(f"indice d'action (t={t}) — zone de la décision {active_decision}", fontsize=8)
    ax.tick_params(labelsize=7)


def draw_cost_panel(ax, decision: DecisionRender, iteration_marker: int | None) -> None:
    iterations = np.arange(1, ITERATIONS + 1)
    ax.plot(iterations, decision.cost_mean, color="#6b7280", linewidth=1.2, label="coût moyen population")
    ax.plot(iterations, decision.cost_median, color="#2563eb", linewidth=1.2, label="coût médian")
    ax.plot(iterations, decision.cost_min, color="#16a34a", linewidth=1.4, label="meilleur candidat")
    ax.set_xlabel("itération CEM", fontsize=8)
    ax.set_ylabel("coût latent", fontsize=8)
    ax.set_title("Convergence de la recherche CEM", fontsize=9)
    ax.grid(alpha=0.2)
    ax.tick_params(labelsize=7)
    sigma_axis = ax.twinx()
    sigma_axis.plot(iterations, decision.sigma_mean, color=COLOR_SIGMA, linewidth=1.0, linestyle="--", label="σ moyen")
    sigma_axis.set_ylabel("σ moyen (actions)", fontsize=7, color=COLOR_SIGMA)
    sigma_axis.tick_params(labelsize=6, labelcolor=COLOR_SIGMA)
    if iteration_marker is not None:
        ax.axvline(iteration_marker, color="#111827", linewidth=1.0, alpha=0.7)
        ax.annotate(
            f"itération {iteration_marker}/30",
            xy=(iteration_marker, decision.cost_mean[iteration_marker - 1]),
            xytext=(iteration_marker - 6.5, ax.get_ylim()[1] * 0.98),
            fontsize=7, color="#111827",
        )
    lines, labels = ax.get_legend_handles_labels()
    sigma_lines, sigma_labels = sigma_axis.get_legend_handles_labels()
    ax.legend(lines + sigma_lines, labels + sigma_labels, loc="upper right", fontsize=6)


def draw_plan_panel(ax, decision: DecisionRender, iteration: int, t: int | None) -> None:
    """Current CEM distribution per block (search) or final plan (execution)."""
    mean = decision.mean_after[iteration]
    std = decision.std_after[iteration]
    ax.set_title(f"Plan sélectionné (blocs 1-{HORIZON}) — itération {iteration + 1}/30", fontsize=9)
    for block in range(HORIZON):
        dx = float(mean[block, 0])
        dy = float(mean[block, 1])
        spread = float(np.linalg.norm(std[block, :2]))
        length = min(1.0, np.hypot(dx, dy))
        ax.add_patch(
            FancyArrow(
                block + 0.5, 0.5, dx * 0.38, dy * 0.38,
                width=0.045, length_includes_head=True, head_width=0.09, head_length=0.09,
                color=COLOR_ELITE if t is None or (t - decision.offset) // 5 >= block else "#9ca3af",
                zorder=4,
            )
        )
        ax.text(block + 0.5, 0.94, f"b{block + 1}", ha="center", fontsize=7, color="#374151")
        ax.text(block + 0.5, 0.10, f"σ={spread:.2f}", ha="center", fontsize=6, color=COLOR_SIGMA)
    ax.set_xlim(0, HORIZON)
    ax.set_ylim(0, 1.1)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xlabel("flèches : action moyenne normalisée par bloc (μ après mise à jour)", fontsize=7)


def draw_error_panel(ax, decision: DecisionRender, executed_blocks: int) -> None:
    blocks = np.arange(1, HORIZON + 1)
    alpha = np.where(blocks <= executed_blocks, 1.0, 0.22)
    bars = ax.bar(
        blocks,
        decision.block_errors_px,
        color=COLOR_PREDICTED,
        label="erreur de position du T (px)",
    )
    for bar, bar_alpha in zip(bars, alpha):
        bar.set_alpha(bar_alpha)
    for index, bar in enumerate(bars):
        if blocks[index] <= executed_blocks:
            ax.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.4,
                f"{decision.block_errors_px[index]:.1f}", ha="center", fontsize=6,
            )
    mse_axis = ax.twinx()
    mse_axis.plot(blocks, decision.block_latent_mse, "o-", color=COLOR_SIGMA, linewidth=1.1, markersize=3, label="MSE latente")
    mse_axis.set_ylabel("MSE latente", fontsize=7, color=COLOR_SIGMA)
    mse_axis.tick_params(labelsize=6, labelcolor=COLOR_SIGMA)
    ax.set_xticks(blocks)
    ax.set_xticklabels([str(value * ACTION_BLOCK) for value in blocks], fontsize=7)
    ax.set_xlabel("bloc (actions écoulées depuis la décision)", fontsize=8)
    ax.set_ylabel("erreur T (px)", fontsize=8)
    ax.set_title("Futur prédit vs réel (blocs exécutés seulement)", fontsize=9)
    ax.grid(alpha=0.2, axis="y")
    ax.tick_params(labelsize=7)
    lines, labels = ax.get_legend_handles_labels()
    mse_lines, mse_labels = mse_axis.get_legend_handles_labels()
    ax.legend(lines + mse_lines, labels + mse_labels, loc="upper left", fontsize=6)


def draw_result_panel(ax, episode: EpisodeRender) -> None:
    outcome = "SUCCÈS" if episode.success else "ÉCHEC"
    color = "#16a34a" if episode.success else "#dc2626"
    ax.text(0.5, 0.78, f"Résultat final : {outcome}", ha="center", va="center", fontsize=13, fontweight="bold", color=color)
    ax.text(
        0.5, 0.52,
        f"erreur T finale : {episode.final_block_position_error_px:.1f} px · "
        f"angle : {episode.final_block_angle_error_deg:.1f}°",
        ha="center", va="center", fontsize=8, color="#374151",
    )
    ax.text(
        0.5, 0.30,
        "contour sombre : T réel (simulateur) · contour orange pointillé : T prédit (plan, décodé)\n"
        "vert pointillé : objectif · bleu : pousseur réel · violet : σ / MSE latente",
        ha="center", va="center", fontsize=7, color="#6b7280",
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")


def render_frame(episode: EpisodeRender, decision_index: int, phase: str, t: int | None, iteration: int) -> np.ndarray:
    decision = episode.decisions[decision_index]
    offset = decision.offset
    figure = plt.figure(figsize=(10.8, 6.4), dpi=88, constrained_layout=True)
    grid = figure.add_gridspec(3, 2, height_ratios=[3.6, 2.1, 0.85], width_ratios=[5.3, 6.7], hspace=0.36, wspace=0.24)
    ax_scene = figure.add_subplot(grid[0, 0])
    ax_plan = figure.add_subplot(grid[1, 0])
    ax_timeline = figure.add_subplot(grid[2, 0])
    ax_cost = figure.add_subplot(grid[0, 1])
    ax_error = figure.add_subplot(grid[1, 1])
    ax_result = figure.add_subplot(grid[2, 1])

    if phase == "search":
        scene_t = offset
        context_frame = episode.observations[scene_t]
        draw_scene(ax_scene, context_frame, episode.states[scene_t], episode.goal_state)
        ax_scene.set_title(
            f"Contexte réel au replan (décision {decision_index}, action {offset}) — recherche en cours",
            fontsize=8,
        )
        draw_cost_panel(ax_cost, decision, iteration_marker=iteration)
        draw_plan_panel(ax_plan, decision, iteration=iteration - 1, t=None)
        draw_error_panel(ax_error, decision, executed_blocks=0)
        ax_error.text(
            0.5, 0.5,
            f"itération {iteration}/30 : comparaison prédit/réel en attente de la fenêtre d'exécution",
            transform=ax_error.transAxes, ha="center", va="center", fontsize=8, color="#6b7280",
        )
        draw_timeline(ax_timeline, scene_t, decision_index)
    else:
        scene_frame = episode.observations[t]
        draw_scene(ax_scene, scene_frame, episode.states[t], episode.goal_state)
        current_block = min(HORIZON - 1, (t - offset) // ACTION_BLOCK)
        draw_plan_imagined(ax_scene, decision, current_block)
        if t < len(episode.actions_postprocessed):
            action = episode.actions_postprocessed[t]
            action_text = f"[{action[0]:+.2f}, {action[1]:+.2f}]"
        else:
            action_text = "fin de l'épisode"
        ax_scene.set_title(
            f"T réel après {t} actions — décision {decision_index} active "
            f"(replan à l'action {offset}) — action envoyée : {action_text}",
            fontsize=8,
        )
        draw_cost_panel(ax_cost, decision, iteration_marker=None)
        draw_plan_panel(ax_plan, decision, iteration=ITERATIONS - 1, t=t)
        draw_error_panel(ax_error, decision, executed_blocks=(t - offset) // ACTION_BLOCK)
        draw_timeline(ax_timeline, t, decision_index)

    draw_result_panel(ax_result, episode)
    # Keep the headline inside the GIF canvas.  A single line is cropped by
    # the renderer at the published 950 px width, which hides the episode
    # identifier precisely where readers need it most.
    figure.suptitle(
        f"Épisode {episode.episode} · départ {episode.start_step} · "
        f"{'SUCCÈS' if episode.success else 'ÉCHEC'}\n"
        "CEM : 300 candidats · 30 itérations · 30 élites · 5 blocs × 5 actions · objectif +25 · budget 50",
        fontsize=9,
        fontweight="bold",
    )
    figure.canvas.draw()
    image = np.asarray(figure.canvas.buffer_rgba())[..., :3]
    plt.close(figure)
    return image


def build_episode_frames(episode: EpisodeRender) -> list[np.ndarray]:
    frames: list[np.ndarray] = []
    for decision_index, decision in enumerate(episode.decisions):
        for iteration in SEARCH_ITERATIONS:
            frames.append(render_frame(episode, decision_index, "search", None, iteration))
        start, end = decision.offset, decision.offset + 25
        for t in range(start, end + (1 if decision_index == len(episode.decisions) - 1 else 0)):
            frames.append(render_frame(episode, decision_index, "execute", t, ITERATIONS))
    return frames


def write_episode_gif(episode: EpisodeRender, output: Path) -> list[np.ndarray]:
    output = Path(output)
    frames = build_episode_frames(episode)
    output.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(output, frames, format="GIF", duration=0.16, loop=0)
    return frames


def write_overview_png(episodes: list[EpisodeRender], output: Path) -> None:
    output = Path(output)
    figure = plt.figure(figsize=(13.2, 7.6), dpi=110, constrained_layout=True)
    grid = figure.add_gridspec(2, 2, hspace=0.34, wspace=0.22, width_ratios=[1, 1.25])
    for row, episode in enumerate(episodes):
        ax_scene = figure.add_subplot(grid[row, 0])
        ax_curves = figure.add_subplot(grid[row, 1])
        frame = episode.observations[-1]
        draw_scene(ax_scene, frame, episode.states[-1], episode.goal_state)
        for decision in episode.decisions:
            for block, pose in enumerate(decision.plan_poses):
                add_block_outline(ax_scene, pose, COLOR_PREDICTED, 0.8, dashed=True)
        outcome = "SUCCÈS" if episode.success else "ÉCHEC"
        color = "#16a34a" if episode.success else "#dc2626"
        ax_scene.set_title(
            f"Épisode {episode.episode} · départ {episode.start_step} · état final (action 50) — {outcome}",
            fontsize=9,
        )
        ax_scene.text(
            0.02, 0.98,
            f"erreur T finale : {episode.final_block_position_error_px:.1f} px",
            transform=ax_scene.transAxes, ha="left", va="top", fontsize=8,
            color=color, fontweight="bold",
            bbox=dict(facecolor="white", alpha=0.75, edgecolor="none", pad=2),
        )
        iterations = np.arange(1, ITERATIONS + 1)
        for decision in episode.decisions:
            label = f"décision {decision.offset // 25} (replan action {decision.offset})"
            ax_curves.plot(iterations, decision.cost_min, "-", linewidth=1.3, label=f"meilleur coût, {label}")
            ax_curves.plot(iterations, decision.sigma_mean, "--", linewidth=1.0, label=f"σ moyen, {label}")
        ax_curves.set_xlabel("itération CEM", fontsize=8)
        ax_curves.set_ylabel("coût latent (trait plein) / σ (pointillé)", fontsize=8)
        ax_curves.set_title(f"Convergence CEM — épisode {episode.episode}", fontsize=9)
        ax_curves.grid(alpha=0.2)
        ax_curves.tick_params(labelsize=7)
        ax_curves.legend(fontsize=6, loc="upper right")
    figure.suptitle(
        "Démo CEM end-to-end : recherche, plan, actions exécutées, futur prédit vs réel — 2 épisodes",
        fontsize=12, fontweight="bold",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output)
    plt.close(figure)
