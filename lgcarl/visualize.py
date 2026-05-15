from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import FancyBboxPatch
from PIL import Image

from lgcarl.config import load_config
from lgcarl.env.routing_env import RoutingEnv
from lgcarl.graph.topology import build_topology
from lgcarl.rl.agent import DQNAgent
from lgcarl.utils import ensure_parent, set_seed


def plot_training_curves(
    metrics_path: str = "results/curves/train_metrics.csv",
    output_path: str = "results/figures/training_curves.png",
) -> None:
    df = pd.read_csv(metrics_path)
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.ravel()
    pairs = [
        ("reward", "Episode Reward"),
        ("avg_delay", "Average Delay"),
        ("loss_rate", "Packet Loss Rate"),
        ("throughput", "Throughput"),
        ("last_loss", "DQN Loss"),
        ("epsilon", "Epsilon"),
    ]
    for ax, (column, title) in zip(axes, pairs):
        if column not in df.columns:
            ax.set_visible(False)
            continue
        sns.lineplot(data=df, x="episode", y=column, ax=ax)
        ax.set_title(title)
        ax.grid(True, alpha=0.25)
    fig.tight_layout()
    out = ensure_parent(output_path)
    fig.savefig(out, dpi=180)
    plt.close(fig)


def plot_eval_bars(
    summary_path: str = "results/tables/eval_summary.csv",
    output_path: str = "results/figures/eval_bars.png",
) -> None:
    df = pd.read_csv(summary_path)
    metrics = ["avg_delay", "loss_rate", "throughput", "max_utilization"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.ravel()
    for ax, metric in zip(axes, metrics):
        sns.barplot(data=df, x="method", y=metric, ax=ax)
        ax.set_title(metric)
        ax.tick_params(axis="x", rotation=25)
        ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    out = ensure_parent(output_path)
    fig.savefig(out, dpi=180)
    plt.close(fig)


def plot_topology(
    config_path: str = "configs/default.yaml",
    output_path: str = "results/figures/topology.png",
) -> None:
    config = load_config(config_path)
    graph = build_topology(config)
    undirected = nx.Graph()
    for u, v, data in graph.edges(data=True):
        if undirected.has_edge(u, v):
            continue
        undirected.add_edge(u, v, weight=data.get("base_delay", 1.0))
    pos = nx.spring_layout(undirected, seed=int(config.get("seed", 7)))
    fig, ax = plt.subplots(figsize=(9, 6))
    nx.draw_networkx_nodes(undirected, pos, node_color="#2f6f8f", node_size=520, ax=ax)
    nx.draw_networkx_labels(undirected, pos, font_color="white", font_size=9, ax=ax)
    nx.draw_networkx_edges(undirected, pos, edge_color="#777777", width=1.5, ax=ax)
    edge_labels = {(u, v): f"{data['weight']:.0f}" for u, v, data in undirected.edges(data=True)}
    nx.draw_networkx_edge_labels(undirected, pos, edge_labels=edge_labels, font_size=7, ax=ax)
    ax.set_axis_off()
    fig.tight_layout()
    out = ensure_parent(output_path)
    fig.savefig(out, dpi=180)
    plt.close(fig)


def _unique_undirected_edges(graph: nx.DiGraph) -> list[tuple[int, int]]:
    seen: set[tuple[int, int]] = set()
    edges: list[tuple[int, int]] = []
    for u, v in graph.edges:
        key = tuple(sorted((int(u), int(v))))
        if key in seen:
            continue
        seen.add(key)
        edges.append(key)
    return edges


def _undirected_projection(graph: nx.DiGraph) -> nx.Graph:
    undirected = nx.Graph()
    undirected.add_nodes_from(int(node) for node in graph.nodes)
    for u, v, data in graph.edges(data=True):
        key = tuple(sorted((int(u), int(v))))
        if undirected.has_edge(*key):
            continue
        undirected.add_edge(key[0], key[1], weight=float(data.get("base_delay", 1.0)))
    return undirected


def _edge_utilization(env: RoutingEnv, edge: tuple[int, int]) -> float:
    u, v = edge
    values = [
        env.simulator.utilization.get((u, v), 0.0),
        env.simulator.utilization.get((v, u), 0.0),
    ]
    return float(max(values))


def _draw_metric_card(ax: plt.Axes, x: float, y: float, width: float, height: float, label: str, value: str) -> None:
    box = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.018,rounding_size=0.018",
        transform=ax.transAxes,
        linewidth=0.8,
        edgecolor="#27434d",
        facecolor="#0d1b20",
        alpha=0.95,
    )
    ax.add_patch(box)
    ax.text(x + 0.04 * width, y + height * 0.68, label, transform=ax.transAxes, color="#8eb4bd", fontsize=8)
    ax.text(
        x + 0.04 * width,
        y + height * 0.22,
        value,
        transform=ax.transAxes,
        color="#f2fbfd",
        fontsize=12,
        fontweight="bold",
    )


def _draw_sparkline(
    ax: plt.Axes,
    values: list[float],
    x: float,
    y: float,
    width: float,
    height: float,
    label: str,
    color: str,
) -> None:
    box = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.016,rounding_size=0.018",
        transform=ax.transAxes,
        linewidth=0.8,
        edgecolor="#233c45",
        facecolor="#0a171c",
        alpha=0.95,
    )
    ax.add_patch(box)
    ax.text(x + 0.03 * width, y + height * 0.78, label, transform=ax.transAxes, color="#8eb4bd", fontsize=8)
    if not values:
        return

    series = np.asarray(values[-40:], dtype=float)
    if series.size == 1:
        series = np.repeat(series, 2)
    lo = float(np.nanmin(series))
    hi = float(np.nanmax(series))
    if abs(hi - lo) < 1e-9:
        lo -= 0.5
        hi += 0.5
    xs = np.linspace(x + 0.04 * width, x + 0.96 * width, len(series))
    ys = y + 0.18 * height + 0.48 * height * ((series - lo) / (hi - lo))
    ax.plot(xs, ys, color=color, linewidth=2.0, transform=ax.transAxes, solid_capstyle="round")
    ax.fill_between(xs, y + 0.18 * height, ys, color=color, alpha=0.12, transform=ax.transAxes)


def _draw_utilization_legend(ax: plt.Axes, cmap: LinearSegmentedColormap, norm: Normalize) -> None:
    gradient = np.linspace(0.0, 1.0, 256).reshape(1, -1)
    x0, x1 = 0.035, 0.36
    y0, y1 = 0.06, 0.085
    ax.imshow(
        gradient,
        extent=(x0, x1, y0, y1),
        transform=ax.transAxes,
        cmap=cmap,
        aspect="auto",
        interpolation="bicubic",
        zorder=0,
    )
    for value in (0.0, 0.75, 1.5):
        x = x0 + (x1 - x0) * norm(value)
        ax.plot([x, x], [y0 - 0.006, y0], color="#7fa2aa", linewidth=1, transform=ax.transAxes)
        ax.text(
            x,
            y0 - 0.018,
            f"{value:.2g}",
            color="#8eb4bd",
            fontsize=7,
            ha="center",
            va="top",
            transform=ax.transAxes,
        )
    ax.text(
        x0,
        y1 + 0.012,
        "link utilization",
        color="#8eb4bd",
        fontsize=8,
        ha="left",
        va="bottom",
        transform=ax.transAxes,
    )


def _path_nodes(path: list[tuple[int, int]]) -> list[int]:
    if not path:
        return []
    nodes = [int(path[0][0])]
    nodes.extend(int(v) for _, v in path)
    return nodes


def _display_run_name(config_path: str, checkpoint_path: Path) -> str:
    config_name = Path(config_path).stem.replace("_", " ")
    if "sndlib polska congested" in config_name:
        config_name = "SNDlib Polska congested"
    elif config_name == "default":
        config_name = "NSFNet uniform"
    elif config_name.startswith("real sndlib polska"):
        config_name = "SNDlib Polska real traffic"
    elif len(config_name) > 34:
        config_name = f"{config_name[:31]}..."
    checkpoint_label = "trained checkpoint" if checkpoint_path.exists() else "untrained policy"
    return f"{config_name} / {checkpoint_label}"


def _render_route_frame(
    env: RoutingEnv,
    pos: dict[int, np.ndarray],
    info: dict,
    histories: dict[str, list[float]],
    frame_idx: int,
    total_frames: int,
    config_name: str,
) -> Image.Image:
    fig = plt.figure(figsize=(14, 7.875), dpi=120)
    fig.patch.set_facecolor("#061115")
    grid = fig.add_gridspec(1, 2, width_ratios=[3.05, 1.55], wspace=0.055)
    ax_graph = fig.add_subplot(grid[0, 0])
    ax_info = fig.add_subplot(grid[0, 1])
    ax_graph.set_facecolor("#061115")
    ax_info.set_facecolor("#061115")
    ax_info.axis("off")
    ax_graph.axis("off")

    undirected = _undirected_projection(env.simulator.graph)
    edges = _unique_undirected_edges(env.simulator.graph)
    util_values = [_edge_utilization(env, edge) for edge in edges]
    traffic_cmap = LinearSegmentedColormap.from_list("traffic", ["#214b58", "#2ab07f", "#f3c54d", "#e05243"])
    norm = Normalize(vmin=0.0, vmax=1.5)
    edge_colors = [traffic_cmap(norm(value)) for value in util_values]
    edge_widths = [1.4 + 4.4 * norm(value) for value in util_values]

    nx.draw_networkx_edges(
        undirected,
        pos,
        edgelist=edges,
        edge_color=edge_colors,
        width=edge_widths,
        alpha=0.82,
        ax=ax_graph,
    )

    selected_path = [(int(u), int(v)) for u, v in info.get("path", [])]
    if selected_path:
        nx.draw_networkx_edges(
            env.simulator.graph,
            pos,
            edgelist=selected_path,
            edge_color="#2fe7ff",
            width=9.5,
            alpha=0.24,
            arrows=True,
            arrowstyle="-|>",
            arrowsize=18,
            connectionstyle="arc3,rad=0.08",
            ax=ax_graph,
        )
        nx.draw_networkx_edges(
            env.simulator.graph,
            pos,
            edgelist=selected_path,
            edge_color="#dffcff",
            width=3.2,
            alpha=0.96,
            arrows=True,
            arrowstyle="-|>",
            arrowsize=15,
            connectionstyle="arc3,rad=0.08",
            ax=ax_graph,
        )

    nodes = list(undirected.nodes)
    nx.draw_networkx_nodes(
        undirected,
        pos,
        nodelist=nodes,
        node_color="#10252c",
        edgecolors="#5c7f88",
        linewidths=1.4,
        node_size=520,
        ax=ax_graph,
    )

    route_nodes = _path_nodes(selected_path)
    if route_nodes:
        nx.draw_networkx_nodes(
            undirected,
            pos,
            nodelist=route_nodes,
            node_color="#16323a",
            edgecolors="#dbfbff",
            linewidths=2.5,
            node_size=650,
            ax=ax_graph,
        )

    src = int(info.get("src", -1))
    dst = int(info.get("dst", -1))
    if src in undirected:
        nx.draw_networkx_nodes(
            undirected,
            pos,
            nodelist=[src],
            node_color="#35d0ff",
            edgecolors="#effcff",
            linewidths=2.0,
            node_size=760,
            ax=ax_graph,
        )
    if dst in undirected:
        nx.draw_networkx_nodes(
            undirected,
            pos,
            nodelist=[dst],
            node_color="#ffd166",
            edgecolors="#fff4cf",
            linewidths=2.0,
            node_size=760,
            ax=ax_graph,
        )

    nx.draw_networkx_labels(undirected, pos, font_color="#f7fbfd", font_size=9, font_weight="bold", ax=ax_graph)

    _draw_utilization_legend(ax_graph, traffic_cmap, norm)

    ax_graph.text(
        0.02,
        0.98,
        "LG-CARL route replay",
        transform=ax_graph.transAxes,
        va="top",
        ha="left",
        color="#f4fbfd",
        fontsize=18,
        fontweight="bold",
    )
    ax_graph.text(
        0.02,
        0.93,
        config_name,
        transform=ax_graph.transAxes,
        va="top",
        ha="left",
        color="#8eb4bd",
        fontsize=10,
    )

    ax_info.text(0.05, 0.95, "Current flow", color="#8eb4bd", fontsize=10, transform=ax_info.transAxes)
    ax_info.text(
        0.05,
        0.892,
        f"{src} -> {dst}",
        color="#f4fbfd",
        fontsize=28,
        fontweight="bold",
        transform=ax_info.transAxes,
    )
    ax_info.text(
        0.05,
        0.84,
        f"path: {' - '.join(str(n) for n in route_nodes) if route_nodes else 'invalid'}",
        color="#b7d4da",
        fontsize=9.5,
        transform=ax_info.transAxes,
    )

    progress = min(1.0, (frame_idx + 1) / max(total_frames, 1))
    ax_info.plot([0.05, 0.95], [0.805, 0.805], color="#16313a", linewidth=5, transform=ax_info.transAxes)
    ax_info.plot([0.05, 0.05 + 0.9 * progress], [0.805, 0.805], color="#35d0ff", linewidth=5, transform=ax_info.transAxes)

    cards = [
        ("Step", str(int(info.get("step", frame_idx)))),
        ("Demand", f"{float(info.get('demand', 0.0)):.2f}"),
        ("Delay", f"{float(info.get('delay', 0.0)):.2f}"),
        ("Loss", f"{float(info.get('loss', 0.0)):.3f}"),
        ("Delivered", f"{float(info.get('delivered', 0.0)):.2f}"),
        ("Max util", f"{float(info.get('max_utilization', 0.0)):.2f}"),
    ]
    x_positions = [0.05, 0.515]
    y_positions = [0.675, 0.545, 0.415]
    idx = 0
    for y in y_positions:
        for x in x_positions:
            label, value = cards[idx]
            _draw_metric_card(ax_info, x, y, 0.405, 0.105, label, value)
            idx += 1

    _draw_sparkline(ax_info, histories["delay"], 0.05, 0.285, 0.9, 0.1, "delay trend", "#35d0ff")
    _draw_sparkline(ax_info, histories["loss"], 0.05, 0.16, 0.9, 0.1, "loss trend", "#ffd166")
    _draw_sparkline(ax_info, histories["util"], 0.05, 0.035, 0.9, 0.1, "max utilization trend", "#e05243")

    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba())
    image = Image.fromarray(rgba).convert("RGB")
    plt.close(fig)
    return image


def make_route_replay_gif(
    config_path: str = "configs/default.yaml",
    checkpoint: str | None = None,
    output_path: str = "results/figures/route_replay.gif",
    frames: int = 80,
    seed: int | None = None,
    duration_ms: int = 120,
    device: str | None = None,
) -> Path:
    config = load_config(config_path)
    seed_value = int(seed if seed is not None else config.get("seed", 7) + 2024)
    set_seed(seed_value)
    graph = build_topology(config)
    env = RoutingEnv(graph, config)
    agent = DQNAgent(
        num_nodes=env.num_nodes,
        model_config=config.get("model", {}),
        dqn_config=config.get("dqn", {}),
        device=device,
    )
    checkpoint_path = Path(checkpoint or config.get("train", {}).get("checkpoint_path", "results/lgcarl.pt"))
    if checkpoint_path.exists():
        agent.load(checkpoint_path)

    undirected = _undirected_projection(graph)
    layout_seed = int(config.get("seed", 7))
    pos = nx.spring_layout(undirected, seed=layout_seed, weight="weight", iterations=250)
    obs = env.reset(seed=seed_value)
    histories = {"delay": [], "loss": [], "util": []}
    rendered_frames: list[Image.Image] = []
    total_frames = min(int(frames), env.episode_length)
    config_name = _display_run_name(config_path, checkpoint_path)

    for frame_idx in range(total_frames):
        action = agent.select_action(obs, training=False)
        obs, _reward, done, info = env.step(action)
        info["step"] = frame_idx
        histories["delay"].append(float(info["delay"]))
        histories["loss"].append(float(info["loss"]))
        histories["util"].append(float(info["max_utilization"]))
        rendered_frames.append(_render_route_frame(env, pos, info, histories, frame_idx, total_frames, config_name))
        if done:
            break

    if not rendered_frames:
        raise RuntimeError("No frames were rendered for the route replay GIF.")

    out = ensure_parent(output_path)
    palette_frames = [frame.convert("P", palette=Image.ADAPTIVE, colors=128) for frame in rendered_frames]
    palette_frames[0].save(
        out,
        save_all=True,
        append_images=palette_frames[1:],
        optimize=True,
        duration=int(duration_ms),
        loop=0,
        disposal=2,
    )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot LG-CARL results.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--train-metrics", default=None)
    parser.add_argument("--eval-summary", default=None)
    parser.add_argument("--out-dir", default="results/figures")
    args = parser.parse_args()

    config = load_config(args.config)
    train_metrics = args.train_metrics or config.get("train", {}).get("log_path", "results/curves/train_metrics.csv")
    eval_summary = args.eval_summary or config.get("eval", {}).get("output_path", "results/tables/eval_summary.csv")

    out_dir = Path(args.out_dir)
    if Path(train_metrics).exists():
        plot_training_curves(train_metrics, str(out_dir / "training_curves.png"))
    if Path(eval_summary).exists():
        plot_eval_bars(eval_summary, str(out_dir / "eval_bars.png"))
    plot_topology(args.config, str(out_dir / "topology.png"))


if __name__ == "__main__":
    main()
