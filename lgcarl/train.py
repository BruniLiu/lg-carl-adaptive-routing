from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm import trange

from lgcarl.config import load_config
from lgcarl.env.routing_env import RoutingEnv
from lgcarl.graph.topology import build_topology
from lgcarl.rl.agent import DQNAgent
from lgcarl.utils import ensure_parent, set_seed


def episode_summary(rows: list[dict[str, Any]], total_reward: float, loss: float | None, episode: int, epsilon: float) -> dict[str, Any]:
    total_demand = sum(row["demand"] for row in rows)
    total_dropped = sum(row["dropped"] for row in rows)
    total_delivered = sum(row["delivered"] for row in rows)
    return {
        "episode": episode,
        "reward": total_reward,
        "avg_delay": sum(row["delay"] for row in rows) / max(len(rows), 1),
        "loss_rate": total_dropped / max(total_demand, 1e-9),
        "throughput": total_delivered / max(len(rows), 1),
        "max_utilization": max((row["max_utilization"] for row in rows), default=0.0),
        "load_balance_std": sum(row["utilization_std"] for row in rows) / max(len(rows), 1),
        "switch_rate": sum(1 for row in rows if row["switch_cost"] > 0.0) / max(len(rows), 1),
        "invalid_rate": sum(1 for row in rows if row["invalid"]) / max(len(rows), 1),
        "last_loss": loss if loss is not None else float("nan"),
        "epsilon": epsilon,
    }


def train(config_path: str = "configs/default.yaml", episodes: int | None = None, device: str | None = None) -> list[dict[str, Any]]:
    config = load_config(config_path)
    set_seed(int(config.get("seed", 7)))
    graph = build_topology(config)
    env = RoutingEnv(graph, config)
    agent = DQNAgent(
        num_nodes=env.num_nodes,
        model_config=config.get("model", {}),
        dqn_config=config.get("dqn", {}),
        device=device,
    )

    train_cfg = config.get("train", {})
    num_episodes = int(episodes if episodes is not None else train_cfg.get("episodes", 100))
    max_steps = int(train_cfg.get("max_steps_per_episode", env.episode_length))
    update_every = max(1, int(train_cfg.get("update_every", 1)))
    print_every = int(train_cfg.get("print_every", 10))
    logs: list[dict[str, Any]] = []

    progress = trange(num_episodes, desc="training", dynamic_ncols=True)
    for episode in progress:
        obs = env.reset(seed=int(config.get("seed", 7)) + episode)
        done = False
        total_reward = 0.0
        step_rows: list[dict[str, Any]] = []
        last_loss: float | None = None
        steps = 0

        while not done and steps < max_steps:
            action = agent.select_action(obs, training=True)
            next_obs, reward, done, info = env.step(action)
            agent.remember(obs, action, reward, next_obs, done)
            loss = agent.update() if agent.steps_done % update_every == 0 else None
            if loss is not None:
                last_loss = loss
            total_reward += reward
            step_rows.append(info)
            obs = next_obs
            steps += 1

        epsilon = agent.epsilon.value(agent.steps_done)
        row = episode_summary(step_rows, total_reward, last_loss, episode, epsilon)
        logs.append(row)
        progress.set_postfix(reward=f"{row['reward']:.2f}", delay=f"{row['avg_delay']:.2f}", loss=f"{row['loss_rate']:.3f}")
        if print_every > 0 and (episode + 1) % print_every == 0:
            print(
                f"episode={episode + 1} reward={row['reward']:.3f} "
                f"avg_delay={row['avg_delay']:.3f} loss_rate={row['loss_rate']:.3f} "
                f"epsilon={row['epsilon']:.3f}"
            )

    log_path = ensure_parent(train_cfg.get("log_path", "results/curves/train_metrics.csv"))
    pd.DataFrame(logs).to_csv(log_path, index=False)
    agent.save(train_cfg.get("checkpoint_path", "results/lgcarl.pt"))
    return logs


def main() -> None:
    parser = argparse.ArgumentParser(description="Train LG-CARL.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    train(config_path=args.config, episodes=args.episodes, device=args.device)


if __name__ == "__main__":
    main()
