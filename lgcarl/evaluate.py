from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

from lgcarl.baselines.policies import DQNPolicy, make_baseline_policy
from lgcarl.config import load_config
from lgcarl.env.routing_env import RoutingEnv
from lgcarl.graph.topology import build_topology
from lgcarl.rl.agent import DQNAgent
from lgcarl.utils import ensure_parent, set_seed


def run_episode(env: RoutingEnv, policy: Any, seed: int) -> dict[str, Any]:
    obs = env.reset(seed=seed)
    done = False
    total_reward = 0.0
    rows: list[dict[str, Any]] = []
    while not done:
        action = policy.select_action(obs)
        obs, reward, done, info = env.step(action)
        total_reward += reward
        rows.append(info)

    total_demand = sum(row["demand"] for row in rows)
    total_dropped = sum(row["dropped"] for row in rows)
    total_delivered = sum(row["delivered"] for row in rows)
    return {
        "reward": total_reward,
        "avg_delay": sum(row["delay"] for row in rows) / max(len(rows), 1),
        "loss_rate": total_dropped / max(total_demand, 1e-9),
        "throughput": total_delivered / max(len(rows), 1),
        "max_utilization": max((row["max_utilization"] for row in rows), default=0.0),
        "load_balance_std": sum(row["utilization_std"] for row in rows) / max(len(rows), 1),
        "switch_rate": sum(1 for row in rows if row["switch_cost"] > 0.0) / max(len(rows), 1),
        "invalid_rate": sum(1 for row in rows if row["invalid"]) / max(len(rows), 1),
    }


def make_lgcarl_policy(config: dict[str, Any], env: RoutingEnv, checkpoint: str | Path | None, device: str | None) -> DQNPolicy:
    agent = DQNAgent(
        num_nodes=env.num_nodes,
        model_config=config.get("model", {}),
        dqn_config=config.get("dqn", {}),
        device=device,
    )
    if checkpoint and Path(checkpoint).exists():
        agent.load(checkpoint)
    else:
        print("warning: checkpoint not found; evaluating an untrained LG-CARL policy")
    return DQNPolicy(agent)


def evaluate(
    config_path: str = "configs/default.yaml",
    checkpoint: str | None = None,
    episodes: int | None = None,
    methods: list[str] | None = None,
    device: str | None = None,
) -> pd.DataFrame:
    config = load_config(config_path)
    set_seed(int(config.get("seed", 7)))
    graph = build_topology(config)
    eval_cfg = config.get("eval", {})
    episode_count = int(episodes if episodes is not None else eval_cfg.get("episodes", 20))
    methods = methods or list(eval_cfg.get("methods", ["random_k", "shortest_path", "dynamic_shortest_path", "lgcarl"]))
    checkpoint = checkpoint or config.get("train", {}).get("checkpoint_path", "results/lgcarl.pt")

    summary_rows: list[dict[str, Any]] = []
    for method in methods:
        env = RoutingEnv(graph, config)
        if method == "lgcarl":
            policy = make_lgcarl_policy(config, env, checkpoint, device)
        else:
            policy = make_baseline_policy(method)

        episode_rows = []
        for i in range(episode_count):
            episode_rows.append(run_episode(env, policy, seed=int(config.get("seed", 7)) + 10000 + i))

        mean_row = pd.DataFrame(episode_rows).mean(numeric_only=True).to_dict()
        mean_row["method"] = method
        summary_rows.append(mean_row)
        print(
            f"{method}: avg_delay={mean_row['avg_delay']:.3f} "
            f"loss_rate={mean_row['loss_rate']:.3f} throughput={mean_row['throughput']:.3f}"
        )

    df = pd.DataFrame(summary_rows)
    columns = ["method"] + [col for col in df.columns if col != "method"]
    df = df[columns]
    out_path = ensure_parent(eval_cfg.get("output_path", "results/tables/eval_summary.csv"))
    df.to_csv(out_path, index=False)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate LG-CARL and baselines.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--methods", nargs="*", default=None)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    evaluate(
        config_path=args.config,
        checkpoint=args.checkpoint,
        episodes=args.episodes,
        methods=args.methods,
        device=args.device,
    )


if __name__ == "__main__":
    main()

