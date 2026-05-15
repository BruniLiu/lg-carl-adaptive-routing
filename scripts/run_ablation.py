from __future__ import annotations

import argparse
import copy
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from lgcarl.config import load_config, save_config
from lgcarl.evaluate import evaluate
from lgcarl.train import train
from lgcarl.utils import ensure_parent


def main() -> None:
    parser = argparse.ArgumentParser(description="Run reward ablations for LG-CARL.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    base = load_config(args.config)
    variants = {
        "full": {},
        "w_o_switch": {"reward": {"lambda_switch": 0.0}},
        "w_o_risk": {"reward": {"eta_risk": 0.0}},
        "delay_loss_only": {"reward": {"gamma_congestion": 0.0, "lambda_switch": 0.0, "eta_risk": 0.0}},
    }
    rows = []
    for name, patch in variants.items():
        config = copy.deepcopy(base)
        for section, values in patch.items():
            config.setdefault(section, {}).update(values)
        config["train"]["log_path"] = f"results/curves/ablation_{name}.csv"
        config["train"]["checkpoint_path"] = f"results/ablation_{name}.pt"
        config["eval"]["output_path"] = f"results/tables/ablation_eval_{name}.csv"
        config_path = f"results/tables/config_ablation_{name}.yaml"
        save_config(config, config_path)
        train(config_path=config_path, episodes=args.episodes, device=args.device)
        df = evaluate(
            config_path=config_path,
            checkpoint=config["train"]["checkpoint_path"],
            episodes=args.eval_episodes,
            methods=["lgcarl"],
            device=args.device,
        )
        row = df.iloc[0].to_dict()
        row["variant"] = name
        rows.append(row)

    out = ensure_parent("results/tables/ablation_summary.csv")
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
