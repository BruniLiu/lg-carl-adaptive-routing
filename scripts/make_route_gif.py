from pathlib import Path
import argparse
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lgcarl.visualize import make_route_replay_gif


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a polished LG-CARL route replay GIF.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--output", default="results/figures/route_replay.gif")
    parser.add_argument("--frames", type=int, default=80)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--duration-ms", type=int, default=120)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    out = make_route_replay_gif(
        config_path=args.config,
        checkpoint=args.checkpoint,
        output_path=args.output,
        frames=args.frames,
        seed=args.seed,
        duration_ms=args.duration_ms,
        device=args.device,
    )
    print(f"saved {out}")


if __name__ == "__main__":
    main()
