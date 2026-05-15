from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lgcarl.data.topohub_dataset import save_topohub_topology, slugify


DEFAULT_TOPOLOGIES = [
    "topozoo/Abilene",
    "topozoo/Geant2012",
    "sndlib/polska",
    "sndlib/nobel-germany",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Download/convert real topologies from TopoHub.")
    parser.add_argument("--topologies", nargs="*", default=DEFAULT_TOPOLOGIES)
    parser.add_argument("--out-dir", default="data/topologies")
    parser.add_argument("--capacity-scale", type=float, default=1.0)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for topo_id in args.topologies:
        output_path = out_dir / f"{slugify(topo_id)}.json"
        save_topohub_topology(topo_id, output_path, capacity_scale=args.capacity_scale)
        print(f"saved {topo_id} -> {output_path}")


if __name__ == "__main__":
    main()

