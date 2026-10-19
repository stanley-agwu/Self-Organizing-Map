"""Two-level SOM clustering of 3D human postures.

Usage:
    python som_pose_tracking.py                 # full run, results in results/
    python som_pose_tracking.py --quick         # small map, few epochs (smoke test)
    python som_pose_tracking.py --map-sweep     # also report map-size sensitivity
    python som_pose_tracking.py --algorithm online
    python som_pose_tracking.py --rows 15 --cols 20 --k 8   # override the data-driven K

See README.md for the method and plans/som-clustering-fix.md for the design.
"""

import argparse
from pathlib import Path

from som_clustering.config import RESULTS_DIR, PipelineConfig
from som_clustering.pipeline import run


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=RESULTS_DIR, help="output folder (default: results/)")
    parser.add_argument("--quick", action="store_true", help="small map and few epochs, for a fast check")
    parser.add_argument("--map-sweep", action="store_true", help="report validation QE/TE for other map sizes")
    parser.add_argument("--algorithm", choices=["batch", "online"], help="SOM update rule (default: batch)")
    parser.add_argument("--rows", type=int, help="map rows (default: Vesanto heuristic)")
    parser.add_argument("--cols", type=int, help="map columns (default: Vesanto heuristic)")
    parser.add_argument("--k", type=int,
                        help="override the number of clusters (default: chosen from validation indices)")
    parser.add_argument("--cluster-method", choices=["ward", "kmeans"], help="prototype clustering (default: ward)")
    parser.add_argument("--seed", type=int, help="random seed (default: 42)")
    return parser.parse_args()


def main():
    args = parse_args()
    config = PipelineConfig()
    if args.seed is not None:
        config.seed = args.seed
    if args.quick:
        config.som.rows, config.som.cols = 8, 10
        config.som.n_epochs, config.som.n_finetune = 8, 4
        config.clustering.k_max = 8
    if args.algorithm:
        config.som.algorithm = args.algorithm
    if args.rows and args.cols:
        config.som.rows, config.som.cols = args.rows, args.cols
    if args.cluster_method:
        config.clustering.method = args.cluster_method
    if args.k is not None:
        config.clustering.n_clusters = args.k

    run(config, out_dir=args.out, map_sweep=args.map_sweep)


if __name__ == "__main__":
    main()
