"""Animate one recording as a 3D skeleton.

Usage:
    python pose_tracking.py                                  # default test file
    python pose_tracking.py test_data/A5s10c1-w.csv --track 1
    python pose_tracking.py training_data/A1s1c1-w.csv --canonical
    python pose_tracking.py test_data/A1s10c1-w.csv --save a1.gif

Uses the same reader and skeleton definition as the clustering pipeline, so
multi-person files and joined exports are split into tracks first.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

from som_clustering.config import KEPT_JOINTS, SKELETON_EDGES, TEST_DIR, PreprocessConfig
from som_clustering.data import canonicalize, fill_missing_joints, read_tracks

JOINT_POSITION = {joint: position for position, joint in enumerate(KEPT_JOINTS)}
FRAMES_PER_SECOND = 15


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("file", nargs="?", type=Path, default=TEST_DIR / "A1s10c1-w.csv")
    parser.add_argument("--track", type=int, default=0, help="track number in the file (default 0)")
    parser.add_argument("--canonical", action="store_true",
                        help="show body-centred coordinates as used for clustering")
    parser.add_argument("--save", type=Path, help="save the animation as a GIF instead of showing it")
    return parser.parse_args()


def main():
    args = parse_args()
    tracks = read_tracks(args.file)
    print(f"{args.file.name}: {len(tracks)} track(s): "
          + ", ".join(f"#{i} person {t.person}, {len(t.frames)} frames" for i, t in enumerate(tracks)))
    track = tracks[args.track]

    config = PreprocessConfig()
    joints, _ = fill_missing_joints(track.joints, track.confidence, config)
    if args.canonical:
        joints = canonicalize(joints, config)

    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(projection="3d")
    low, high = np.nanmin(joints, axis=(0, 1)), np.nanmax(joints, axis=(0, 1))
    centre, half = (low + high) / 2, (high - low).max() / 2

    def draw(frame_index):
        ax.clear()
        frame = joints[frame_index]
        for parent, child in SKELETON_EDGES:
            p, c = frame[JOINT_POSITION[parent]], frame[JOINT_POSITION[child]]
            # Plot (x, z, y) so the vertical y axis points up on screen.
            ax.plot([p[0], c[0]], [p[2], c[2]], [p[1], c[1]], color="#2a78d6", linewidth=2)
        ax.scatter(frame[:, 0], frame[:, 2], frame[:, 1], color="#2a78d6", s=10)
        ax.set_xlim(centre[0] - half, centre[0] + half)
        ax.set_ylim(centre[2] - half, centre[2] + half)
        ax.set_zlim(centre[1] - half, centre[1] + half)
        ax.set_xlabel("x")
        ax.set_ylabel("z")
        ax.set_zlabel("y (up)")
        ax.set_title(f"{args.file.name}  track {args.track}  frame {int(track.frames[frame_index])}")

    animation = FuncAnimation(fig, draw, frames=len(joints), interval=1000 / FRAMES_PER_SECOND, repeat=False)
    if args.save:
        animation.save(args.save, writer="pillow", fps=FRAMES_PER_SECOND)
        print(f"Saved {args.save}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
