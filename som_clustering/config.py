"""Shared constants: paths, skeleton layout and default hyper-parameters."""

from dataclasses import dataclass, field, asdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRAINING_DIR = PROJECT_ROOT / "training_data"
TEST_DIR = PROJECT_ROOT / "test_data"
RESULTS_DIR = PROJECT_ROOT / "results"

SEED = 42

# ------------------------------------------------------------------
# Skeleton layout
# ------------------------------------------------------------------
# Raw joint ids kept from the tracker output (same 19 joints as before).
# Names follow the labels used in pose_tracking.py.
KEPT_JOINTS = (0, 1, 2, 3, 4, 5, 6, 7, 11, 12, 13, 14, 18, 19, 20, 22, 23, 24, 27)
JOINT_NAMES = {
    0: "pelvis", 1: "stomach", 2: "chest", 3: "neck",
    4: "r_shoulder", 5: "r_forearm", 6: "r_elbow", 7: "r_wrist",
    11: "l_shoulder", 12: "l_forearm", 13: "l_elbow", 14: "l_wrist",
    18: "r_hip", 19: "r_knee", 20: "r_ankle",
    22: "l_hip", 23: "l_knee", 24: "l_ankle",
    27: "head",
}
# Bones between raw joint ids.
SKELETON_EDGES = (
    (0, 1), (1, 2), (2, 3), (3, 27),            # spine and head
    (3, 4), (4, 5), (5, 6), (6, 7),             # right arm
    (3, 11), (11, 12), (12, 13), (13, 14),      # left arm
    (0, 18), (18, 19), (19, 20),                # right leg
    (0, 22), (22, 23), (23, 24),                # left leg
)

ROOT_JOINT = 0        # origin of the body frame (pelvis)
CHEST_JOINT = 2       # with the pelvis, defines torso length (body-size scale)
RIGHT_HIP_JOINT = 18  # the hip line defines the facing direction
LEFT_HIP_JOINT = 22
VERTICAL_AXIS = 1     # y points up in the camera world frame (checked on the data)

# Joint ids in each file format. 138 columns = 34 joints + frame/person columns,
# 128 columns = 32 joints ("Waseda" format, y axis flipped).
FORMAT_JOINT_COUNT = {138: 34, 128: 32}


@dataclass
class PreprocessConfig:
    # A joint whose confidence p is <= this value is treated as missing.
    # Joint 3 is always reported with p of about 1, so keep this at 0.
    min_joint_confidence: float = 0.0
    # Drop a frame when more than this share of kept joints is missing.
    max_missing_fraction: float = 0.5
    # A joint is also treated as missing when the bone ending at it is longer
    # than this multiple of that bone's median length in the file (tracking glitch).
    max_bone_ratio: float = 2.0
    # Tracks (one person in one export segment) shorter than this are ghost
    # detections and are dropped (30 frames is about 1 s).
    min_track_frames: int = 30
    # Split a track where the pelvis moves more than this (metres) in one frame.
    # 0.3 m per frame is about 9 m/s at 30 fps, far above real motion.
    max_root_step: float = 0.3
    # "yaw": rotate about the vertical axis so the hips face the same way.
    # "none": keep the camera orientation.
    alignment: str = "yaw"
    # Divide by the file's median torso length.
    scale_by_torso: bool = True


@dataclass
class SOMConfig:
    # None means use the Vesanto heuristic (about 5 * sqrt(N) units, capped).
    rows: int | None = None
    cols: int | None = None
    max_units: int = 400
    algorithm: str = "batch"          # "batch" or "online"
    n_epochs: int = 40                # epochs over which sigma (and alpha) decay
    n_finetune: int = 20              # extra epochs at sigma_end, with early stop
    # Final neighbourhood width. Chosen as 1.0 (SOM Toolbox convention): on
    # validation data 0.5 gave TE 0.34 vs 0.13 at 1.0, for a small QE gain.
    sigma_end: float = 1.0
    alpha_start: float = 0.5          # online only
    alpha_end: float = 0.01           # online only
    tol: float = 1e-3                 # relative weight change for early stop


@dataclass
class ClusteringConfig:
    method: str = "ward"              # "ward" or "kmeans"
    # None (default): K is chosen from the data, by the mean rank of silhouette,
    # Davies-Bouldin and Calinski-Harabasz on validation data over k_min..k_max.
    # An integer overrides that choice (for experiments); the index table is
    # still computed and reported.
    n_clusters: int | None = None
    k_min: int = 2
    k_max: int = 20
    silhouette_sample: int = 5000     # seeded subsample for the silhouette
    min_run_length: int = 10          # frames; shorter runs count as unstable


@dataclass
class PipelineConfig:
    seed: int = SEED
    validation_fraction: float = 0.2
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    som: SOMConfig = field(default_factory=SOMConfig)
    clustering: ClusteringConfig = field(default_factory=ClusteringConfig)

    def to_dict(self):
        return asdict(self)
