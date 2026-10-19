"""Synthetic data shared by the tests (no real recordings needed)."""

import numpy as np

from som_clustering.config import KEPT_JOINTS

# A rough standing skeleton in metres, y up, person facing -z, right hip at +x.
_STANDING = {
    0: (0.0, 0.0, 0.0), 1: (0.0, 0.15, 0.0), 2: (0.0, 0.32, 0.0), 3: (0.0, 0.50, 0.0),
    4: (0.10, 0.48, 0.0), 5: (0.18, 0.46, 0.0), 6: (0.20, 0.20, 0.0), 7: (0.21, -0.02, 0.0),
    11: (-0.10, 0.48, 0.0), 12: (-0.18, 0.46, 0.0), 13: (-0.20, 0.20, 0.0), 14: (-0.21, -0.02, 0.0),
    18: (0.09, -0.02, 0.0), 19: (0.10, -0.45, 0.02), 20: (0.10, -0.85, 0.0),
    22: (-0.09, -0.02, 0.0), 23: (-0.10, -0.45, 0.02), 24: (-0.10, -0.85, 0.0),
    27: (0.0, 0.68, -0.05),
}


def standing_skeleton():
    """(n_kept_joints, 3) array."""
    return np.array([_STANDING[j] for j in KEPT_JOINTS], dtype=float)


def walking_track(n_frames=60, seed=0):
    """A skeleton drifting slowly with a little noise: (n_frames, n_kept_joints, 3)."""
    rng = np.random.default_rng(seed)
    base = standing_skeleton()
    drift = np.linspace(0, 0.5, n_frames)[:, None, None] * np.array([1.0, 0.0, 0.3])
    return base[None] + drift + rng.normal(0, 0.003, size=(n_frames, len(KEPT_JOINTS), 3))


def rotate_y(joints, angle):
    cos, sin = np.cos(angle), np.sin(angle)
    rotation = np.array([[cos, 0, sin], [0, 1, 0], [-sin, 0, cos]])
    return joints @ rotation.T


def blobs(n_per_blob=200, n_dims=6, separation=8.0, seed=0):
    """Three well separated Gaussian blobs. Returns X, true labels."""
    rng = np.random.default_rng(seed)
    centres = np.eye(3, n_dims) * separation
    X = np.vstack([rng.normal(centre, 1.0, size=(n_per_blob, n_dims)) for centre in centres])
    y = np.repeat(np.arange(3), n_per_blob)
    return X, y


def write_pose_csv(path, rows):
    """Write a 138-column tracker CSV. rows: list of (frame, person, joints34x3) or "HEADER"."""
    header = ["Frame number", "person count"]
    for j in range(34):
        header += [f"x{j}", f"y{j}", f"z{j}", f"p{j}"]
    lines = [",".join(header)]
    for row in rows:
        if row == "HEADER":
            lines.append(",".join(header))
            continue
        frame, person, joints = row
        values = [str(frame), str(person)]
        for j in range(34):
            values += [f"{v:.6f}" for v in joints[j]] + ["90.0"]
        lines.append(",".join(values))
    path.write_text("\n".join(lines) + "\n")


def full_skeleton_34(kept_joints):
    """Embed the kept joints into a 34-joint array (other joints copied from the pelvis)."""
    full = np.repeat(kept_joints[:1], 34, axis=0).copy()
    for position, joint in enumerate(KEPT_JOINTS):
        full[joint] = kept_joints[position]
    return full
