"""Loading and preprocessing of skeleton CSV files.

A CSV file can hold several tracks: some files are two exports joined with a
repeated header row, and the "person count" column is really a person index,
so two-person activities have two rows per frame. Each track (export segment x
person) is processed on its own, in frame order:

1. drop non-numeric rows (repeated headers) and split the file into tracks;
   tracks shorter than ``min_track_frames`` are ghost detections and dropped
2. split a track where the pelvis jumps implausibly far in one frame
   (re-acquisition or garbage frame)
3. mark joints as missing when their coordinates are NaN, their confidence is
   at or below the threshold, or the bone ending at them is implausibly long
   (tracking glitch); drop frames with too many missing joints and fill the
   rest by linear interpolation over time, relative to the pelvis
4. convert to a body-centred frame: pelvis at the origin, rotated about the
   vertical axis so the hip line lies on x, scaled by the track's median torso
   length

The transform has no fitted parameters, so training, validation and test data
are treated the same way and nothing is learned from the test set.
"""

import re
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .config import (
    CHEST_JOINT,
    KEPT_JOINTS,
    LEFT_HIP_JOINT,
    RIGHT_HIP_JOINT,
    ROOT_JOINT,
    SKELETON_EDGES,
    VERTICAL_AXIS,
    PreprocessConfig,
)

FILE_NAME_PATTERN = re.compile(r"A(\d+)s(\d+)c(\d+)", re.IGNORECASE)
JOINT_POSITION = {joint: position for position, joint in enumerate(KEPT_JOINTS)}
FEATURE_JOINTS = tuple(j for j in KEPT_JOINTS if j != ROOT_JOINT)
N_FEATURES = 3 * len(FEATURE_JOINTS)
COORDINATE_COLUMNS = [f"{axis}{j}" for j in KEPT_JOINTS for axis in "xyz"]
CONFIDENCE_COLUMNS = [f"p{j}" for j in KEPT_JOINTS]


@dataclass
class PoseDataset:
    """Feature matrix plus one metadata row per frame.

    meta columns: file, track, activity, session, camera, recording, person, frame
    """

    X: np.ndarray        # (n_frames, N_FEATURES) float32
    meta: pd.DataFrame

    def __len__(self):
        return len(self.X)

    def subset(self, mask):
        mask = np.asarray(mask)
        return PoseDataset(self.X[mask], self.meta[mask].reset_index(drop=True))

    @property
    def labels(self):
        """Activity label per frame (for external evaluation only)."""
        return self.meta["activity"].to_numpy()

    @property
    def tracks(self):
        return self.meta["track"].to_numpy()


@dataclass
class Track:
    """One continuous skeleton sequence of one person."""

    joints: np.ndarray      # (n_frames, n_kept_joints, 3) raw coordinates
    confidence: np.ndarray  # (n_frames, n_kept_joints)
    frames: np.ndarray      # (n_frames,)
    segment: int
    person: int


def parse_file_name(path):
    """Return activity, session and camera numbers from a name like A1s2c1-w.csv."""
    match = FILE_NAME_PATTERN.search(Path(path).name)
    if match is None:
        raise ValueError(f"Cannot parse activity/session/camera from {path}")
    activity, session, camera = (int(value) for value in match.groups())
    return {
        "activity": activity,
        "session": session,
        "camera": camera,
        "recording": f"A{activity}s{session}",
    }


def read_tracks(path):
    """Read one CSV file and split it into tracks (export segment x person)."""
    raw = pd.read_csv(path, dtype=str)
    raw.columns = [column.strip() for column in raw.columns]
    numeric = raw.apply(pd.to_numeric, errors="coerce")

    missing_columns = [c for c in COORDINATE_COLUMNS if c not in numeric.columns]
    if missing_columns:
        raise ValueError(f"{path}: missing columns {missing_columns[:6]}")
    n_joints = sum(1 for column in numeric.columns if re.fullmatch(r"x\d+", column))

    # A repeated header row (all-NaN after conversion) starts a new export segment.
    is_header = numeric[COORDINATE_COLUMNS].isna().all(axis=1).to_numpy()
    segment = np.cumsum(is_header)
    numeric = numeric.assign(_segment=segment)[~is_header]

    frame_numbers = (
        numeric["Frame number"] if "Frame number" in numeric.columns
        else pd.Series(np.arange(len(numeric)), index=numeric.index)
    )
    person = (
        numeric["person count"].fillna(0).astype(int) if "person count" in numeric.columns
        else pd.Series(0, index=numeric.index)
    )
    numeric = numeric.assign(_frame=frame_numbers, _person=person)

    tracks = []
    for (segment_id, person_id), group in numeric.groupby(["_segment", "_person"], sort=True):
        # Keep file order; a frame number that does not increase starts a new segment.
        restart = np.r_[False, np.diff(group["_frame"].to_numpy()) <= 0]
        for sub_id, part in group.groupby(np.cumsum(restart)):
            joints = part[COORDINATE_COLUMNS].to_numpy(dtype=float).reshape(-1, len(KEPT_JOINTS), 3)
            if all(c in part.columns for c in CONFIDENCE_COLUMNS):
                confidence = np.nan_to_num(part[CONFIDENCE_COLUMNS].to_numpy(dtype=float), nan=0.0)
            else:
                confidence = np.full(joints.shape[:2], np.inf)
            # 32-joint "Waseda" exports use the opposite y direction.
            if n_joints == 32:
                joints[:, :, 1] = -joints[:, :, 1]
            tracks.append(Track(
                joints=joints,
                confidence=confidence,
                frames=part["_frame"].to_numpy(),
                segment=int(segment_id) * 100 + int(sub_id),
                person=int(person_id),
            ))
    return tracks


def bone_length_outliers(joints, max_ratio):
    """Flag the child joint of every bone longer than max_ratio x its median length."""
    flags = np.zeros(joints.shape[:2], dtype=bool)
    if max_ratio is None or len(joints) == 0:
        return flags
    for parent, child in SKELETON_EDGES:
        p, c = JOINT_POSITION[parent], JOINT_POSITION[child]
        length = np.linalg.norm(joints[:, c] - joints[:, p], axis=1)
        if np.isnan(length).all():
            continue
        median = np.nanmedian(length)
        if median > 0:
            flags[:, c] |= length > max_ratio * median
    return flags


def fill_missing_joints(joints, confidence, config):
    """Mark unreliable joints as missing and interpolate them over time.

    Returns the filled joints and a boolean mask of the frames that were kept.
    """
    missing = np.isnan(joints).any(axis=2) | (confidence <= config.min_joint_confidence)
    missing |= bone_length_outliers(joints, config.max_bone_ratio)
    keep = missing.mean(axis=1) <= config.max_missing_fraction

    filled = joints.copy()
    filled[missing] = np.nan
    filled = filled[keep]
    if len(filled) == 0:
        return filled, keep

    filled = _interpolate_body(filled)

    # When a whole limb glitches, only the first joint is caught at first; once
    # it is filled, the next bone becomes too long. Repeat a few times.
    for _ in range(3):
        outliers = bone_length_outliers(filled, config.max_bone_ratio)
        if not outliers.any():
            break
        filled[outliers] = np.nan
        filled = _interpolate_body(filled)

    # A joint missing in every frame of a track cannot be filled.
    still_missing = np.isnan(filled).any(axis=(1, 2))
    if still_missing.any():
        kept_indices = np.flatnonzero(keep)
        keep[kept_indices[still_missing]] = False
        filled = filled[~still_missing]

    return filled, keep


def _interpolate(joints):
    """Linear interpolation over time of NaN coordinates, edges filled."""
    flat = pd.DataFrame(joints.reshape(len(joints), -1))
    flat = flat.interpolate(method="linear", limit_direction="both", axis=0)
    return flat.to_numpy(copy=True).reshape(joints.shape)


def _interpolate_body(joints):
    """Fill missing joints relative to the pelvis, so a filled limb moves with the body.

    The pelvis itself is interpolated in absolute coordinates.
    """
    root = _interpolate(joints[:, [JOINT_POSITION[ROOT_JOINT]]])
    relative = _interpolate(joints - root)
    return relative + root


def split_at_jumps(track, max_root_step):
    """Split a track wherever the pelvis moves more than max_root_step in one frame.

    Such jumps are tracking re-acquisitions or garbage frames, not real motion.
    """
    root = track.joints[:, JOINT_POSITION[ROOT_JOINT]]
    step = np.linalg.norm(np.diff(root, axis=0), axis=1)
    breaks = np.flatnonzero(np.nan_to_num(step, nan=0.0) > max_root_step) + 1
    bounds = np.r_[0, breaks, len(root)]
    return [
        (part, Track(
            joints=track.joints[start:end],
            confidence=track.confidence[start:end],
            frames=track.frames[start:end],
            segment=track.segment,
            person=track.person,
        ))
        for part, (start, end) in enumerate(zip(bounds[:-1], bounds[1:]))
    ]


def canonicalize(joints, config):
    """Body-centred coordinates: pelvis origin, hips along x, unit torso length."""
    root = joints[:, JOINT_POSITION[ROOT_JOINT]]
    relative = joints - root[:, None, :]

    if config.alignment == "yaw":
        relative = _align_yaw(relative)
    elif config.alignment != "none":
        raise ValueError(f"Unknown alignment {config.alignment!r}")

    if config.scale_by_torso:
        torso = np.linalg.norm(relative[:, JOINT_POSITION[CHEST_JOINT]], axis=1)
        scale = np.median(torso)
        if scale > 1e-6:
            relative = relative / scale

    return relative


def _align_yaw(relative):
    """Rotate each frame about the vertical axis so the hip line points along +x."""
    a_axis, b_axis = [axis for axis in range(3) if axis != VERTICAL_AXIS]

    hip = relative[:, JOINT_POSITION[RIGHT_HIP_JOINT]] - relative[:, JOINT_POSITION[LEFT_HIP_JOINT]]
    a, b = hip[:, a_axis], hip[:, b_axis]
    angle = np.arctan2(b, a)

    # If the hips are almost vertical the direction is undefined: reuse the
    # previous frame's angle (or 0 for the first frames).
    valid = np.hypot(a, b) > 1e-6
    if not valid.all():
        angle = pd.Series(np.where(valid, angle, np.nan)).ffill().fillna(0.0).to_numpy()

    cos, sin = np.cos(angle)[:, None], np.sin(angle)[:, None]
    rotated = relative.copy()
    rotated[:, :, a_axis] = relative[:, :, a_axis] * cos + relative[:, :, b_axis] * sin
    rotated[:, :, b_axis] = -relative[:, :, a_axis] * sin + relative[:, :, b_axis] * cos
    return rotated


def joints_to_features(joints):
    """(n, n_kept_joints, 3) canonical joints -> (n, N_FEATURES), dropping the pelvis."""
    positions = [JOINT_POSITION[j] for j in FEATURE_JOINTS]
    return joints[:, positions, :].reshape(len(joints), -1).astype(np.float32)


def features_to_joints(features):
    """(n, N_FEATURES) -> (n, n_kept_joints, 3), with the pelvis put back at the origin."""
    features = np.atleast_2d(features)
    joints = np.zeros((len(features), len(KEPT_JOINTS), 3))
    positions = [JOINT_POSITION[j] for j in FEATURE_JOINTS]
    joints[:, positions, :] = features.reshape(len(features), -1, 3)
    return joints


def load_file(path, config=None):
    """Load and preprocess every usable track in one file.

    Returns (features, metadata DataFrame); both empty if nothing is usable.
    """
    config = config or PreprocessConfig()
    info = parse_file_name(path)
    name = Path(path).name

    features, metas = [], []
    pieces = [
        (track, part, piece)
        for track in read_tracks(path)
        for part, piece in split_at_jumps(track, config.max_root_step)
    ]
    for track, part, piece in pieces:
        if len(piece.frames) < config.min_track_frames:
            continue
        filled, keep = fill_missing_joints(piece.joints, piece.confidence, config)
        if len(filled) < config.min_track_frames:
            continue

        features.append(joints_to_features(canonicalize(filled, config)))
        meta = pd.DataFrame({"frame": piece.frames[keep]})
        meta.insert(0, "file", name)
        meta.insert(1, "track", f"{name}#seg{track.segment}p{track.person}t{part}")
        meta["person"] = track.person
        for key, value in info.items():
            meta[key] = value
        metas.append(meta)

    if not features:
        return np.empty((0, N_FEATURES), dtype=np.float32), pd.DataFrame()
    return np.vstack(features), pd.concat(metas, ignore_index=True)


def load_dataset(directory, config=None, pattern="*.csv"):
    """Load every matching file in a folder, in sorted (reproducible) order."""
    config = config or PreprocessConfig()
    paths = sorted(Path(directory).glob(pattern))
    if not paths:
        raise FileNotFoundError(f"No files matching {pattern} in {directory}")

    features, metas = [], []
    for path in paths:
        file_features, file_meta = load_file(path, config)
        if len(file_features) == 0:
            warnings.warn(f"{path.name}: no usable frames, skipped")
            continue
        features.append(file_features)
        metas.append(file_meta)

    return PoseDataset(np.vstack(features), pd.concat(metas, ignore_index=True))


def split_by_recording(dataset, validation_fraction, seed):
    """Split into fit and validation sets by recording (activity + session).

    Both camera views and all people of one recording stay on the same side,
    and every activity with more than one recording contributes to validation.
    """
    rng = np.random.default_rng(seed)
    meta = dataset.meta
    validation_recordings = []

    for activity in sorted(meta["activity"].unique()):
        recordings = np.array(sorted(meta.loc[meta["activity"] == activity, "recording"].unique()))
        if len(recordings) < 2:
            continue
        n_validation = max(1, int(round(validation_fraction * len(recordings))))
        chosen = rng.choice(recordings, size=n_validation, replace=False)
        validation_recordings.extend(chosen.tolist())

    is_validation = meta["recording"].isin(validation_recordings).to_numpy()
    return dataset.subset(~is_validation), dataset.subset(is_validation)
