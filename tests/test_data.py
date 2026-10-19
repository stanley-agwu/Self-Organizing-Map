import tempfile
import unittest
from pathlib import Path

import numpy as np

from som_clustering.config import PreprocessConfig
from som_clustering.data import (
    JOINT_POSITION,
    N_FEATURES,
    Track,
    bone_length_outliers,
    canonicalize,
    features_to_joints,
    fill_missing_joints,
    joints_to_features,
    load_file,
    parse_file_name,
    read_tracks,
    split_at_jumps,
    split_by_recording,
)
from som_clustering.data import PoseDataset

from tests.helpers import full_skeleton_34, rotate_y, standing_skeleton, walking_track, write_pose_csv

import pandas as pd


class CanonicalFeatureTests(unittest.TestCase):
    def setUp(self):
        self.config = PreprocessConfig()
        self.track = walking_track()

    def test_invariant_to_position_facing_and_body_size(self):
        reference = joints_to_features(canonicalize(self.track, self.config))
        moved = rotate_y(self.track, 1.1) * 1.3 + np.array([2.0, 0.5, -3.0])
        transformed = joints_to_features(canonicalize(moved, self.config))
        np.testing.assert_allclose(transformed, reference, atol=1e-5)

    def test_keeps_lean(self):
        """Rotation about a horizontal axis (leaning) must change the features."""
        upright = joints_to_features(canonicalize(self.track, self.config))
        angle = 0.4
        lean = np.array([[1, 0, 0], [0, np.cos(angle), -np.sin(angle)], [0, np.sin(angle), np.cos(angle)]])
        leaning = joints_to_features(canonicalize(self.track @ lean.T, self.config))
        self.assertGreater(np.abs(leaning - upright).max(), 0.1)

    def test_geometry_preserved_and_round_trip(self):
        features = joints_to_features(canonicalize(self.track, self.config))
        self.assertEqual(features.shape[1], N_FEATURES)
        joints = features_to_joints(features)
        torso = np.linalg.norm(joints[:, JOINT_POSITION[2]], axis=1)
        self.assertAlmostEqual(float(np.median(torso)), 1.0, places=5)
        hip = joints[:, JOINT_POSITION[18]] - joints[:, JOINT_POSITION[22]]
        np.testing.assert_allclose(hip[:, 2], 0.0, atol=1e-6)  # hips along x after alignment
        self.assertTrue((hip[:, 0] > 0).all())


class MissingJointTests(unittest.TestCase):
    def test_zero_confidence_joint_is_interpolated_relative_to_pelvis(self):
        track = walking_track(40)
        confidence = np.full(track.shape[:2], 90.0)
        wrist = JOINT_POSITION[7]
        confidence[10:15, wrist] = 0.0
        corrupted = track.copy()
        corrupted[10:15, wrist] = 99.0  # junk that must not survive

        filled, keep = fill_missing_joints(corrupted, confidence, PreprocessConfig())
        self.assertTrue(keep.all())
        np.testing.assert_allclose(filled[10:15, wrist], track[10:15, wrist], atol=0.02)

    def test_bone_glitch_is_flagged_and_removed(self):
        track = walking_track(40)
        glitched = track.copy()
        glitched[20, JOINT_POSITION[20]] += np.array([2.0, 0.0, 0.0])  # ankle jumps 2 m
        flags = bone_length_outliers(glitched, 2.0)
        self.assertTrue(flags[20, JOINT_POSITION[20]])
        filled, _ = fill_missing_joints(glitched, np.full(track.shape[:2], 90.0), PreprocessConfig())
        np.testing.assert_allclose(filled[20, JOINT_POSITION[20]], track[20, JOINT_POSITION[20]], atol=0.02)

    def test_frames_with_too_many_missing_joints_are_dropped(self):
        track = walking_track(20)
        confidence = np.full(track.shape[:2], 90.0)
        confidence[5, :15] = 0.0
        _, keep = fill_missing_joints(track, confidence, PreprocessConfig())
        self.assertFalse(keep[5])
        self.assertEqual(keep.sum(), 19)

    def test_track_is_split_at_pelvis_jump(self):
        joints = walking_track(50)
        joints[30:] += np.array([1.5, 0.0, 0.0])
        track = Track(joints, np.full(joints.shape[:2], 90.0), np.arange(50), segment=0, person=0)
        pieces = split_at_jumps(track, max_root_step=0.3)
        self.assertEqual([len(piece.frames) for _, piece in pieces], [30, 20])


class FileReadingTests(unittest.TestCase):
    def test_tracks_split_by_person_and_repeated_header(self):
        skeleton = full_skeleton_34(standing_skeleton())
        rows = []
        for frame in range(3):
            rows.append((frame, 0, skeleton))
            rows.append((frame, 1, skeleton + 1.0))
        rows.append("HEADER")
        for frame in range(4):
            rows.append((frame, 0, skeleton + 2.0))
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "A1s2c1-w.csv"
            write_pose_csv(path, rows)
            tracks = read_tracks(path)
        self.assertEqual([(t.segment, t.person, len(t.frames)) for t in tracks],
                         [(0, 0, 3), (0, 1, 3), (100, 0, 4)])
        np.testing.assert_allclose(tracks[1].joints[0], standing_skeleton() + 1.0, atol=1e-5)

    def test_short_ghost_tracks_are_dropped(self):
        skeleton = full_skeleton_34(standing_skeleton())
        rows = [(frame, 0, skeleton) for frame in range(40)] + [(5, 1, skeleton + 1.0)]
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "A3s4c2-w.csv"
            write_pose_csv(path, rows)
            features, meta = load_file(path)
        self.assertEqual(len(features), 40)
        self.assertEqual(set(meta["person"]), {0})
        self.assertEqual(meta["activity"].iloc[0], 3)

    def test_parse_file_name(self):
        self.assertEqual(parse_file_name("A12s3c2-w.csv")["recording"], "A12s3")


class SplitTests(unittest.TestCase):
    def test_recordings_never_cross_the_split(self):
        rows = []
        for activity in (1, 2, 3):
            for session in range(1, 6):
                for camera in (1, 2):
                    for frame in range(5):
                        rows.append({"activity": activity, "session": session, "camera": camera,
                                     "recording": f"A{activity}s{session}", "track": f"{activity}{session}{camera}",
                                     "frame": frame})
        meta = pd.DataFrame(rows)
        dataset = PoseDataset(np.zeros((len(meta), N_FEATURES), dtype=np.float32), meta)
        fit, val = split_by_recording(dataset, 0.2, seed=1)
        self.assertFalse(set(fit.meta["recording"]) & set(val.meta["recording"]))
        self.assertEqual(sorted(val.meta["activity"].unique()), [1, 2, 3])
        self.assertEqual(len(fit) + len(val), len(dataset))


if __name__ == "__main__":
    unittest.main()
