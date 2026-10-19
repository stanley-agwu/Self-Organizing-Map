"""Temporal stability of cluster labels, always within one track.

A track is one person in one continuous recording segment, so runs never
cross file or person boundaries.
"""

import numpy as np
import pandas as pd

FRAMES_PER_SECOND = 30


def run_lengths(labels):
    """Length of the run each frame belongs to."""
    labels = np.asarray(labels)
    if len(labels) == 0:
        return np.empty(0, dtype=int)
    starts = np.r_[0, np.flatnonzero(labels[1:] != labels[:-1]) + 1]
    lengths = np.diff(np.r_[starts, len(labels)])
    return np.repeat(lengths, lengths)


def stable_mask(labels, tracks, min_run_length):
    """True for frames inside a run of at least min_run_length frames of one cluster."""
    labels = np.asarray(labels)
    tracks = np.asarray(tracks)
    mask = np.zeros(len(labels), dtype=bool)
    for track in pd.unique(tracks):
        index = np.flatnonzero(tracks == track)
        mask[index] = run_lengths(labels[index]) >= min_run_length
    return mask


def stability_summary(labels, tracks, min_run_length):
    """Share of stable frames, mean run length and label switches per second."""
    labels = np.asarray(labels)
    tracks = np.asarray(tracks)
    run_counts, switches, frames = [], 0, 0
    for track in pd.unique(tracks):
        track_labels = labels[tracks == track]
        changes = int((track_labels[1:] != track_labels[:-1]).sum())
        switches += changes
        run_counts.append(changes + 1)
        frames += len(track_labels)
    return {
        "stable_fraction": float(stable_mask(labels, tracks, min_run_length).mean()),
        "mean_run_length_frames": float(frames / sum(run_counts)),
        "switches_per_second": float(switches / (frames / FRAMES_PER_SECOND)),
    }
