# Self-Organizing Map posture clustering

Unsupervised clustering of 3D human body postures from skeleton-tracking CSV
files, using a **two-level SOM**: a Kohonen map quantises posture space into
prototype postures, then the prototypes are grouped into posture clusters
(Vesanto & Alhoniemi, 2000).

## Quick start

```bash
pip install -r requirements.txt
python som_pose_tracking.py --quick        # about 20 s smoke test
python som_pose_tracking.py --map-sweep    # full run plus map-size sensitivity
python -m unittest discover -s tests -t .  # unit tests (synthetic data only)
python pose_tracking.py test_data/A5s10c1-w.csv --track 1   # animate one track
```

Results go to `results/`: `summary.md`, `metrics.json`, CSV tables, the model
(`som_model.npz`) and `figures/`.

## Data

`training_data/` and `test_data/` hold one CSV per recording, named
`A{activity}s{session}c{camera}-w.csv`. Each row is one person in one frame:
`Frame number`, `person count` (actually a **person index**), then
`x, y, z, p` per joint (metres, confidence 0–100). 19 of the 34 joints are used.

Things the loader handles:
- **Two-person recordings** (all of A5 and A6): rows alternate person 0 and person 1.
- **Joined exports**: a repeated header row inside a file starts a new segment.
- **Ghost detections**: tracks shorter than 30 frames are dropped.
- **Tracking jumps**: a track is split where the pelvis moves more than 0.3 m in one frame.
- **Missing joints**: confidence `p == 0`, NaN, or a bone longer than 2× its median
  length. These are filled by linear interpolation over time, relative to the pelvis.

## Method

| Step | What happens | Why |
|---|---|---|
| Features | Pelvis at the origin, rotated about the vertical axis so the hips lie on x, divided by the track's median torso length. 18 joints × 3 = 54 features | Removes position, facing direction / camera azimuth and body size, keeps lean and bend. No statistics are fitted, so train and test are processed the same way |
| Splits | `training_data` → fit 80% / validation 20% **by recording** (both cameras and both people stay together). `test_data` is used only for the final report | No leakage between camera views; no choice is made on test data |
| SOM | Rectangular grid sized by the Vesanto heuristic (≈ 5√N units, capped at 400), PCA initialisation, Euclidean distance, batch Kohonen updates (online optional). σ shrinks from max(rows, cols)/2 to a final width of **1.0**, then fine-tuning with early stop on relative weight change | Standard SOM practice; the distance matches the update rule |
| Clusters | Ward clustering of the prototypes that have training hits; a frame's cluster = its BMU's cluster | Two-level clustering separates map resolution from cluster count |
| Number of clusters | **Chosen from the data**: K = 2–20 is scored on **validation** data by silhouette, Davies–Bouldin and Calinski–Harabasz; the best mean rank wins (ties go to the smaller K). Nothing is hard-coded; `--k N` overrides it for experiments, and the override is marked in the summary | Standard internal validity indices, combined by rank so no scale dominates |
| Evaluation | On test data with the final weights: QE, TE, internal indices, ARI / NMI / purity against the activity labels, BMU confidence, cluster margin, novelty rate, temporal stability per track | Activity labels are used only here, never for training or selection |

Scores:
- **BMU confidence** = 1 − d₁ / (95th-percentile training BMU distance), clipped to [0, 1].
- **Cluster margin** = 1 − d_own / d_other: distance to the BMU vs the nearest unit of another cluster.
- **Novelty** = BMU distance above the 99th training percentile.

## Layout

```
som_clustering/   config, data, som, clustering, temporal, plots, pipeline
som_pose_tracking.py   command-line entry point
pose_tracking.py       skeleton animation
tests/                 unit tests
plans/                 design and fix plan
REVIEW.md              review of the previous version
```
