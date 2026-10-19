# Plan: SOM Clustering Fix

Branch: `feat/som_clustering_fix`
Source of issues: [`REVIEW.md`](../REVIEW.md)
Goal: rebuild the pipeline so it follows standard Self-Organizing Map practice and two-level SOM clustering (Kohonen 2001; Vesanto & Alhoniemi 2000), and fix every issue raised in the review.

---

## 1. Facts checked before planning

| Check | Result | Consequence |
|---|---|---|
| Data size | `training_data/` 140 files, 40,697 frames; `test_data/` 40 files, 11,951 frames. All files are 138-column (34-joint) format. 150–732 frames per file | Vectorised NumPy is enough; no GPU needed |
| `-1` markers | **None** in any raw coordinate | The `replace(-1, 0)` step never did anything. Missing joints have to be found another way |
| Confidence `p` | Range 0–100. `p == 0` for 0.04% of kept joint values. **Joint 3 (neck) is always ≤ 5** (around 1), so it is a low-confidence joint that the SDK always reports | Missing joint = `p == 0` (configurable). A higher threshold would wipe out joint 3 |
| Header/NaN rows | 2 repeated header rows in train and 1 in test (from concatenated exports) | Drop non-numeric rows per file |
| Vertical axis | In both cameras, head − ankle ≈ (−0.1, **1.35**, 0.1) m, so **y is up** | Facing direction (yaw) can be removed by rotating about y, keeping lean and bend information |
| Python | The committed `venv` points to another user's interpreter. `C:\Program Files\Python313` has numpy 2.3.5, pandas **3.0.1**, scipy 1.16.3, sklearn 1.8.0, matplotlib 3.10.8. No pytest | `fillna(method=...)` raises on pandas 3, so the fix is required. Tests use stdlib `unittest` |

---

## 2. Design decisions

### 2.1 Features (fixes review issues 3 and 4, and the per-feature scaling problem)
Per file, in time order:
1. Read the file on its own (no blind concat). Detect the 138- or 128-column format per file and flip y for the 128 (Waseda) format.
2. Drop non-numeric rows and sort by `Frame number`.
3. **Missing joints:** a joint with confidence `p <= MIN_JOINT_CONFIDENCE` (default 0) is set to NaN. Drop frames where more than 50% of kept joints are missing. Fill the rest by **linear interpolation over time inside the file**, then edge-fill. This happens **before** any transform.
4. Keep the same 19 joints as before: `0,1,2,3,4,5,6,7,11,12,13,14,18,19,20,22,23,24,27`.
5. **Body-centred, view-invariant coordinates** that keep the skeleton's shape:
   - origin = pelvis (joint 0)
   - rotate about the vertical y axis so the hip line (18→22) lies on the x axis. This removes camera azimuth and facing direction, which is the real reason c1 and c2 looked different, but keeps lean and bend
   - divide by the file's **median torso length** |chest(2) − pelvis(0)|, which removes body size
   - drop the pelvis (always 0), giving 18 joints × 3 = **54 features**
6. **No statistics are fitted on test data.** The transform has no fitted parameters, so train, validation and test are handled the same way by construction, with no per-camera or per-batch scaling. Geometry is preserved, so prototypes can be drawn as real skeletons.

### 2.2 Data splits (fixes issue 2)
- `training_data/` is split into **fit** (80%) and **validation** (20%) by recording group `(activity, session)`, seeded. Both camera views of the same recording stay in the same split, so the two views cannot leak across.
- `test_data/` is used **only for the final report**, never for any choice.
- Activity labels (`A1`–`A9`) are parsed from file names. They are used **only** for external evaluation, never for training or selection.

### 2.3 SOM (fixes issues 5 and 6, plus grid, initialisation and seed)
New class `SelfOrganizingMap` (vectorised NumPy):
- **Full rectangular grid** `rows × cols`. Default size from the Vesanto heuristic `≈ 5·√N` units, with the side ratio from the two largest PCA eigenvalues. Capped (frames are strongly autocorrelated at 30 fps, so N overstates the independent samples) and configurable.
- **PCA linear initialisation**, which is deterministic.
- **Euclidean distance everywhere.** It matches the Kohonen update `w += α·h·(x − w)`. Cosine and Hamming are removed.
- **Batch SOM** (default; deterministic, standard Kohonen batch map) and **online SOM** (optional) share one schedule:
  - σ decays **exponentially from σ₀ = max(rows, cols)/2 to σ_end = 0.5** over the schedule. At σ = 0.5 a direct neighbour gets weight e⁻² ≈ 0.14, then falls off quickly, so the map ends close to k-means refinement
  - online α decays from 0.5 to 0.01 per step
  - then up to `n_finetune` extra epochs at σ_end, which **stop early** when the relative weight change ‖ΔW‖/‖W‖ < `tol`. The weight-change plot uses **the same relative measure and the same `tol`**
- **QE** = mean Euclidean distance from each sample to its BMU, with one definition everywhere. **TE** = share of samples whose first and second BMU are not 4-neighbours.
- Per-epoch history of QE and TE on **fit and validation data**. These are the train/held-out curves that replace the mislabelled "testing" plots.
- Saved as a small `.npz` (weights, grid, config, history, confidence constants, prototype cluster labels). No pickle, and no training data inside.
- A single seeded `np.random.Generator` is passed everywhere (plus `random_state` for sklearn).

### 2.4 Two-level clustering (fixes issue 7)
Level 1 = SOM prototypes. Level 2 = cluster the **prototypes**:
- Agglomerative **Ward** clustering on the prototypes that have at least one hit (k-means available as an option). Units with no hits get the label of the nearest cluster centroid, for display only.
- A data sample's cluster = the cluster of its BMU.
- **Choose K ∈ [2, 20] on validation data** using standard internal indices: **silhouette** (max; seeded subsample), **Davies–Bouldin** (min), **Calinski–Harabasz** (max). Decision rule: **best mean rank** across the three. Ranks avoid the min-max stretching of the old combined score, and no index enters twice.
- Map size is **not** chosen by QE, because QE always falls as the map grows. An optional `--map-sweep` reports validation QE/TE against map size for sensitivity only.

### 2.5 Confidence scores (fixes issue 8)
The misleading "discriminant score" is replaced by:
- **BMU confidence** = clip(1 − d₁ / d_ref, 0, 1), where d_ref = 95th percentile of **training** BMU distances. It is fitted once on fit data and saved with the model.
- **BMU margin** = 1 − d₁/d₂. Scale-free, in [0, 1].
- **Novelty flag** = d₁ > 99th percentile of training BMU distances.
- Per-sample silhouette on cluster labels (standard).

### 2.6 Evaluation (fixes issues 1 and 2)
On **test data, with the final weights**:
- QE, TE, silhouette, Davies–Bouldin, Calinski–Harabasz
- confidence, margin and novelty statistics
- **external validity vs activity labels**: ARI, NMI, purity, and a contingency heat-map (cluster × activity)
- **temporal stability per file** (never across file boundaries): cluster runs shorter than `min_run_length` (10 frames) are flagged. Metrics are reported on all frames and on frames in stable runs
- `results/metrics.json` plus `results/summary.md`

### 2.7 Architecture
```
som_clustering/
  __init__.py
  config.py      # paths, joint layout, skeleton edges, hyper-parameters, seed
  data.py        # per-file loading, labels, missing joints, canonical features, splits
  som.py         # SelfOrganizingMap: init, batch/online train, bmu, QE, TE, save/load
  clustering.py  # prototype clustering, internal and external indices, K selection, confidence
  temporal.py    # per-file run-length analysis
  plots.py       # all figures (closed after saving, colour lookups by key)
  pipeline.py    # fit → select K → evaluate → report
som_pose_tracking.py  # thin CLI entry point (`--quick`, `--map-sweep`, `--online`)
pose_tracking.py      # fixed standalone skeleton animation (shared joint config)
tests/                # unittest: preprocessing, SOM, clustering, temporal
requirements.txt, README.md
```
- `data_preprocessing.py` is removed and replaced by `som_clustering/data.py`, so there is only one pipeline.
- Each model is loaded **once** from `.npz`. Training files are never reloaded per model.

---

## 3. Traceability: review issue → fix

| Review item | Fix |
|---|---|
| 1. Testing plots show training data | Test metrics are computed on test data with final weights. "Testing epoch" plots removed. Fit/validation history plotted per epoch (§2.3, §2.6) |
| 2. Mixed train/test model selection, no held-out data | Fit/validation/test split by recording. K chosen on validation only. Test used only for the report (§2.2, §2.4) |
| 3. Test scaled differently | Parameter-free, view-invariant, body-size-normalised transform applied the same way everywhere (§2.1) |
| 4. Missing joints never handled | `p == 0` → NaN → temporal interpolation **before** the transform (§2.1) |
| 5. Training never settles | σ → 0.5 and α → 0.01 schedules, batch SOM, fine-tune phase with early stop (§2.3) |
| 6. Cosine vs Euclidean mismatch | Euclidean only (§2.3) |
| 7. Neurons = clusters, no validity indices, labels unused | Two-level clustering, silhouette/DB/CH, ARI/NMI/purity (§2.4, §2.6) |
| 8. Discriminant score mis-scaled and misnamed | BMU confidence, margin and novelty, normalised on training data (§2.5) |
| 88 MB pickles committed | Small `.npz` models. `trained_som_models/*.pkl` untracked and git-ignored (history is not rewritten) |
| No random seed | One seeded generator plus `random_state` (§2.3) |
| Test phase reloads training files per model | Models loaded once from `.npz`, data loaded once (§2.7) |
| Colour lookups can crash | Colours keyed by cluster id from one shared map |
| `pose_tracking.py` wrong folder | Path argument defaulting to `test_data/…`, CSV round-trip removed, shared joint config |
| Ragged grids | Full `rows × cols` grids only |
| Squared vs unsquared QE | One QE definition |
| Normalisation computed on scored batch | Confidence constants fitted on fit data and saved |
| Wrong debug print, "± Std" without error bars, mixed-scale combined plot | Old selection plots replaced. New K-selection plot shows each index on its own axis |
| `fillna(method=)` deprecated or removed | `.ffill()` / `.bfill()` / `interpolate` |
| Concat of mixed formats | Per-file format detection |
| Unsorted glob, runs crossing file boundaries | `sorted(glob)`, all temporal logic per file |
| Duplicate `training_error.png`, figure leaks | Every figure saved once and closed |
| Relative vs absolute convergence threshold mismatch | Same relative measure and `tol` in check and plot |
| Python loops (distances, updates, transform) | Vectorised with `cdist` / broadcasting, chunked |
| Winners taken mid-epoch | Labels recomputed with final weights |
| Dead code | Removed with the old class |
| Wrong `joint_connections` | Skeleton edges defined for the real 19-joint layout |
| Posture-match file overwritten | File name includes cluster id |
| Nested-quote f-strings (Python ≥ 3.12) | Not used |
| `CompetitiveLearning` / `classify` naming | `SelfOrganizingMap`, `bmu()`, `assign_clusters()` |
| No requirements, README, tests | Added |

---

## 4. Steps
1. Write `som_clustering/config.py` and `data.py`, then check feature shapes and invariances on the real data.
2. Write `som.py` and test it on synthetic data (QE falls, TE is correct on a known grid).
3. Write `clustering.py` and `temporal.py`, and test them on synthetic blobs (ARI ≈ 1).
4. Write `plots.py` and `pipeline.py`, plus the CLI in `som_pose_tracking.py`.
5. Fix `pose_tracking.py`. Remove `data_preprocessing.py`.
6. Add `requirements.txt`, `README.md` and `.gitignore` entries. Untrack the `.pkl` models.
7. Run unit tests, then the full pipeline on the real data. Check the figures and metrics.
8. Record the results and any open points below.

## 5. Out of scope / open points
- Old output folders (`figures_new_SOM_*`, `model_comparison_plots/`, `model_decision/`) came from the old pipeline. They were **removed** at the author's request (`git rm`), so they now exist only in git history. New outputs go to `results/`.
- The `.pkl` files stay in git **history**. Removing them from history needs a history rewrite and a force-push, which is the author's call.
- No subject IDs exist in the file names, so splits are by recording, not by person. If the same people appear in train and test, results may be optimistic.

## 6. Changes to the plan made during implementation

Found while checking the real data or measuring on validation data:

| Plan said | Done instead | Reason |
|---|---|---|
| Process each **file** in frame order | Process each **track** (export segment × person) | `person count` is a person index. All A5/A6 files have two people per frame (rows alternate), 3 files hold two joined exports, and some files have short ghost detections. The old code read these as one sequence |
| Missing = `p == 0` | Also flag joints whose bone is > 2× its median length, and split tracks where the pelvis jumps > 0.3 m in one frame | 33 frames had single joints 5–360 torso lengths away (glitches, re-acquisitions). Afterwards the largest joint distance is 3.2 torso lengths |
| Interpolate in camera coordinates | Interpolate **relative to the pelvis** (pelvis itself absolute), iterated up to 3× | A filled limb must move with the body; a whole glitched limb is only caught joint by joint |
| σ_end = 0.5 | **σ_end = 1.0** | Measured on validation (19×21 map): σ_end 0.5 → QE 0.83 / TE 0.34; 0.75 → 0.91 / 0.18; **1.0 → 0.98 / 0.13**. Two-level clustering and the U-matrix need topology; 1.0 is also the SOM Toolbox convention |
| `tol` 1e-4 | `tol` 1e-3 | QE is flat (< 0.2% change) once ΔW/W ≈ 1e-3 |
| BMU margin 1 − d₁/d₂ | **Cluster margin** 1 − d_own/d_other (nearest unit of another cluster) | With about 400 units, the first and second BMU are almost always in the same cluster, so the BMU margin says nothing about the cluster assignment |
| Review: "committed `venv/`" | No change needed | `venv/` was already git-ignored. It is only broken locally (it points to another user's Python). The review statement was wrong |
| About 2% of frames are lost | — | Fragments shorter than 30 frames after splitting at jumps are dropped (train 40,695 → 39,798; test 11,950 → 11,616) |

## 7. Results

Full run: `python som_pose_tracking.py --map-sweep` (seed 42, about 4 minutes). All outputs are in `results/`. 28 unit tests pass (`python -m unittest discover -s tests -t .`).

**Model**
- Map 19×21 (399 units, Vesanto heuristic with cap 400), batch SOM, converged at epoch 59 (ΔW/W 7.7e-4 < 1e-3). 16 units have no training hits.
- K chosen on validation data: **K = 3** (mean rank 1.33; K = 2 is close at 1.67, silhouette 0.356 vs 0.354). Silhouette falls sharply after K = 3. K = 9 is a small local optimum, available with `--k 9` if a finer posture vocabulary is wanted.

**Metrics with final weights**

| | fit | validation | test |
|---|---|---|---|
| QE | 0.878 | 0.977 | 1.008 |
| TE | 0.107 | 0.126 | 0.142 |
| Silhouette | 0.366 | 0.356 | 0.391 |
| Davies–Bouldin | 1.189 | 1.237 | 1.118 |
| ARI vs activity | 0.035 | 0.044 | 0.062 |
| NMI vs activity | 0.179 | 0.182 | 0.213 |
| Purity vs activity | 0.239 | 0.256 | 0.340 |
| Novelty rate | 1.0% | 1.4% | 1.4% |
| Frames in stable runs (≥ 10) | 99.4% | 98.9% | 99.9% |
| Mean run length | 69 frames | 64 frames | 85 frames |

**Interpretation**
- QE and TE on validation and test are close to the fit values, so the map generalises to unseen recordings. These are the honest held-out numbers the old code never produced.
- The three clusters are clear posture types (see `figures/cluster_postures.png`): **0 = seated** (thighs horizontal), **1 = leaning or bending forward**, **2 = standing** (72% of training frames).
- The clusters are **postures, not activities**. Activities share postures, which is why ARI is low. The contingency table explains it: A1 is 100% standing. A5 and A6, the two-person activities, split exactly 50/50 between seated and standing, so one person sits while the other stands. The old code merged those two people into one sequence.
- Assignments are temporally stable: labels change about 0.16 times per second on test data, and almost no frames sit in runs shorter than 10 frames. The old run-length filter is no longer needed to clean labels; it is kept only as a stability measure.
- Map-size sweep (validation): QE falls monotonically with size (1.21 → 0.91 from 100 to 598 units), as expected, while TE is lowest at 13×15 (0.104) and 0.126 at the used 19×21. This confirms that QE cannot be used to pick the map size.

**Open points for the author**
- The old output folders and the `.pkl` files in git history were left in place (§5).
- The 3-cluster solution is what the internal indices support. **Author decision (2026-10-19): σ_end = 1.0; K is not fixed** but always chosen from the data (validation indices), so it follows whatever the data support. `ClusteringConfig.n_clusters = None` is the default; `--k N` is only an override for experiments and is marked as such in the summary.
