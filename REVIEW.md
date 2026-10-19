# Code Review: SOM-based Human Posture Clustering

Reviewed branch: `feat/som_v2.0.0` (commit `7210c95`, with the one-line uncommitted comment fix in `som_pose_tracking.py`).
Files reviewed: `data_preprocessing.py`, `som_pose_tracking.py`, `pose_tracking.py`, the repo layout, the saved models and the generated figures.

> **Status (branch `feat/som_clustering_fix`):** every issue below has been addressed by a rewrite into the `som_clustering/` package. See [`plans/som-clustering-fix.md`](plans/som-clustering-fix.md) §3 for the issue-by-issue fixes, §6 for changes found during implementation (two-person files, joined exports, tracking glitches) and §7 for the new results.
> **Correction:** §2.3 and §5.5 say `venv/` is committed. It is not; it was already git-ignored. It only points to another user's Python, so it doesn't work on this machine.

---

## 1. Summary

The project clusters **3D human body postures** recorded by a skeleton-tracking camera, using a **Self-Organizing Map (SOM)**. It is unsupervised: every SOM neuron is treated as one "posture cluster".

The overall idea is sound. A small SOM is a reasonable way to quantise posture space into prototype postures, and the core Kohonen update rule is written correctly. But the pipeline has several bugs and method problems that **change the reported results**:

| # | Most important problems | Effect |
|---|---|---|
| 1 | "Testing" QE, TE and discriminant plots are drawn from **training** data and training history | All `testing_*.png` figures are training results under a test label |
| 2 | Model-selection QE/TE come from the **training** run; discriminant and gap scores come from the **test** set, normalised by **test-set** statistics | The "best model" is chosen from a mix of train and test metrics, and the test set is used for selection, so no data is held out |
| 3 | Test data is normalised with **its own** mean and range, not the training statistics, and as one block rather than per camera as in training | Train and test live in different feature spaces |
| 4 | Missing-joint marker `-1` is replaced **after** the coordinates are transformed and scaled, so it never matches | Missing joints are never handled |
| 5 | Learning rate and neighbourhood radius barely decay (they only fall to about 37% of their start values) | The map never settles (QE is still noisy at epoch 100) and prototypes stay smoothed toward their neighbours |
| 6 | The number of SOM neurons is used as the number of clusters, with only 6–20 neurons | The SOM's topology adds little at this size, so it acts mostly like a noisy online k-means |
| 7 | "Discriminant score" is normalised by the **largest distance to any neuron**, not the largest BMU distance (the docstring says BMU) | Scores are squeezed toward 1, and the name is misleading |

Details, with file and line references, follow.

---

## 2. What the codebase does

### 2.1 Data

- `training_data/` holds 140 CSV files and `test_data/` holds 40. File names look like `A{activity}s{session}c{camera}-w.csv`, for example `A1s1c1-w.csv`. There are 9 activities (A1–A9) and 2 cameras (c1, c2). Train and test are split by session, and no file appears in both. That split is a good choice.
- Each row is one video frame: `Frame number, person count`, then for each of 34 joints `x, y, z, p` (coordinates in metres plus a confidence value `p`). That makes 138 columns. A 128-column variant (32 joints, no frame columns, called "Waseda format" in the code) is also supported.
- 15 joints (hand tips, thumbs, feet, face points and so on) are dropped. That leaves **19 joints × 3 = 57 features** per posture.

### 2.2 Pipeline

```
CSV files ──► data_preprocessing.py
               read_data()        concatenate all CSVs in a folder
               load_training_data()  per camera (c1, c2):
                   drop columns → to_numeric → dropna
                   → transform_data(): ×1000 (m→mm), subtract joint 1 (torso) from every joint
                   → replace(-1, 0) → per-feature (x-mean)/(max-min)
               load_test_data()   same steps, all cameras together, NaNs imputed

          ──► som_pose_tracking.py  (class CompetitiveLearning)
               for K in 6, 8, …, 20 neurons:
                   init weights ~ N(mean, std) on a rows×cols grid
                   train(): online Kohonen SOM, cosine distance, Gaussian neighbourhood,
                            per-epoch exponential decay of lr and radius, early stopping
                   save the whole object as a pickle → trained_som_models/SOM_K_cls.pkl
               for each saved model:
                   classify test data (BMU / 2nd BMU), filter BMU runs shorter than 10 frames (plot only)
                   discriminant score = 1 - d_BMU / max_d
                   gap score          = (d_2nd - d_BMU) / max_gap
                   per-sample QE and TE
               compare models → plots in model_comparison_plots/, model_decision/
               choose_best_som_model(): equal-weight sum of min-max-normalised metrics

          ──► pose_tracking.py  (standalone) animates one CSV as a 3D skeleton
```

### 2.3 Architecture

- **One large file** (`som_pose_tracking.py`, about 2,465 lines) holds one class, `CompetitiveLearning`, that does everything: model, training, metrics, evaluation, model selection, persistence and 15+ plotting routines. Plotting code is roughly 70% of the class.
- `data_preprocessing.py` contains two near-copies of the same pipeline (train and test). They have **drifted apart** (see §5.1), and the dropped-column lists are pasted four times.
- `pose_tracking.py` is a separate visualisation script. It does not share code with the rest and points to a folder that does not exist.
- Persistence pickles the whole `__dict__`, including the full training set. Each model file is **about 88 MB**, and all 8 are committed to git (about 700 MB in history).
- There is no `requirements.txt`, no random seed, no tests, no CLI or config file (hyper-parameters are hard-coded in `__main__`), and the README is one line. The committed `venv/` points to another user's Python (`C:\Users\alicj\...`), so it will not run on this machine.

**Recommended structure:**

```
som/model.py        # SOM class: init, train, bmu, quantization/topographic error (vectorised)
som/metrics.py      # discriminant/gap scores, cluster-validity indices
som/selection.py    # model selection
data/preprocess.py  # one pipeline, fit on train, apply to test
plots/*.py          # all plotting
scripts/train.py, scripts/evaluate.py
```

---

## 3. Check against the Self-Organizing Map concept

Reference: Kohonen, *Self-Organizing Maps* (2001); Vesanto & Alhoniemi, *Clustering of the SOM* (IEEE TNN 2000).

| SOM element | Standard practice | This code | Verdict |
|---|---|---|---|
| Lattice | 2D rectangular or hexagonal grid, fully populated | `compute_grid_shape` gives 2×3, 2×4, **3×4 for 10**, 3×4, **3×5 for 14**, 4×4, **4×5 for 18**, 4×5 | ⚠ 10, 14 and 18 neurons give **ragged grids** with a partly empty last row. That distorts the neighbourhood and the TE |
| Initialisation | Random samples, or linear init along the first 2 principal components | `np.random.normal(mean(all), std(all))`: one scalar mean and std for all 57 features | ⚠ Works, but random and unseeded. PCA init is standard and reproducible |
| BMU search | argmin of a distance, usually Euclidean | argmin of **cosine** distance | ⚠ See the next row |
| Update rule | `w_i += α(t)·h_ci(t)·(x − w_i)` | Same (`update_weights`, l.161) | ✅ Correct |
| Neighbourhood | Gaussian over **grid** distance | Gaussian over grid distance (l.179–182) | ✅ Correct |
| Distance vs update | The update rule minimises **Euclidean** distortion. A cosine/dot-product SOM normalises inputs and weights to unit length | Cosine BMU with an un-normalised Euclidean update | ❌ Mismatch. Weight norms are irrelevant to the BMU but still updated. Cosine also ignores magnitude: on mean-centred data, a slight lean and a strong lean in the same direction look **identical** |
| Learning-rate schedule | Decays to a small value (about 0.01 or less) in a "convergence phase" | `0.2·exp(-epoch/E)`: ends at **0.074** | ❌ Too high at the end. Online updates keep jittering the map (training QE is still noisy at epoch 100, see `figures_new_SOM_10/training_quantization_error.png`) |
| Radius schedule | Ordering phase (large σ), then convergence phase with σ → about 0.5–1 so each neuron fits its own data | `σ0·exp(-epoch/E)` with σ0 = max(rows,cols)/2. For 2×3 this ends at σ ≈ 0.55, where a direct neighbour still gets weight `exp(-1/(2·0.55²)) ≈ 0.19` | ❌ Neighbours keep pulling each other until the end, which biases prototypes toward each other and raises QE |
| Decay granularity | Usually per sample step | Per epoch | Minor. Acceptable |
| Convergence | Fixed schedule, or early stopping on QE | Relative weight change < 1e-3 for 10 epochs | ⚠ With lr ≥ 0.074 this almost never triggers. The saved SOM-10 run used all 100 epochs |
| Quantization error | Mean distance from samples to their BMU | Training: mean **squared** cosine distance. Test: **unsquared** cosine distance | ❌ Two different definitions under one name. Neither is the usual Euclidean QE |
| Topographic error | Share of samples whose 1st and 2nd BMU are not adjacent | 4-neighbour adjacency on a rectangular grid | ✅ Correct. Note that TE is barely meaningful on 6–20 neuron maps |
| Map size | Rule of thumb about 5·√N neurons (hundreds for this data size) | 6–20 | ⚠ At this size the map is a **quantiser**, not a topology-preserving map |

**Conclusion on the SOM:** the core update and neighbourhood are correct. The training schedule, the cosine/Euclidean mismatch and the very small, sometimes ragged maps mean the trained models are only weakly self-organised. Each one behaves mostly like an online k-means with K = number of neurons plus extra smoothing.

---

## 4. Check against the clustering concept

1. **Neurons treated as clusters.** The standard SOM approach to clustering is two-level: train a fairly large map (fine resolution, good topology), then cluster the **prototypes** (U-matrix, hierarchical or k-means on the codebook) into the final groups (Vesanto & Alhoniemi 2000). This code ties map resolution and cluster count together, so "choosing the number of neurons" and "choosing the number of posture clusters" become the same decision. That is the main reason QE always falls as K grows.
2. **No standard cluster-validity measures.** Silhouette, Davies–Bouldin, Calinski–Harabasz and similar are not computed. The home-made "discriminant" and "gap" scores are not standard and are not validated against anything.
3. **Labels are available but unused.** File names encode the activity (A1–A9). External validation (purity, ARI, NMI, or a contingency table of BMU vs activity) would show whether the clusters mean anything. Right now there is **no evidence that the clusters match postures or activities**.
4. **"Discriminant score" is misnamed.** It is not a discriminant in the Fisher/LDA sense. It is `1 − d_BMU / max_distance`, a normalised closeness. The gap score `(d₂ − d₁)/max_gap` is a reasonable margin-style confidence. A scale-free form such as `1 − d₁/d₂` would compare better across models.
5. **Confidence normalised on the evaluation set.** Both scores divide by maxima taken over the test samples (`initialize_distance_normalization_constants`). Scores therefore depend on the batch being scored, and one outlier rescales everything.
6. **Temporal filtering.** `filter_short_bmu_runs` (drop BMU runs under 10 frames) is a sensible smoothing idea. But (a) it only affects one plot, not any metric, and (b) the test set is **all 40 files glued together**, so runs cross file boundaries. `glob` order is also not guaranteed to be sorted.
7. **Model selection by combined score** (`choose_best_som_model`): equal-weight sum of min-max-normalised QE, TE, mean/std discriminant and mean/std gap.
   - QE falls almost monotonically with K by construction, which biases the choice toward larger maps.
   - Min-max over only 8 models stretches tiny differences to the full 0–1 range.
   - The terms are correlated (discriminant and QE both come from `d_BMU`), so the same thing is counted twice.
   - It mixes training metrics (QE, TE) with test metrics (discriminant, gap). See §5.2.

---

## 5. Errors and mistakes

Severity: 🔴 Critical (wrong results), 🟠 Major (method or design flaw, or likely to break), 🟡 Minor (quality, performance, clean-up).

### 5.1 `data_preprocessing.py`

| Sev | Location | Problem |
|---|---|---|
| 🔴 | l.262, l.543 | `data.replace(-1, 0)` runs **after** `transform_data`, which scales by 1000 and subtracts the reference joint. A raw `-1` marker becomes something like `-1000 − ref` and never equals `-1` again, so missing joints are **never** handled. Also, zeroing a raw coordinate before taking relative coordinates would itself give a fake position. Missing joints should be masked or imputed **before** the transform |
| 🔴 | l.264–270 vs l.550–555 | Test data is normalised with **its own** mean, min and max. It should use the statistics **fitted on training data**. Right now any train/test shift is hidden, and results change with the size and contents of the test batch |
| 🔴 | l.116–276 vs l.314 | Training normalises **each camera separately** (c1 and c2 each get their own mean and range). Test normalises **c1 and c2 together**. The same posture lands at different coordinates in train and test |
| 🟠 | l.239–257 vs l.521–538 | NaN handling differs. Training runs `dropna()` before the transform, which makes the following `fillna`/`ffill`/`bfill` code unreachable. Test **imputes** with column means before the transform. Means of absolute coordinates across different files and people are not real postures |
| 🟠 | l.264–270 | Per-feature `(x−mean)/(max−min)` scales x, y and z of every joint independently. That **breaks skeleton geometry** (bone lengths and angles are no longer consistent), and it is the reason `plot_posture_match` cannot draw a realistic skeleton. Consider one global scale, or normalising by body size (for example torso length) |
| 🟠 | l.60 | The reference point is joint 1 (`posture_vector[3:6]`, the stomach/spine joint according to `pose_tracking.py`), not joint 0 (pelvis). That is acceptable if intended, but joint 1 then becomes 3 constant-zero features (0/eps → 0) that add nothing |
| 🟠 | l.530, l.533 | `fillna(method="ffill"/"bfill")` is deprecated in pandas 2.x (FutureWarning on the installed 2.3.3) and removed in pandas 3.0. Use `.ffill()` and `.bfill()` as the training path already does |
| 🟠 | l.30–32 | `pd.concat` of files with different column layouts (138 vs 128) produces the union of columns plus NaNs. The format check only looks at the combined width, so a folder with mixed formats fails or silently drops everything at `dropna` |
| 🟠 | l.185–231 | The 128-column (Waseda) branch differs between train and test (frame/person-count drop, list of dropped `p` columns). It works only because of `errors="ignore"` |
| 🟡 | l.56 | The ×1000 (m→mm) scaling has no effect, since min-max normalisation follows |
| 🟡 | l.59–69 | `transform_data` loops over every row in Python. It could be one vectorised `reshape(-1, J, 3) - ref` |
| 🟡 | whole file | Column-drop lists are pasted 4 times, and train/test are two diverging copies of one pipeline. Use one `preprocess(df, stats=None)` that returns `stats` when fitting |
| 🟡 | l.23 | `glob` order is OS-dependent. Use `sorted(...)` so the temporal order of the test sequence is reproducible |

### 5.2 `som_pose_tracking.py` — evaluation and model selection

| Sev | Location | Problem |
|---|---|---|
| 🔴 | l.2332–2334 | `plot_quantization_error(is_training=False)`, `plot_topographic_error(False)` and `plot_discriminant_scores(False)` plot `self.error`, `self.topographic_error`, `self.epoch_weights` and `self.input_data` loaded from the training pickle. **Every `testing_*.png` is the training curve with a "Testing" label.** Confirmed: `figures_new_SOM_10/testing_quantization_error.png` matches `training_quantization_error.png` point for point. "Testing epoch" is also not a real concept, since testing runs once |
| 🔴 | l.2357–2358 | `final_qe = loaded_som.error[-1]` and `final_te = loaded_som.topographic_error[-1]` are the **last training-epoch** values. The per-sample test QE/TE computed a few lines above (l.2311–2323) are only used for plots |
| 🔴 | l.2044–2120 + main | Model selection uses the **test set** (discriminant and gap scores), so there is no held-out data left to report final performance. Use a validation split for selection and keep the test set for the final report |
| 🔴 | l.1027–1044 | `get_max_bmu_distance` takes `max(neuron_distances)`, the farthest neuron, not the BMU distance its name and docstring describe. The discriminant score is therefore `1 − d_BMU/max_any`, which squeezes values toward 1 |
| 🟠 | l.970–992 | Normalisation constants are computed **on the test samples being scored**, and the method **re-opens the pickle from disk** to read weights already in `self.neuron_weights` |
| 🟠 | l.1729–1738 vs l.1001–1006 | Two different "discriminant score" definitions. Per epoch it divides by the max BMU distance. At test time it divides by the max distance to any neuron. They are not comparable |
| 🟠 | l.342–347 vs l.2311 | QE is the mean **squared** distance in training but the **unsquared** distance at test time |
| 🟠 | l.1839–1845 | The debug print shows `neurons` instead of `te_values`, `mean_ds` and so on. The output recorded at the end of the file (l.2455–2465, `te [6.0, 8.0, …]`) is therefore wrong, so don't quote it |
| 🟠 | l.1109, l.1117, l.1189, l.1207 | `cluster_colors[cluster_id % len(cluster_colors)]` indexes a **dict keyed by cluster id** with a modulo. If the active ids are not `0..n−1` this gives the wrong colour or a **KeyError** (for example active ids {1,2,3}: `3 % 3 = 0`, which is not a key). Use `cluster_colors[cluster_id]` |
| 🟠 | l.2326–2329 | `bmu_sequences_by_model[...]` and `get_cluster_colors(bmu_sequence)` sit **inside the per-sample loop**, so they are recomputed once per test sample. Move them out of the loop |
| 🟡 | l.1950, l.1962 | The y-label says "± Std" but `yerr` is never passed |
| 🟡 | l.1970–1980 | The combined plot overlays QE, TE, scores and stds on one axis even though they have very different scales |
| 🟡 | l.1565 | The QE plot title always says "Training", even for "testing" |
| 🟡 | l.1535, l.2168 | `np.convolve(..., mode="same")` pads with zeros, so smoothed curves drop artificially at both ends (visible in every QE plot) |

### 5.3 `som_pose_tracking.py` — SOM model and training

| Sev | Location | Problem |
|---|---|---|
| 🟠 | l.236–243 | lr and radius only decay to e⁻¹ (37%) of their initial values, so the map never reaches a convergence phase (see §3) |
| 🟠 | l.115, l.122 | Cosine BMU combined with a Euclidean update (§3). `"hamming"` is offered but is meaningless for continuous coordinates. The comment says "default to Euclidean" but the code defaults to **cosine** |
| 🟠 | l.21–24 | Ragged grids for 10, 14 and 18 neurons |
| 🟠 | l.42–46 | No `np.random.seed` / `default_rng(seed)` anywhere (init, shuffle, subsampling), so runs **cannot be reproduced** |
| 🟠 | l.2253–2264 | In the test phase, `CompetitiveLearning(...)` reloads and preprocesses all 140 training files for **each** of 8 models, then `load()` throws that away |
| 🟠 | l.2152 | `pickle.dump(self.__dict__)` stores the training data, all epoch weights and `average_distance` (epochs × N floats): about 88 MB per model, all committed to git (GitHub warns above 50 MB and rejects above 100 MB). Save only `neuron_weights`, `grid_shape`, hyper-parameters and metric history (for example `np.savez`). Unpickling untrusted files is also unsafe |
| 🟠 | l.290–301 | `train()` saves `training_error.png` twice. The second inline block overwrites the nicer plot and never closes its figure (memory leak across 8 models) |
| 🟠 | l.1225, l.1259 | `plot_weight_changes` draws an **absolute** threshold of 0.05, while convergence uses a **relative** threshold of 1e-3. The plotted line has nothing to do with the stopping rule |
| 🟡 | l.125–128, l.177–188 | Distances and updates are pure Python loops, run about 3 times per epoch (train, QE, TE) and again for every stored epoch in `plot_discriminant_scores`. `scipy.spatial.distance.cdist` plus a vectorised neighbourhood would be 100× or more faster |
| 🟡 | l.132–133 | `find_winner_neuron` appends to `average_distance` for every sample and epoch. The list is never used and bloats the pickle |
| 🟡 | l.255–263 | `winners_list` (used by `plot_train`) is collected **during** the last epoch while weights are still moving, so the labels are not those of the final map |
| 🟡 | — | Dead code: `potential`, `activation`, `gaussian` (stored, never read; the update is always Gaussian), `kohonen_neighborhood`, `find_two_bmus`, `classify_with_matches`, `all_steps`, `Line2D` import, an inner `import numpy as np` (l.905). PCA of weight history is computed twice (l.590–603 and l.644–650) |
| 🟡 | l.1147–1189 | `plot_discriminant_subplots` recomputes all distances K times (O(K·N·K)). Compute the distance matrix once |
| 🟡 | l.60–74 | `joint_connections` assumes a 14-joint skeleton with the arms at index 0. The data has 19 joints and index 0 is the pelvis (`pose_tracking.py`), so (0,1),(1,2),(2,3), labelled "right arm", is actually the spine, and joints 14–18 are never drawn |
| 🟡 | l.884–887 | `plot_posture_match` names its file by neuron count only, so different `sample_index` values overwrite each other. The call is also commented out (l.2394) |
| 🟡 | l.1671, l.1756, l.1784 | f-strings with nested double quotes need Python ≥ 3.12. That works on the venv's 3.13, but fails on older interpreters |
| 🟡 | naming | The class is called `CompetitiveLearning` and its method `classify`, but this is an unsupervised SOM doing BMU assignment, not classification |

### 5.4 `pose_tracking.py`

| Sev | Location | Problem |
|---|---|---|
| 🟠 | l.16 | Reads `test/A1s1c1-w.csv`, but the folder is `test_data/`, so the script crashes as committed |
| 🟡 | l.121–145 | Writes the DataFrame to `CSV_DATA.csv` then reads it back with `csv.reader`. `data.values` would do. It also leaves a stray file in the repo |
| 🟡 | l.1, l.8 | `import pdb` twice. Unused imports (`axes3d`, `os`) |
| 🟡 | — | No y-flip for the 128-column format, unlike the preprocessing |

### 5.5 Repository and process

- 🟠 About 700 MB of pickles are committed. Use `.gitignore` or Git LFS, and save compact models.
- 🟠 No `requirements.txt` or `pyproject.toml`. The `venv/` is tied to another user's machine.
- 🟡 The README is a single heading. It should cover data format, how to run, hyper-parameters and output folders.
- 🟡 Output folders are named `figures_new_SOM_*`. "new" will stop meaning anything over time.
- 🟡 No tests. At minimum: preprocessing shapes, `index_to_2d`/`are_neighbours`, and a SOM on toy 2D data where clusters are known.

---

## 6. Recommended fixes, in priority order

1. **Fix the evaluation data flow.** Compute test QE, TE, discriminant and gap scores **on test data with the final weights**, store them in `results`, and delete or rename the "testing epoch" plots.
2. **One preprocessing pipeline, fitted on training data.** Fit mean/range (or a body-size normaliser) on training data, save it with the model, and apply it to test. Handle missing joints **before** the relative transform. Handle cameras the same way in train and test.
3. **Split train / validation / test.** Select K on validation and report on test. Use the activity labels for external validation (ARI, NMI, purity, BMU × activity heat-map).
4. **Fix the SOM schedule.** For example, `lr: 0.5 → 0.01` and `σ: max(rows,cols)/2 → 0.5` over all steps (linear or `exp(-t/τ)` with τ = T/ln(σ0/σ_end)). Seed the RNG and use PCA initialisation.
5. **Pick one geometry.** Either Euclidean distance (consistent with the update rule) or unit-normalised inputs and weights for a true cosine SOM. Use one QE definition everywhere.
6. **Consider two-level clustering.** Train a larger map (for example 10×10), then cluster the prototypes with Ward or k-means and pick the cluster count with silhouette or Davies–Bouldin. Show the U-matrix.
7. **Fix the discriminant score.** Fix `get_max_bmu_distance`, take normalisation constants from training or validation data, and rename the score to something like "BMU confidence".
8. **Vectorise** distances and updates with `cdist` and NumPy broadcasting.
9. **Clean up the repo.** Remove dead code, split into modules, save compact models, add `requirements.txt`, a README and basic tests.
