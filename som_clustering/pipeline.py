"""End-to-end run: fit the SOM, choose K on validation data, report on test data.

Data roles:
* fit (80% of training recordings): trains the SOM and the prototype clustering
  and sets the confidence reference distances;
* validation (20% of training recordings): chooses K;
* test: final report only. Nothing is fitted or chosen on it.
"""

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import plots
from .clustering import (
    PrototypeClusterer,
    bmu_confidence,
    cluster_margin,
    contingency_table,
    external_indices,
    fit_confidence_reference,
    internal_indices,
    select_k,
)
from .config import RESULTS_DIR, TEST_DIR, TRAINING_DIR, PipelineConfig
from .data import features_to_joints, load_dataset, split_by_recording
from .som import SelfOrganizingMap, heuristic_map_shape
from .temporal import stability_summary, stable_mask

N_EXAMPLE_TRACKS = 4


def run(config=None, out_dir=RESULTS_DIR, map_sweep=False):
    config = config or PipelineConfig()
    out_dir = Path(out_dir)
    figures = out_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    print("Loading data")
    train = load_dataset(TRAINING_DIR, config.preprocess)
    test = load_dataset(TEST_DIR, config.preprocess)
    fit, val = split_by_recording(train, config.validation_fraction, config.seed)
    datasets = {"fit": fit, "validation": val, "test": test}
    for name, data in datasets.items():
        print(f"  {name:<10} {len(data):6d} frames, {data.meta['track'].nunique():3d} tracks, "
              f"activities {sorted(int(a) for a in data.meta['activity'].unique())}")

    # ------------------------------------------------------------------
    # SOM (level 1)
    # ------------------------------------------------------------------
    if config.som.rows and config.som.cols:
        rows, cols = config.som.rows, config.som.cols
    else:
        rows, cols = heuristic_map_shape(fit.X, config.som.max_units)

    sweep_table = None
    if map_sweep:
        sweep_table = run_map_sweep(fit.X, val.X, rows, cols, config)
        sweep_table.to_csv(out_dir / "map_size_sweep.csv", index=False)
        plots.plot_map_size_sweep(sweep_table, rows * cols, figures / "map_size_sweep.png")

    print(f"Training {config.som.algorithm} SOM {rows}x{cols}")
    som = SelfOrganizingMap(rows, cols, config.som, seed=config.seed).fit(fit.X, val.X)
    plots.plot_training_history(som.history, config.som.tol, config.som.n_epochs,
                                figures / "training_history.png")

    fit_bmu, fit_distances = som.bmu(fit.X, k=2)
    reference = fit_confidence_reference(fit_distances[:, 0])
    hits_fit = np.bincount(fit_bmu[:, 0], minlength=som.n_units)

    # ------------------------------------------------------------------
    # Prototype clustering (level 2), K chosen on validation data
    # ------------------------------------------------------------------
    print("Choosing K on validation data")
    clusterer = PrototypeClusterer(som.weights, hits_fit, config.clustering.method, config.seed)
    val_bmu = som.bmu(val.X)[0][:, 0]
    rows_k = []
    for k in range(config.clustering.k_min, config.clustering.k_max + 1):
        val_labels = clusterer.unit_labels(k)[val_bmu]
        indices = internal_indices(val.X, val_labels, config.clustering.silhouette_sample, config.seed)
        rows_k.append({"k": k, **indices})
    index_k, k_table = select_k(pd.DataFrame(rows_k))
    fixed_k = config.clustering.n_clusters
    chosen_k = index_k if fixed_k is None else fixed_k
    print(f"  chosen K = {chosen_k}" + ("" if fixed_k is None else f" (set by the user; index choice K = {index_k})"))
    k_table.to_csv(out_dir / "k_selection.csv", index=False)
    plots.plot_k_selection(k_table, chosen_k, figures / "k_selection.png")

    unit_labels = clusterer.unit_labels(chosen_k)

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------
    print("Evaluating")
    evaluations = {
        name: evaluate(som, unit_labels, reference, data, config)
        for name, data in datasets.items()
    }
    metrics = {name: result["metrics"] for name, result in evaluations.items()}

    som.save(
        out_dir / "som_model.npz",
        unit_labels=unit_labels,
        hits_fit=hits_fit,
        confidence_reference=reference,
        pipeline_config=config.to_dict(),
    )

    # ------------------------------------------------------------------
    # Figures
    # ------------------------------------------------------------------
    print("Plotting")
    plots.plot_u_matrix(som.u_matrix(), unit_labels, figures / "u_matrix_clusters.png")
    plots.plot_hit_maps(
        {name: np.bincount(result["bmu"], minlength=som.n_units) for name, result in evaluations.items()},
        unit_labels, rows, cols, figures / "hit_maps.png",
    )
    for name in ("fit", "test"):
        plots.plot_clusters_pca(datasets[name].X, evaluations[name]["labels"], fit.X,
                                figures / f"clusters_pca_{name}.png",
                                title=f"Clusters on {name} data", seed=config.seed)
    for name in ("validation", "test"):
        table = contingency_table(datasets[name].labels, evaluations[name]["labels"])
        table.to_csv(out_dir / f"contingency_{name}.csv")
        plots.plot_contingency(table, figures / f"contingency_{name}.png",
                               title=f"Clusters vs activities ({name} data)")

    fit_labels = evaluations["fit"]["labels"]
    cluster_joints = {c: features_to_joints(fit.X[fit_labels == c].mean(axis=0))[0]
                      for c in np.unique(fit_labels)}
    cluster_shares = {c: float((fit_labels == c).mean()) for c in cluster_joints}
    plots.plot_cluster_postures(cluster_joints, cluster_shares, figures / "cluster_postures.png")

    plots.plot_track_timelines(example_tracks(test, evaluations["test"]),
                               config.clustering.min_run_length, chosen_k, figures / "test_track_timelines.png")
    plots.plot_score_distributions(
        {name: {"confidence": result["confidence"], "margin": result["margin"]}
         for name, result in evaluations.items()},
        figures / "score_distributions.png",
    )

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------
    report = {
        "config": config.to_dict(),
        "map": {"rows": rows, "cols": cols, "units": rows * cols,
                "epochs_run": len(som.history), "dead_units_fit": int((hits_fit == 0).sum())},
        "chosen_k": chosen_k,
        "index_k": index_k,
        "k_fixed": fixed_k is not None,
        "confidence_reference": reference,
        "data": {name: {"frames": len(data), "tracks": int(data.meta["track"].nunique()),
                        "recordings": int(data.meta["recording"].nunique())}
                 for name, data in datasets.items()},
        "metrics": metrics,
        "runtime_seconds": round(time.time() - started, 1),
    }
    (out_dir / "metrics.json").write_text(json.dumps(_jsonable(report), indent=2), encoding="utf-8")
    (out_dir / "summary.md").write_text(summary_markdown(report, k_table, sweep_table), encoding="utf-8")
    print(f"Done in {report['runtime_seconds']} s. Results in {out_dir}")
    return report


def evaluate(som, unit_labels, reference, data, config):
    """All metrics for one data set, using the final weights."""
    bmu, distances = som.bmu(data.X, k=2)
    labels = unit_labels[bmu[:, 0]]
    confidence = bmu_confidence(distances[:, 0], reference["reference_distance"])
    margin = cluster_margin(data.X, som.weights, unit_labels, bmu[:, 0])
    stable = stable_mask(labels, data.tracks, config.clustering.min_run_length)

    metrics = {
        "quantization_error": float(distances[:, 0].mean()),
        "topographic_error": float(np.mean(~som.adjacent[bmu[:, 0], bmu[:, 1]])),
        "clusters_present": int(len(np.unique(labels))),
        **internal_indices(data.X, labels, config.clustering.silhouette_sample, config.seed),
        "external": external_indices(data.labels, labels),
        "external_stable_frames_only": (
            external_indices(data.labels[stable], labels[stable]) if stable.any() else None
        ),
        "mean_bmu_confidence": float(confidence.mean()),
        "mean_cluster_margin": float(margin.mean()),
        "novelty_rate": float((distances[:, 0] > reference["novelty_distance"]).mean()),
        "temporal": stability_summary(labels, data.tracks, config.clustering.min_run_length),
    }
    return {"metrics": metrics, "labels": labels, "bmu": bmu[:, 0],
            "confidence": confidence, "margin": margin, "stable": stable}


def run_map_sweep(X_fit, X_val, rows, cols, config, scales=(0.25, 0.5, 1.0, 1.5)):
    """Validation QE/TE for maps of different sizes with the same side ratio."""
    print("Map-size sweep (sensitivity only)")
    records = []
    for scale in scales:
        r = max(2, int(round(rows * np.sqrt(scale))))
        c = max(2, int(round(cols * np.sqrt(scale))))
        som = SelfOrganizingMap(r, c, config.som, seed=config.seed).fit(X_fit, X_val, verbose=False)
        last = som.history[-1]
        records.append({"rows": r, "cols": c, "units": r * c,
                        "val_qe": last["val_qe"], "val_te": last["val_te"]})
        print(f"  {r}x{c}: val QE {last['val_qe']:.4f}, val TE {last['val_te']:.4f}")
    return pd.DataFrame(records)


def example_tracks(data, result):
    """The longest test track of each of the first few activities."""
    meta = data.meta.assign(_row=np.arange(len(data)))
    lengths = meta.groupby(["activity", "track"]).size().reset_index(name="n")
    chosen = lengths.sort_values(["activity", "n"], ascending=[True, False]).groupby("activity").head(1)
    tracks = []
    for _, row in chosen.head(N_EXAMPLE_TRACKS).iterrows():
        index = meta.loc[meta["track"] == row["track"], "_row"].to_numpy()
        tracks.append({
            "name": f"A{row['activity']}: {row['track']}",
            "labels": result["labels"][index],
            "confidence": result["confidence"][index],
            "margin": result["margin"][index],
            "stable": result["stable"][index],
        })
    return tracks


def summary_markdown(report, k_table, sweep_table):
    m = report["metrics"]
    names = list(m)
    lines = [
        "# SOM posture clustering — results",
        "",
        f"- Map: {report['map']['rows']}×{report['map']['cols']} units, "
        f"{report['map']['epochs_run']} epochs, {report['map']['dead_units_fit']} units without training hits",
        (f"- Number of clusters: **K = {report['chosen_k']}**, chosen on validation data by the mean rank of "
         f"silhouette / Davies–Bouldin / Calinski–Harabasz"
         if not report["k_fixed"] else
         f"- Number of clusters: **K = {report['chosen_k']}**, set by the user "
         f"(the validation indices chose K = {report['index_k']})"),
        f"- Confidence reference distances (fit data): 95th pct "
        f"{report['confidence_reference']['reference_distance']:.3f}, "
        f"99th pct {report['confidence_reference']['novelty_distance']:.3f}",
        "",
        "## Data",
        "",
        "| set | frames | tracks | recordings |",
        "|---|---|---|---|",
        *[f"| {n} | {d['frames']} | {d['tracks']} | {d['recordings']} |" for n, d in report["data"].items()],
        "",
        "## Metrics (final weights)",
        "",
        "| metric | " + " | ".join(names) + " |",
        "|---|" + "---|" * len(names),
    ]
    rows = [
        ("Quantization error", lambda x: x["quantization_error"]),
        ("Topographic error", lambda x: x["topographic_error"]),
        ("Silhouette", lambda x: x["silhouette"]),
        ("Davies–Bouldin", lambda x: x["davies_bouldin"]),
        ("Calinski–Harabasz", lambda x: x["calinski_harabasz"]),
        ("ARI vs activity", lambda x: x["external"]["adjusted_rand"]),
        ("NMI vs activity", lambda x: x["external"]["normalized_mutual_info"]),
        ("Purity vs activity", lambda x: x["external"]["purity"]),
        ("Mean BMU confidence", lambda x: x["mean_bmu_confidence"]),
        ("Mean cluster margin", lambda x: x["mean_cluster_margin"]),
        ("Novelty rate", lambda x: x["novelty_rate"]),
        ("Stable-frame share", lambda x: x["temporal"]["stable_fraction"]),
        ("Mean run length (frames)", lambda x: x["temporal"]["mean_run_length_frames"]),
        ("Cluster switches / s", lambda x: x["temporal"]["switches_per_second"]),
    ]
    for title, getter in rows:
        lines.append(f"| {title} | " + " | ".join(f"{getter(m[n]):.3f}" for n in names) + " |")

    lines += ["", "## K selection (validation)", "", _table_markdown(k_table.round(4))]
    if sweep_table is not None:
        lines += ["", "## Map-size sensitivity (validation)", "", _table_markdown(sweep_table.round(4))]
    lines += ["", "Figures are in `figures/`. Activity labels were used only for the external metrics.", ""]
    return "\n".join(lines)


def _table_markdown(frame):
    header = "| " + " | ".join(frame.columns) + " |"
    divider = "|" + "---|" * len(frame.columns)
    body = ["| " + " | ".join(str(v) for v in row) + " |" for row in frame.itertuples(index=False)]
    return "\n".join([header, divider, *body])


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value
