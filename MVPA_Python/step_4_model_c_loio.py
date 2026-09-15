"""
================================================================================
Model C: leave-one-image-out (LOIO) -- is the reward code abstract across images?
ISSC_wCue
================================================================================

THE QUESTION
------------
Each participant sees four distinct images per reward condition (drawn fresh per
participant). Every prior model trains and tests on the SAME four swCond / rpCond
images, so an above-chance decode could in principle exploit image-specific
(visual) features rather than the learned reward meaning. Model C removes that
possibility by generalizing ACROSS images:

    train the rpCond-vs-swCond decoder on 3 of the 4 images per condition,
    test on the HELD-OUT image (one per condition).

If decoding of a never-trained image is above chance, the representation is
ABSTRACT across items (about the learned reward association), not tied to
specific pixels. This is the strongest visual-confound control and the capstone
of the item-level argument: A-neutral showed the effect is not block context;
Model C shows it is not image identity either.

SCHEME: 16-combination crossed leave-one-image-out
--------------------------------------------------
The four rpCond images and four swCond images are not matched pairs, so we do NOT
arbitrarily pair them. Instead we cross every held-out swCond image with every
held-out rpCond image (4 x 4 = 16 folds). Each fold:
    test  = all trials of held-out swCond image i + held-out rpCond image j
    train = the remaining 3 swCond images + remaining 3 rpCond images
Crossing all 16 combinations (rather than 4 arbitrary pairs) uses all held-out
combinations, adds averaging (lower-variance subject estimate), and removes any
artifact of a specific pairing.

IMPORTANT (fold non-independence): the 16 folds reuse trials in overlapping ways
(each image is the test image in 4 folds). This is fine for a stable POINT
estimate -- we average the 16 fold curves into one curve per subject, and all
inference is at the GROUP level across subjects (each subject contributes one
averaged curve). The 16 per-fold scores are NOT treated as independent samples.

DESIGN NOTE (validation scheme)
-------------------------------
Like A-neutral and B, this is a generalization test: the held-out image is never
in training, so no k-fold CV is layered on top -- the image hold-out IS the
train/test separation. Pseudotrial repeats (N_REPEATS) and the group cluster
permutation test are retained.

OUTPUTS (same philosophy as model_a_neutral / model_b -- save raw, aggregate later)
  scores_full/<sid>.npy   (n_repeats, n_folds_valid, n_times) per-fold AUC per repeat
  preds/<sid>.npz         held-out-image decision scores + labels + fold index per timepoint
  group_loio_scores.npy   (n_subjects, n_times)
  subject_ids.npy, times.npy, provenance.json
  decode_model_c_N*.png   group curve with significant cluster shaded
================================================================================
"""

from pathlib import Path
import json, platform, itertools
from datetime import datetime, timezone
import numpy as np
import mne, sklearn
from mne.decoding import SlidingEstimator
from mne.stats import permutation_cluster_1samp_test
from sklearn.base import clone
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm

from mvpa_io import load_subject, discover_subjects, ROOT

# ---- output ----
OUT_DIR    = ROOT / "mvpa_results" / "model_c_loio"
SCORES_DIR = OUT_DIR / "scores_full"
PREDS_DIR  = OUT_DIR / "preds"
for d in (OUT_DIR, SCORES_DIR, PREDS_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ---- config (mirrors model_a_neutral / model_b) ----
RESUME            = True
KEEP_SUBJECTS     = None
N_AVG             = 4
N_REPEATS         = 100
C_REG             = 1.0
RANDOM_SEED       = 42
MIN_PT_TRAIN      = 4     # min pseudotrials/class required in a fold's TRAIN set
MIN_PT_TEST       = 2     # min pseudotrials/class required in a fold's TEST set (thin held-out image)
MIN_VALID_FOLDS   = 4     # skip a subject if fewer than this many of the 16 folds are usable

CLUSTER_TAIL      = 1
CLUSTER_ALPHA     = 0.05
N_PERMUTATIONS    = 10000
CHANCE_LEVEL      = 0.5

RP_CODE, SW_CODE = 1, 3   # triggerCode1 for rpCond / swCond inducers


def make_clf():
    return make_pipeline(StandardScaler(),
                         LogisticRegression(C=C_REG, max_iter=1000, solver="liblinear"))


def make_pseudotrials(X, y, n_avg, rng):
    """Average random groups of n_avg same-class trials. Remainder dropped."""
    Xp, yp = [], []
    for cls in np.unique(y):
        idx = np.where(y == cls)[0]
        rng.shuffle(idx)
        for g in range(len(idx) // n_avg):
            grp = idx[g*n_avg:(g+1)*n_avg]
            Xp.append(X[grp].mean(0)); yp.append(cls)
    if not Xp:
        return np.empty((0,)+X.shape[1:]), np.empty((0,), int)
    return np.stack(Xp), np.asarray(yp, int)


def decode_loio(epochs, rng, desc=""):
    """16-combination crossed leave-one-image-out. Returns dict with per-repeat,
    per-fold AUC and pooled held-out predictions, or None if too few usable folds."""
    meta = epochs.metadata
    is_ind = meta.itemType.values == 'inducer'
    code   = meta.triggerCode1.values
    stim   = meta.stimId.values
    Xall   = epochs.get_data()
    n_times = Xall.shape[2]

    rp_imgs = np.unique(stim[is_ind & (code == RP_CODE)])
    sw_imgs = np.unique(stim[is_ind & (code == SW_CODE)])
    # need >=2 images per condition to hold one out and still have a training set
    if len(rp_imgs) < 2 or len(sw_imgs) < 2:
        return None

    # enumerate all held-out (sw_image, rp_image) combinations
    folds = list(itertools.product(sw_imgs, rp_imgs))   # up to 4x4 = 16

    sl = SlidingEstimator(make_clf(), scoring="roc_auc", n_jobs=1, verbose=False)

    # accumulate: for each repeat, a list of per-fold AUC curves (valid folds only)
    per_repeat = []                 # list over repeats of (n_valid_folds, n_times)
    pt_true, pt_score, pt_fold = [], [], []   # pooled held-out predictions
    valid_fold_ids = None

    for rep in tqdm(range(N_REPEATS), desc=desc, leave=False, unit="rep"):
        fold_curves, fold_ids_this = [], []
        for fi, (sw_hold, rp_hold) in enumerate(folds):
            test_mask  = is_ind & (((code == SW_CODE) & (stim == sw_hold)) |
                                   ((code == RP_CODE) & (stim == rp_hold)))
            train_mask = is_ind & ~test_mask & np.isin(code, (RP_CODE, SW_CODE))

            ytr_raw = (code[train_mask] == SW_CODE).astype(int)   # 1=swCond,0=rpCond
            yte_raw = (code[test_mask]  == SW_CODE).astype(int)

            # feasibility for this fold
            ctr = np.bincount(ytr_raw, minlength=2)
            cte = np.bincount(yte_raw, minlength=2)
            if ctr.min() < N_AVG * MIN_PT_TRAIN or cte.min() < N_AVG * MIN_PT_TEST:
                continue

            Xtr, ytr = make_pseudotrials(Xall[train_mask], ytr_raw, N_AVG, rng)
            Xte, yte = make_pseudotrials(Xall[test_mask],  yte_raw, N_AVG, rng)
            if len(np.unique(ytr)) < 2 or len(np.unique(yte)) < 2:
                continue

            est = clone(sl).fit(Xtr, ytr)
            fold_curves.append(est.score(Xte, yte))       # (n_times,) AUC on held-out images
            fold_ids_this.append(fi)

            # pooled predictions for any-metric-later (held-out-image pseudotrials)
            proba = est.predict_proba(Xte)[..., 1]        # P(swCond)
            pt_true.append(yte); pt_score.append(proba)
            pt_fold.append(np.full(len(yte), fi))

        if not fold_curves:
            continue
        per_repeat.append(np.mean(fold_curves, axis=0))   # avg over valid folds this repeat
        if valid_fold_ids is None:
            valid_fold_ids = set(fold_ids_this)
        else:
            valid_fold_ids |= set(fold_ids_this)

    if len(per_repeat) == 0 or valid_fold_ids is None or len(valid_fold_ids) < MIN_VALID_FOLDS:
        return None

    scores = np.stack(per_repeat)                          # (n_repeats_kept, n_times)
    return {"scores": scores, "n_times": n_times,
            "n_valid_folds": len(valid_fold_ids),
            "preds": {"y_true": np.concatenate(pt_true),
                      "y_score": np.concatenate(pt_score, axis=0),
                      "fold":   np.concatenate(pt_fold)}}


# ---- group stats / plotting / provenance (shared conventions) ----
def group_cluster_test(group_scores):
    from scipy.stats import t as t_dist
    data = group_scores - CHANCE_LEVEL
    thr = t_dist.ppf(1 - CLUSTER_ALPHA, df=data.shape[0]-1)
    t_obs, clusters, pv, _ = permutation_cluster_1samp_test(
        data, threshold=thr, n_permutations=N_PERMUTATIONS,
        tail=CLUSTER_TAIL, seed=RANDOM_SEED, out_type="mask", verbose=False)
    return t_obs, clusters, pv


def _mask(c, n):
    """Normalize a cluster (bool mask / int idx / tuple / slice) to a bool mask."""
    m = np.zeros(n, bool)
    if isinstance(c, tuple):
        c = c[0]
    if isinstance(c, slice):
        m[c] = True
        return m
    c = np.asarray(c)
    if c.dtype == bool:
        m[:len(c)] = c
    else:
        m[c.astype(int)] = True
    return m


def report_clusters(times, clusters, pv, label):
    print(f"\n=== {label}: clusters p<{CLUSTER_ALPHA} ===")
    hit = False
    for c, p in zip(clusters, pv):
        if p < CLUSTER_ALPHA:
            hit = True; tt = times[_mask(c, len(times))]
            print(f"  {tt.min()*1000:6.0f} to {tt.max()*1000:6.0f} ms   p={p:.4f}")
    if not hit:
        print("  none (held-out-image test is thin; a small effect may not survive correction)")


def plot_curve(times, g, clusters, pv, title, fname, ylabel="Decoding (roc_auc)"):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    mean = g.mean(0); sem = g.std(0, ddof=1)/np.sqrt(g.shape[0])
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.axhline(CHANCE_LEVEL, color="k", ls="--", lw=1, label="chance")
    ax.axvline(0, color="k", ls=":", lw=1); ax.axvline(0.152, color="grey", ls=":", lw=1)
    ax.fill_between(times*1000, mean-sem, mean+sem, alpha=0.25)
    ax.plot(times*1000, mean, lw=2)
    shaded = False
    for c, p in zip(clusters, pv):
        if p < CLUSTER_ALPHA:
            ax.fill_between(times*1000, CHANCE_LEVEL, mean, where=_mask(c, len(times)),
                            color="red", alpha=0.2,
                            label="p < .05 (cluster-corrected)" if not shaded else None)
            shaded = True
    ax.set(xlabel="Time from item cue (ms)", ylabel=ylabel,
           title=title, xlim=[times[0]*1000, times[-1]*1000])
    ax.legend(loc="upper right", fontsize=8); fig.tight_layout()
    fig.savefig(fname, dpi=200); plt.close(fig)
    print(f"  wrote {fname}")


def write_provenance(ids, times, dropped, fold_counts):
    prov = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "analysis": "model_c_loio (leave-one-image-out, 16-combination crossed)",
        "scheme": "cross every held-out swCond image with every held-out rpCond image "
                  "(4x4=16 folds); train on remaining 3+3 images, test on held-out pair; "
                  "average valid-fold curves per subject; group inference across subjects.",
        "interpretation": "above chance on held-out images = abstract, image-general reward "
                          "code (not image-specific/visual).",
        "n_subjects": len(ids), "subject_ids": [int(s) for s in ids],
        "dropped_subjects": [int(s) for s in dropped],
        "valid_folds_per_subject": {int(s): int(n) for s, n in fold_counts.items()},
        "params": {"n_avg": N_AVG, "n_repeats": N_REPEATS, "C_reg": C_REG,
                   "random_seed": RANDOM_SEED, "min_pt_train": MIN_PT_TRAIN,
                   "min_pt_test": MIN_PT_TEST, "min_valid_folds": MIN_VALID_FOLDS},
        "stats": {"cluster_tail": CLUSTER_TAIL, "cluster_alpha": CLUSTER_ALPHA,
                  "n_permutations": N_PERMUTATIONS, "chance_level": CHANCE_LEVEL},
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "mne": mne.__version__, "sklearn": sklearn.__version__},
    }
    with open(OUT_DIR / "provenance.json", "w") as f:
        json.dump(prov, f, indent=2)


def main():
    mne.set_log_level("WARNING")
    subs = discover_subjects(KEEP_SUBJECTS)
    print(f"Found {len(subs)} subjects.")

    times_path = OUT_DIR / "times.npy"
    times = np.load(times_path) if times_path.exists() else None

    curves, ids, dropped, fold_counts = [], [], [], {}

    for sid in tqdm(subs, desc="Subjects", unit="subj"):
        cached = SCORES_DIR / f"S{sid}.npy"
        if RESUME and cached.exists():
            arr = np.load(cached)
            if arr.ndim != 2:
                raise ValueError(f"S{sid}: cached scores shape {arr.shape}; expected "
                                 f"(n_repeats, n_times).")
            curves.append(arr.mean(0)); ids.append(sid)
            continue

        try:
            epochs, _ = load_subject(sid)
        except FileNotFoundError as exc:
            dropped.append(sid)
            tqdm.write(f"  S{sid}: EEG/metadata missing -> dropped\n  {exc}")
            continue

        if times is None:
            times = epochs.times.copy()
        elif len(times) != len(epochs.times) or not np.allclose(times, epochs.times):
            raise ValueError(f"S{sid}: epoch time vector differs from earlier subjects/cache.")

        rng = np.random.default_rng([RANDOM_SEED, int(sid), 2])   # subject-specific stream
        res = decode_loio(epochs, rng, desc=f"S{sid}")
        if res is None:
            dropped.append(sid)
            tqdm.write(f"  S{sid}: too few usable held-out-image folds -> dropped")
            continue

        np.save(cached, res["scores"])
        np.savez_compressed(PREDS_DIR / f"S{sid}.npz",
                            y_true=res["preds"]["y_true"],
                            y_score=res["preds"]["y_score"],
                            fold=res["preds"]["fold"])
        curves.append(res["scores"].mean(0)); ids.append(sid)
        fold_counts[sid] = res["n_valid_folds"]

    if not curves:
        print("No subjects had enough held-out-image data. "
              "Consider N_AVG=2 or relaxing MIN_PT_TEST.")
        return

    if times is None:
        epochs, _ = load_subject(ids[0]); times = epochs.times.copy()

    ids = np.asarray(ids, dtype=int)
    g = np.stack(curves)
    if g.shape[1] != len(times):
        raise ValueError(f"Group curves have {g.shape[1]} time points, "
                         f"time vector has {len(times)}.")

    np.save(OUT_DIR / "group_loio_scores.npy", g)
    np.save(OUT_DIR / "subject_ids.npy", ids)
    np.save(OUT_DIR / "times.npy", times)
    write_provenance(ids, times, dropped, fold_counts)

    print(f"\nLeave-one-image-out (16-combination crossed): N={len(ids)}, dropped={len(dropped)}")
    if fold_counts:
        vf = np.array(list(fold_counts.values()))
        print(f"valid folds/subject: median={int(np.median(vf))}, min={vf.min()}, max={vf.max()} (of 16)")
    print("Above chance on held-out images = abstract, image-general reward code.")
    _, clusters, pv = group_cluster_test(g)
    report_clusters(times, clusters, pv, "Model C leave-one-image-out")
    plot_curve(times, g, clusters, pv,
               f"Leave-one-image-out: decode held-out images (16-fold crossed, N={len(ids)})",
               OUT_DIR / f"decode_model_c_N{len(ids)}.png",
               ylabel="held-out image decode (roc_auc)")

    print("\nDone. Outputs in", OUT_DIR)


if __name__ == "__main__":
    main()
