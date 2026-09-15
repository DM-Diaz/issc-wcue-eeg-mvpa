"""
================================================================================
Model A: Time-resolved decoding of the learned reward association (rpCond vs swCond)
ISSC_wCue  --  item-cue-locked EEG, blocks 1-5 (cued phase)
================================================================================

Decodes bin 1 (rpCond / lowSR, item cue code 1) vs bin 2 (swCond / highSR, code 3)
at every time point, per subject, with pseudotrial averaging + stratified k-fold CV,
then tests the group curve against chance with a cluster-based permutation test.

--------------------------------------------------------------------------------
OUTPUT PHILOSOPHY  --  save raw, aggregate later
--------------------------------------------------------------------------------
Nothing that would require a re-run (or that a re-run could not recover) is thrown
away. Specifically we persist the following for reproducibility purposes:

  scores_full/<sid>.npy     (n_repeats, n_folds, n_times) AUC per fold per repeat
                            -> within-subject CIs, fold variance, bootstrap,
                               any re-aggregation later.
  preds/<sid>.npz           cross-validated y_true, y_pred, y_proba per timepoint
                            -> compute ANY metric after the fact (accuracy,
                               balanced acc, Cohen's kappa, F1, confusion matrices,
                               d-prime, sensitivity/specificity) with no re-run.
                               This is the one thing an aggregate-only script's
                               re-run could NOT recover.
  cv_group_scores.npy       (n_subjects, n_times) per-subject mean curve (convenience)
  subject_ids.npy           row-order for every group array (align to behavior / drop subjects)
  times.npy                 (n_times,) seconds
  provenance.json           exact config + versions that produced these files

Classifier WEIGHTS / Haufe patterns are NOT saved here (that would be a huge,
mostly-unused array). You run the companion `extract_patterns.py` over the significant
time window once the cluster is known -- patterns are interpreted where decoding
is significant.

--------------------------------------------------------------------------------
DEPENDENCIES (resolved)
--------------------------------------------------------------------------------
[A] Class selection by the item-cue tag in the event label ('(S1)' / '(S3)').
    Epochs must already be artifact-clean (flagged epochs removed in MATLAB via
    pop_rejepoch before export). Verify a subject's loaded count == its accepted
    count in Sx_artifactDetection.txt.
================================================================================
"""

from pathlib import Path
import json
import platform
from datetime import datetime, timezone
import numpy as np
import mne
import sklearn
from mne.decoding import SlidingEstimator, cross_val_multiscore
from mne.stats import permutation_cluster_1samp_test
from sklearn.base import clone
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm

# ==============================================================================
# CONFIG
# ==============================================================================
ROOT      = Path("/home/moon/Desktop/ISSC_wCue")
DATA_DIR  = ROOT / "preprocessed_2"        # artifact-clean Sx_preprocessed.set (+ .fdt)
OUT_DIR   = ROOT / "mvpa_results"
OUT_DIR.mkdir(exist_ok=True)

SCORES_DIR = OUT_DIR / "scores_full"       # per-subject (repeats, folds, times)
PREDS_DIR  = OUT_DIR / "preds"             # per-subject cross-validated predictions
SCORES_DIR.mkdir(exist_ok=True)
PREDS_DIR.mkdir(exist_ok=True)

KEEP_SUBJECTS = None    # None = every Sx_preprocessed.set in DATA_DIR; else a list

# --- class definition ---
CLASS_A_TAG  = "(S1)"    # rpCond / lowSR  (code 1)
CLASS_B_TAG  = "(S3)"    # swCond / highSR (code 3)
CLASS_A_NAME = "rpCond"
CLASS_B_NAME = "swCond"

# --- decoding parameters ---
N_AVG        = 4
N_REPEATS    = 100
N_FOLDS_MAX  = 5
SCORING      = "roc_auc"
C_REG        = 1.0
RANDOM_SEED  = 42

# --- what to save ---
SAVE_PREDICTIONS = True   # cross-validated preds for any-metric-later (recommended)

# --- group statistics ---
CLUSTER_TAIL    = 1
CLUSTER_ALPHA   = 0.05
N_PERMUTATIONS  = 10000
CHANCE_LEVEL    = 0.5


# ==============================================================================
# CLASSIFIER FACTORY
# ==============================================================================
def make_clf():
    """StandardScaler fit inside each fold (no leakage). Logistic regression with
    fixed C, deliberately untuned at this N."""
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(C=C_REG, max_iter=1000, solver="liblinear"),
    )


# ==============================================================================
# DATA LOADING
# ==============================================================================
def load_subject_epochs(set_path, verbose_ids=False):
    """Return (X, y, times). X: (n_epochs, n_ch, n_times); y in {0,1}
    (0=CLASS_A, 1=CLASS_B). Returns (None, None, None) if tags unset."""
    epochs = mne.io.read_epochs_eeglab(set_path, verbose="ERROR")

    if verbose_ids or CLASS_A_TAG is None or CLASS_B_TAG is None:
        print(f"\n  event_id in {set_path.name}:")
        for k, v in epochs.event_id.items():
            print(f"      {k!r}: {v}")
        if CLASS_A_TAG is None or CLASS_B_TAG is None:
            print("  >> Set CLASS_A_TAG / CLASS_B_TAG to match the labels above.")
            return None, None, None

    # select classes by the item-cue tag (bin prefixes aren't cleanly matchable
    # because of the comma in e.g. 'B1,3(S1)/4')
    a_names = [n for n in epochs.event_id if CLASS_A_TAG in n]
    b_names = [n for n in epochs.event_id if CLASS_B_TAG in n]
    if not a_names or not b_names:
        raise RuntimeError(f"class tags matched nothing in {set_path.name}: "
                           f"A={len(a_names)}, B={len(b_names)}")
    overlap = set(a_names) & set(b_names)
    assert not overlap, f"labels matched both classes in {set_path.name}: {overlap}"

    Xa = epochs[a_names].get_data()
    Xb = epochs[b_names].get_data()
    X = np.concatenate([Xa, Xb], axis=0)
    y = np.concatenate([np.zeros(len(Xa), int), np.ones(len(Xb), int)])
    return X, y, epochs.times


# ==============================================================================
# PSEUDOTRIALS
# ==============================================================================
def make_pseudotrials(X, y, n_avg, rng):
    """Average random groups of n_avg same-class trials. Remainder dropped."""
    Xp, yp = [], []
    for cls in np.unique(y):
        idx = np.where(y == cls)[0]
        rng.shuffle(idx)
        for g in range(len(idx) // n_avg):
            grp = idx[g * n_avg:(g + 1) * n_avg]
            Xp.append(X[grp].mean(axis=0))
            yp.append(cls)
    if not Xp:
        return np.empty((0,) + X.shape[1:]), np.empty((0,), int)
    return np.stack(Xp), np.asarray(yp, int)


# ==============================================================================
# WITHIN-SUBJECT DECODING  (returns EVERYTHING; aggregation happens later)
# ==============================================================================
def decode_within_subject(X, y, rng, desc=""):
    """Returns a dict:
      scores : (n_repeats_kept, N_FOLDS_MAX, n_times) AUC per fold per repeat
               (missing folds padded with NaN so the array is rectangular)
      preds  : dict {y_true (N,), y_pred (N,n_times), y_proba (N,n_times)} or None
      n_times : int
    or None if no valid repeat could be formed."""
    n_times = X.shape[2]
    sl = SlidingEstimator(make_clf(), scoring=SCORING, n_jobs=1, verbose=False)

    per_repeat_scores = []
    pt_true, pt_pred, pt_proba = [], [], []

    for r in tqdm(range(N_REPEATS), desc=desc, leave=False, unit="rep"):
        Xp, yp = make_pseudotrials(X, y, N_AVG, rng)
        counts = np.bincount(yp)
        if len(counts) < 2 or counts.min() < 2:
            continue
        n_splits = int(min(N_FOLDS_MAX, counts.min()))
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True,
                             random_state=rng.integers(1e9))

        # (a) scores per fold per timepoint
        scores = cross_val_multiscore(sl, Xp, yp, cv=cv, n_jobs=-1)  # (folds, times)
        padded = np.full((N_FOLDS_MAX, n_times), np.nan)
        padded[:scores.shape[0]] = scores
        per_repeat_scores.append(padded)

        # (b) cross-validated predictions for any-metric-later
        if SAVE_PREDICTIONS:
            for tr, te in cv.split(Xp, yp):
                est = clone(sl)
                est.fit(Xp[tr], yp[tr])
                pt_true.append(yp[te])
                pt_pred.append(est.predict(Xp[te]))                  # (n_te, n_times)
                pt_proba.append(est.predict_proba(Xp[te])[..., 1])   # P(class1), (n_te, n_times)

    if not per_repeat_scores:
        return None

    out = {"scores": np.stack(per_repeat_scores), "n_times": n_times, "preds": None}
    if SAVE_PREDICTIONS and pt_true:
        out["preds"] = {
            "y_true":  np.concatenate(pt_true),
            "y_pred":  np.concatenate(pt_pred, axis=0),
            "y_proba": np.concatenate(pt_proba, axis=0),
        }
    return out


# ==============================================================================
# GROUP STATISTICS
# ==============================================================================
def group_cluster_test(group_scores):
    """One-sample cluster permutation test of (scores - chance) across subjects."""
    from scipy.stats import t as t_dist
    data = group_scores - CHANCE_LEVEL
    n = data.shape[0]
    thresh = t_dist.ppf(1 - CLUSTER_ALPHA, df=n - 1)   # 1-sided t-threshold
    t_obs, clusters, cluster_pv, _ = permutation_cluster_1samp_test(
        data, threshold=thresh, n_permutations=N_PERMUTATIONS,
        tail=CLUSTER_TAIL, seed=RANDOM_SEED, out_type="mask", verbose=False)
    return t_obs, clusters, cluster_pv


def _cluster_to_mask(c, n_times):
    """Normalize any cluster form (bool mask / int idx / tuple / slice) to a mask."""
    mask = np.zeros(n_times, dtype=bool)
    if isinstance(c, tuple):
        c = c[0]
    if isinstance(c, slice):
        mask[c] = True
        return mask
    c = np.asarray(c)
    if c.dtype == bool:
        mask[:len(c)] = c
    else:
        mask[c.astype(int)] = True
    return mask


def report_clusters(times, clusters, cluster_pv, label):
    print(f"\n=== {label}: significant clusters (p < {CLUSTER_ALPHA}) ===")
    found = False
    for c, p in zip(clusters, cluster_pv):
        if p < CLUSTER_ALPHA:
            found = True
            m = _cluster_to_mask(c, len(times))
            tt = times[m]
            print(f"  {tt.min()*1000:6.0f} to {tt.max()*1000:6.0f} ms   p = {p:.4f}")
    if not found:
        print("  none (check the curve visually; a small effect may not survive correction)")


# ==============================================================================
# PLOTTING
# ==============================================================================
def plot_curve(times, group_scores, clusters, cluster_pv, title, fname):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    mean = group_scores.mean(0)
    sem = group_scores.std(0, ddof=1) / np.sqrt(group_scores.shape[0])

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.axhline(CHANCE_LEVEL, color="k", ls="--", lw=1, label="chance")
    ax.axvline(0, color="k", ls=":", lw=1)          # item cue
    ax.axvline(0.152, color="grey", ls=":", lw=1)   # task cue ~152 ms
    ax.fill_between(times * 1000, mean - sem, mean + sem, alpha=0.25)
    ax.plot(times * 1000, mean, lw=2)
    for c, p in zip(clusters, cluster_pv):
        if p < CLUSTER_ALPHA:
            m = _cluster_to_mask(c, len(times))
            ax.fill_between(times * 1000, CHANCE_LEVEL, mean, where=m,
                            color="red", alpha=0.2)
    ax.set(xlabel="Time from item cue (ms)", ylabel=f"Decoding ({SCORING})",
           title=title, xlim=[times[0]*1000, times[-1]*1000])
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(fname, dpi=200)
    plt.close(fig)
    print(f"  wrote {fname}")


# ==============================================================================
# PROVENANCE
# ==============================================================================
def write_provenance(subject_ids, times):
    prov = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "data_dir": str(DATA_DIR),
        "n_subjects": len(subject_ids),
        "subject_ids": [int(s) for s in subject_ids],
        "class_a": {"name": CLASS_A_NAME, "tag": CLASS_A_TAG},
        "class_b": {"name": CLASS_B_NAME, "tag": CLASS_B_TAG},
        "params": {
            "n_avg": N_AVG, "n_repeats": N_REPEATS, "n_folds_max": N_FOLDS_MAX,
            "scoring": SCORING, "C_reg": C_REG, "random_seed": RANDOM_SEED,
            "save_predictions": SAVE_PREDICTIONS,
        },
        "stats": {
            "cluster_tail": CLUSTER_TAIL, "cluster_alpha": CLUSTER_ALPHA,
            "n_permutations": N_PERMUTATIONS, "chance_level": CHANCE_LEVEL,
        },
        "n_times": int(len(times)),
        "tmin_s": float(times[0]), "tmax_s": float(times[-1]),
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__, "mne": mne.__version__,
            "sklearn": sklearn.__version__,
        },
    }
    with open(OUT_DIR / "provenance.json", "w") as f:
        json.dump(prov, f, indent=2)
    print(f"  wrote {OUT_DIR / 'provenance.json'}")


# ==============================================================================
# MAIN
# ==============================================================================
def discover_subjects():
    subs = []
    for f in sorted(DATA_DIR.glob("S*_preprocessed.set")):
        try:
            sid = int(f.stem.replace("S", "").replace("_preprocessed", ""))
        except ValueError:
            continue
        if KEEP_SUBJECTS is None or sid in KEEP_SUBJECTS:
            subs.append((sid, f))
    return subs


def main():
    mne.set_log_level("WARNING")   # silences per-fit "Fitting SlidingEstimator" bars
    rng = np.random.default_rng(RANDOM_SEED)
    subjects = discover_subjects()
    print(f"Found {len(subjects)} subjects to decode.")

    subject_means, subject_ids, times = [], [], None

    for sid, fpath in tqdm(subjects, desc="Subjects", unit="subj"):
        X, y, t = load_subject_epochs(fpath, verbose_ids=(times is None))
        if X is None:
            print("  (class tags not configured -- stopping)")
            return
        times = t

        res = decode_within_subject(X, y, rng, desc=f"S{sid}")
        if res is None:
            print(f"  S{sid}: too few pseudotrials -> skipped")
            continue

        # ---- save FULL per-subject arrays (nothing aggregated away) ----
        np.save(SCORES_DIR / f"S{sid}.npy", res["scores"])   # (repeats, folds, times)
        if res["preds"] is not None:
            np.savez_compressed(PREDS_DIR / f"S{sid}.npz",
                                y_true=res["preds"]["y_true"],
                                y_pred=res["preds"]["y_pred"],
                                y_proba=res["preds"]["y_proba"])

        # convenience per-subject mean curve (nanmean over repeats & folds)
        subject_means.append(np.nanmean(res["scores"], axis=(0, 1)))
        subject_ids.append(sid)

    if not subject_means:
        print("\nNo subjects decoded.")
        return

    # ---- group-level convenience arrays + provenance ----
    cv_group = np.stack(subject_means)                 # (n_subjects, n_times)
    subject_ids = np.asarray(subject_ids, int)
    np.save(OUT_DIR / "cv_group_scores.npy", cv_group)
    np.save(OUT_DIR / "subject_ids.npy", subject_ids)
    np.save(OUT_DIR / "times.npy", times)
    write_provenance(subject_ids, times)

    # ---- group statistics + figure ----
    _, clusters, cluster_pv = group_cluster_test(cv_group)
    report_clusters(times, clusters, cluster_pv, "Standard CV")
    plot_curve(times, cv_group, clusters, cluster_pv,
               f"{CLASS_A_NAME} vs {CLASS_B_NAME}  (standard CV, N={len(subject_ids)})",
               OUT_DIR / f"decode_cv_N{len(subject_ids)}.png")

    print("\nDone. Raw per-subject arrays in", SCORES_DIR, "and", PREDS_DIR)
    print("Group arrays, provenance, and figure in", OUT_DIR)


if __name__ == "__main__":
    main()
