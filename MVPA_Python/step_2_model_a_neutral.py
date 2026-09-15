"""
================================================================================
Model A-neutral: item-level vs block-context decoding (train blocks 1,2,4,5 -> test block 3)
ISSC_wCue
================================================================================

THE QUESTION
------------
The main Model A decode (rpCond vs swCond) is confounded with block context:
rpCond items live mostly in low-bkSRProb blocks (2,4), swCond mostly in high blocks
(1,5). So above-chance decoding there could reflect the ITEM's learned reward
association (the paper's item-level claim) OR the BLOCK's reward context (a
list/block-level effect, i.e. the Braem-level result, not the item-level one).

Block 3 is the neutral list: BOTH rpCond and swCond items appear there, in the SAME
neutral context. So:

  Train a classifier on rpCond vs swCond using blocks 1,2,4,5 (item & context
  perfectly confounded), then TEST it on block 3 (neutral context, both items).

  - If the classifier learned BLOCK CONTEXT -> chance in block 3 (no context
    difference exists there to read).
  - If it learned the ITEM-level association -> above chance in block 3.

This is a clean dissociation and the neural analog of E1's item-level conditioning
test (inducers in the neutral list).

DESIGN NOTE
-----------
This is a generalization test: block 3 is entirely held out from training, so no
cross-validation is needed for the block-3 score itself (train on all 1/2/4/5,
test on all of block 3). We still resample pseudotrial groupings (N_REPEATS) for
stability. A within-block-3 cross-validated decode is also computed as a secondary
check (see decode_block3_cv), though it is thin.

OUTPUTS (same philosophy as model_a_decode.py -- save raw, aggregate later)
  scores_full/<sid>.npy   (n_repeats, n_times)  block-3 generalization AUC
  preds/<sid>.npz         held-out block-3 decision scores + true labels per timepoint
  group_gen_scores.npy    (n_subjects, n_times)
  subject_ids.npy, times.npy, provenance.json
  decode_a_neutral_N*.png group curve with significant cluster shaded
================================================================================
"""

from pathlib import Path
import json, platform
from datetime import datetime, timezone
import numpy as np
import mne, sklearn
from mne.decoding import SlidingEstimator
from mne.stats import permutation_cluster_1samp_test
from sklearn.base import clone
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm

from mvpa_io import load_subject, discover_subjects, ROOT

# ---- output ----
OUT_DIR = ROOT / "mvpa_results" / "model_a_neutral_58"
SCORES_DIR = OUT_DIR / "scores_full"
PREDS_DIR = OUT_DIR / "preds"
for d in (OUT_DIR, SCORES_DIR, PREDS_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ---- config (mirrors the main decode) ----
RESUME        = True     # reuse per-subject arrays already on disk
RUN_BLOCK3_CV = False    # secondary within-block-3 CV; ~5x the cost of the primary
CV_DIR = OUT_DIR / "block3cv"
CV_DIR.mkdir(parents=True, exist_ok=True)
KEEP_SUBJECTS   = None
N_AVG           = 4
N_REPEATS       = 100
C_REG           = 1.0
RANDOM_SEED     = 42
MIN_PT_PER_CLASS = 4      # skip a subject if train OR block-3 test can't form this many pseudotrials/class

CLUSTER_TAIL    = 1
CLUSTER_ALPHA   = 0.05
N_PERMUTATIONS  = 10000
CHANCE_LEVEL    = 0.5

TRAIN_BLOCKS = [1, 2, 4, 5]
TEST_BLOCK   = 3


def make_clf():
    return make_pipeline(StandardScaler(),
                         LogisticRegression(C=C_REG, max_iter=1000, solver="liblinear"))


def make_pseudotrials(X, y, n_avg, rng):
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


def _condition_labels(meta):
    """1 for swCond, 0 for rpCond, NaN otherwise (inducers only)."""
    y = np.full(len(meta), np.nan)
    y[(meta.itemType.values == 'inducer') & (meta.triggerCode1.values == 1)] = 0  # rpCond
    y[(meta.itemType.values == 'inducer') & (meta.triggerCode1.values == 3)] = 1  # swCond
    return y


def decode_generalize(epochs, rng, desc=""):
    """Train rpCond-vs-swCond on TRAIN_BLOCKS, test (generalize) on TEST_BLOCK.
    Returns dict(scores (n_repeats,n_times), preds{...}) or None if too thin."""
    meta = epochs.metadata
    ycond = _condition_labels(meta)
    is_train = np.isin(meta.blockId.values, TRAIN_BLOCKS) & ~np.isnan(ycond)
    is_test  = (meta.blockId.values == TEST_BLOCK) & ~np.isnan(ycond)

    Xall = epochs.get_data()
    Xtr_raw, ytr_raw = Xall[is_train], ycond[is_train].astype(int)
    Xte_raw, yte_raw = Xall[is_test],  ycond[is_test].astype(int)

    # feasibility: both classes present with enough trials to form pseudotrials
    def ok(y):
        c = np.bincount(y, minlength=2)
        return c.min() >= N_AVG * MIN_PT_PER_CLASS
    if not (ok(ytr_raw) and ok(yte_raw)):
        return None

    sl = SlidingEstimator(make_clf(), scoring="roc_auc", n_jobs=1, verbose=False)
    n_times = Xall.shape[2]
    curves, pt_true, pt_score = [], [], []

    for _ in tqdm(range(N_REPEATS), desc=desc, leave=False, unit="rep"):
        Xtr, ytr = make_pseudotrials(Xtr_raw, ytr_raw, N_AVG, rng)
        Xte, yte = make_pseudotrials(Xte_raw, yte_raw, N_AVG, rng)
        if len(np.unique(ytr)) < 2 or len(np.unique(yte)) < 2:
            continue
        est = clone(sl).fit(Xtr, ytr)
        curves.append(est.score(Xte, yte))                       # (n_times,) AUC
        # decision scores on held-out block-3 pseudotrials (for any-metric-later)
        proba = est.predict_proba(Xte)[..., 1]                   # (n_te, n_times)
        pt_true.append(yte); pt_score.append(proba)

    if not curves:
        return None
    return {"scores": np.stack(curves),
            "preds": {"y_true": np.concatenate(pt_true),
                      "y_score": np.concatenate(pt_score, axis=0)},
            "n_times": n_times}


def decode_block3_cv(epochs, rng):
    """Secondary check: cross-validated rpCond-vs-swCond WITHIN block 3 only.
    Pure item effect, no context, but thin (~8 pseudotrials/class). Returns
    (n_times,) mean AUC or None."""
    meta = epochs.metadata
    ycond = _condition_labels(meta)
    m = (meta.blockId.values == TEST_BLOCK) & ~np.isnan(ycond)
    X, y = epochs.get_data()[m], ycond[m].astype(int)
    if np.bincount(y, minlength=2).min() < N_AVG * MIN_PT_PER_CLASS:
        return None
    sl = SlidingEstimator(make_clf(), scoring="roc_auc", n_jobs=1, verbose=False)
    curves = []
    for _ in range(N_REPEATS):
        Xp, yp = make_pseudotrials(X, y, N_AVG, rng)
        c = np.bincount(yp)
        if len(c) < 2 or c.min() < 2:
            continue
        cv = StratifiedKFold(n_splits=int(min(5, c.min())), shuffle=True,
                             random_state=rng.integers(1e9))
        from mne.decoding import cross_val_multiscore
        curves.append(cross_val_multiscore(sl, Xp, yp, cv=cv, n_jobs=-1).mean(0))
    return np.mean(curves, 0) if curves else None


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
        print("  none (inspect curve; thin test set means low power here)")


def plot_curve(times, g, clusters, pv, title, fname):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    mean = g.mean(0); sem = g.std(0, ddof=1)/np.sqrt(g.shape[0])
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.axhline(CHANCE_LEVEL, color="k", ls="--", lw=1, label="chance")
    ax.axvline(0, color="k", ls=":", lw=1); ax.axvline(0.152, color="grey", ls=":", lw=1)
    ax.fill_between(times*1000, mean-sem, mean+sem, alpha=0.25)
    ax.plot(times*1000, mean, lw=2)
    for c, p in zip(clusters, pv):
        if p < CLUSTER_ALPHA:
            ax.fill_between(times*1000, CHANCE_LEVEL, mean, where=_mask(c, len(times)),
                            color="red", alpha=0.2)
    ax.set(xlabel="Time from item cue (ms)", ylabel="Decoding (roc_auc)",
           title=title, xlim=[times[0]*1000, times[-1]*1000])
    ax.legend(loc="upper right", fontsize=8); fig.tight_layout()
    fig.savefig(fname, dpi=200); plt.close(fig)
    print(f"  wrote {fname}")


def write_provenance(ids, times, dropped):
    prov = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "analysis": "model_a_neutral (train blocks 1,2,4,5 -> test block 3)",
        "train_blocks": TRAIN_BLOCKS, "test_block": TEST_BLOCK,
        "n_subjects": len(ids), "subject_ids": [int(s) for s in ids],
        "dropped_subjects": [int(s) for s in dropped],
        "params": {"n_avg": N_AVG, "n_repeats": N_REPEATS, "C_reg": C_REG,
                   "random_seed": RANDOM_SEED, "min_pt_per_class": MIN_PT_PER_CLASS},
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

    # Reuse saved time vector when resuming.
    times_path = OUT_DIR / "times.npy"
    times = np.load(times_path) if times_path.exists() else None

    gen_curves = []
    cv_curves = []
    ids = []
    cv_ids = []
    dropped = []

    for sid in tqdm(subs, desc="Subjects", unit="subj"):
        gen_cached = SCORES_DIR / f"S{sid}.npy"
        pred_cached = PREDS_DIR / f"S{sid}.npz"
        cv_cached = CV_DIR / f"S{sid}.npy"

        # Treat the main result as cached only when both scores and predictions exist.
        have_gen_cache = (
            RESUME
            and gen_cached.exists()
            and pred_cached.exists()
        )

        have_cv_cache = (
            RUN_BLOCK3_CV
            and RESUME
            and cv_cached.exists()
        )

        need_gen = not have_gen_cache
        need_cv = RUN_BLOCK3_CV and not have_cv_cache

        # Load the primary cached result immediately.
        if have_gen_cache:
            cached_scores = np.load(gen_cached)

            if cached_scores.ndim != 2:
                raise ValueError(
                    f"S{sid}: cached generalization scores have shape "
                    f"{cached_scores.shape}; expected "
                    f"(n_repeats, n_times)."
                )

            gen_curves.append(cached_scores.mean(axis=0))
            ids.append(sid)

        # If everything requested for this subject is cached, never load the .set.
        if not need_gen and not need_cv:
            if RUN_BLOCK3_CV:
                cached_cv = np.load(cv_cached)

                if cached_cv.ndim != 1:
                    raise ValueError(
                        f"S{sid}: cached block-3 CV curve has shape "
                        f"{cached_cv.shape}; expected (n_times,)."
                    )

                cv_curves.append(cached_cv)
                cv_ids.append(sid)

            continue

        # At least one analysis still requires the EEG data.
        try:
            epochs, _ = load_subject(sid)
        except FileNotFoundError as exc:
            if need_gen:
                dropped.append(sid)
                tqdm.write(f" S{sid}: required EEG/metadata missing -> dropped")
            else:
                tqdm.write(
                    f" S{sid}: primary result cached, but EEG/metadata "
                    f"missing; block-3 CV skipped"
                )

            tqdm.write(f" {exc}")
            continue

        # Establish or verify the time vector.
        if times is None:
            times = epochs.times.copy()
        elif (
            len(times) != len(epochs.times)
            or not np.allclose(times, epochs.times)
        ):
            raise ValueError(
                f"S{sid}: epoch time vector differs from previous subjects "
                f"or cached times.npy."
            )

        # Use subject-specific RNG streams so resuming does not alter results.
        gen_rng = np.random.default_rng(
            [RANDOM_SEED, int(sid), 0]
        )
        cv_rng = np.random.default_rng(
            [RANDOM_SEED, int(sid), 1]
        )

        # Primary train-blocks -> neutral-block generalization analysis.
        if need_gen:
            res = decode_generalize(
                epochs,
                gen_rng,
                desc=f"S{sid}",
            )

            if res is None:
                dropped.append(sid)
                tqdm.write(
                    f" S{sid}: too few block-3 or training trials -> dropped"
                )
                continue

            np.save(gen_cached, res["scores"])

            np.savez_compressed(
                pred_cached,
                y_true=res["preds"]["y_true"],
                y_score=res["preds"]["y_score"],
            )

            gen_curves.append(res["scores"].mean(axis=0))
            ids.append(sid)

        # Optional secondary within-block-3 cross-validation.
        if RUN_BLOCK3_CV:
            if have_cv_cache:
                cached_cv = np.load(cv_cached)

                if cached_cv.ndim != 1:
                    raise ValueError(
                        f"S{sid}: cached block-3 CV curve has shape "
                        f"{cached_cv.shape}; expected (n_times,)."
                    )

                cv_curves.append(cached_cv)
                cv_ids.append(sid)

            else:
                cv = decode_block3_cv(epochs, cv_rng)

                if cv is not None:
                    np.save(cv_cached, cv)
                    cv_curves.append(cv)
                    cv_ids.append(sid)
                else:
                    tqdm.write(
                        f" S{sid}: too few block-3 trials for secondary CV"
                    )

    if not gen_curves:
        print(
            "No subjects had enough block-3 data. "
            "Consider N_AVG=2 or reporting the available trial counts."
        )
        return

    # All-cached edge case: recover times if times.npy did not yet exist.
    if times is None:
        first_sid = ids[0]
        epochs, _ = load_subject(first_sid)
        times = epochs.times.copy()

    ids = np.asarray(ids, dtype=int)
    g = np.stack(gen_curves)

    if g.shape[1] != len(times):
        raise ValueError(
            f"Group score curves contain {g.shape[1]} time points, "
            f"but the time vector contains {len(times)}."
        )

    np.save(OUT_DIR / "group_gen_scores.npy", g)
    np.save(OUT_DIR / "subject_ids.npy", ids)
    np.save(OUT_DIR / "times.npy", times)

    write_provenance(ids, times, dropped)

    print(
        f"\nGeneralization "
        f"(train {TRAIN_BLOCKS} -> test block {TEST_BLOCK}): "
        f"N={len(ids)}, dropped={len(dropped)}"
    )

    _, clusters, pv = group_cluster_test(g)

    report_clusters(
        times,
        clusters,
        pv,
        "A-neutral generalization",
    )

    plot_curve(
        times,
        g,
        clusters,
        pv,
        (
            f"Item-level test: train blocks {TRAIN_BLOCKS} "
            f"-> test block {TEST_BLOCK} (N={len(ids)})"
        ),
        OUT_DIR / f"decode_a_neutral_N{len(ids)}.png",
    )

    if cv_curves:
        gcv = np.stack(cv_curves)
        cv_ids_array = np.asarray(cv_ids, dtype=int)

        if gcv.shape[1] != len(times):
            raise ValueError(
                f"Block-3 CV curves contain {gcv.shape[1]} time points, "
                f"but the time vector contains {len(times)}."
            )

        np.save(
            OUT_DIR / "group_block3cv_scores.npy",
            gcv,
        )
        np.save(
            OUT_DIR / "block3cv_subject_ids.npy",
            cv_ids_array,
        )

        _, cl2, pv2 = group_cluster_test(gcv)

        report_clusters(
            times,
            cl2,
            pv2,
            "Block-3 within-CV (secondary, thin)",
        )

        plot_curve(
            times,
            gcv,
            cl2,
            pv2,
            f"Within block-3 CV (secondary, N={len(cv_ids_array)})",
            OUT_DIR / f"decode_block3cv_N{len(cv_ids_array)}.png",
        )

    elif RUN_BLOCK3_CV:
        print("\nNo subjects had sufficient data for block-3 within-CV.")

    print("\nDone. Outputs in", OUT_DIR)


if __name__ == "__main__":
    main()
