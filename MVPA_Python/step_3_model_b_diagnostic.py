"""
================================================================================
Model B: diagnostic transfer -- does the neutral item inherit its list's control state?
ISSC_wCue
================================================================================

THE QUESTION
------------
Diagnostic items carry no reward contingency of their own, but they appear inside
two different lists:
    diag_SC = diagnostic trials from the switch-conditioned list (episode 1, high bkSRProb)
    diag_RC = diagnostic trials from the repeat-conditioned list (episode 2, low  bkSRProb)

We train the rpCond-vs-swCond decoder on the INDUCERS, then apply it to the
diagnostic items and ask whether diag_SC is pushed toward the swCond side and
diag_RC toward the rpCond side. If so, the (objectively neutral) diagnostic item
took on the control state of the list it lived in -- the neural analog of E1's
LIST-LEVEL conditioning effect (diagnostic RT differed by list).

WHAT THIS IS AND IS NOT
-----------------------
* Visual confound: DEFEATED by design. The diagnostic is the SAME four images in
  both lists (verified: identical stimIds across episodes 1 and 2). So a diag_SC
  vs diag_RC difference cannot be a low-level image difference -- it is the same
  pictures in different contexts.
* Context confound: INTRINSIC and unavoidable. diag_SC only ever occurs in
  high-context blocks (1,5) and diag_RC only in low-context blocks (2,4). The
  diagnostic never appears outside its list, so "acquired the list's control
  state" and "reflects the block's reward context" are OPERATIONALLY IDENTICAL
  here. Model B is therefore a LIST-LEVEL test, NOT an item-level one. (The
  item-level test is Model A-neutral, which CAN hold context constant via block 3.)
  This is stated plainly rather than engineered around.

DESIGN NOTE (validation scheme)
-------------------------------
Primary analysis is a GENERALIZATION test: the diagnostic items are a fully
held-out condition never seen during training, so NO cross-validation is needed
for the diagnostic score (train on all inducers, test on all diagnostics). We
still resample pseudotrial groupings (N_REPEATS) for stability, and test the
group AUC curve against chance with a cluster-based permutation test.

The training axis is only interpretable if it is real for that subject. So we
also run a WITHIN-INDUCER cross-validated decode (rpCond vs swCond) as a
sanity/validity check per subject -- this DOES use cross-validation, because
train and test come from the same inducer pool. A subject whose inducer decode
is at chance provides a meaningless axis to project the diagnostic onto; the
sanity curve lets us identify and, if desired, condition on that.

DIRECTION CONVENTION
--------------------
Positive class for the diagnostic AUC = diag_SC, scored by the decoder's
P(swCond). Above 0.5 therefore means diag_SC scored MORE swCond-like than
diag_RC, i.e. list-consistent transfer (the predicted direction). A null (no
transfer) sits at 0.5.

OUTPUTS (same philosophy as model_a_neutral.py -- save raw, aggregate later)
  scores_full/<sid>.npy   (n_repeats, n_times)  diagnostic-transfer AUC per repeat
  preds/<sid>.npz         held-out diagnostic decision scores + list labels per timepoint
                          (y_true = 1 for diag_SC, 0 for diag_RC; y_score = P(swCond))
  inducer_cv/<sid>.npy    (n_times,) within-inducer CV AUC (sanity/validity), if enabled
  group_transfer_scores.npy   (n_subjects, n_times)
  group_inducer_cv_scores.npy (n_subjects_cv, n_times)  if enabled
  subject_ids.npy, inducer_cv_ids.npy, times.npy, provenance.json
  decode_model_b_N*.png       group transfer curve with significant cluster shaded
  decode_inducer_cv_N*.png    group inducer-sanity curve (if enabled)
================================================================================
"""

from pathlib import Path
import json, platform
from datetime import datetime, timezone
import numpy as np
import mne, sklearn
from mne.decoding import SlidingEstimator, cross_val_multiscore
from mne.stats import permutation_cluster_1samp_test
from sklearn.base import clone
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm

from mvpa_io import load_subject, discover_subjects, ROOT

# ---- output ----
OUT_DIR    = ROOT / "mvpa_results" / "model_b_diagnostic"
SCORES_DIR = OUT_DIR / "scores_full"
PREDS_DIR  = OUT_DIR / "preds"
CV_DIR     = OUT_DIR / "inducer_cv"
for d in (OUT_DIR, SCORES_DIR, PREDS_DIR, CV_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ---- config (mirrors model_a_neutral) ----
RESUME            = True    # reuse per-subject arrays already on disk
RUN_INDUCER_CV    = True    # within-inducer CV sanity check (train axis validity)
KEEP_SUBJECTS     = None
N_AVG             = 4
N_REPEATS         = 100
C_REG             = 1.0
RANDOM_SEED       = 42
MIN_PT_PER_CLASS  = 4       # min pseudotrials/class required on BOTH train (inducer) and test (diagnostic) sides

CLUSTER_TAIL      = 1
CLUSTER_ALPHA     = 0.05
N_PERMUTATIONS    = 10000
CHANCE_LEVEL      = 0.5

# episode -> list mapping (verified from metadata: ep1=high/SC, ep2=low/RC)
EPISODE_SC = 1     # switch-conditioned list  (high bkSRProb, blocks 1 & 5)
EPISODE_RC = 2     # repeat-conditioned list  (low  bkSRProb, blocks 2 & 4)


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


def _inducer_labels(meta):
    """1 for swCond, 0 for rpCond, NaN otherwise (inducers only)."""
    y = np.full(len(meta), np.nan)
    y[(meta.itemType.values == 'inducer') & (meta.triggerCode1.values == 1)] = 0  # rpCond
    y[(meta.itemType.values == 'inducer') & (meta.triggerCode1.values == 3)] = 1  # swCond
    return y


def _diagnostic_list_labels(meta):
    """1 for diag_SC (episode 1 / high list), 0 for diag_RC (episode 2 / low list),
    NaN otherwise (diagnostic items only)."""
    y = np.full(len(meta), np.nan)
    is_diag = meta.itemType.values == 'diagnostic'
    y[is_diag & (meta.episode.values == EPISODE_SC)] = 1   # diag_SC (positive)
    y[is_diag & (meta.episode.values == EPISODE_RC)] = 0   # diag_RC
    return y


def decode_transfer(epochs, rng, desc=""):
    """PRIMARY: train rpCond-vs-swCond on ALL inducers, test on the diagnostic
    items split by list. AUC(diag_SC vs diag_RC) using the decoder's P(swCond).

    No cross-validation: the diagnostic items are entirely held out from training.
    Returns dict(scores (n_repeats,n_times), preds{...}) or None if too thin.
    """
    meta = epochs.metadata
    y_ind  = _inducer_labels(meta)
    y_diag = _diagnostic_list_labels(meta)

    Xall = epochs.get_data()
    is_ind  = ~np.isnan(y_ind)
    is_diag = ~np.isnan(y_diag)
    Xtr_raw, ytr_raw = Xall[is_ind],  y_ind[is_ind].astype(int)     # 0=rpCond,1=swCond
    Xte_raw, yte_raw = Xall[is_diag], y_diag[is_diag].astype(int)   # 0=diag_RC,1=diag_SC

    # feasibility: both classes present with enough trials on BOTH sides
    def ok(y):
        c = np.bincount(y, minlength=2)
        return len(c) >= 2 and c.min() >= N_AVG * MIN_PT_PER_CLASS
    if not (ok(ytr_raw) and ok(yte_raw)):
        return None

    sl = SlidingEstimator(make_clf(), scoring="roc_auc", n_jobs=1, verbose=False)
    n_times = Xall.shape[2]
    curves, pt_true, pt_score = [], [], []

    for _ in tqdm(range(N_REPEATS), desc=desc, leave=False, unit="rep"):
        # pseudotrials formed independently within the train (inducer) and test
        # (diagnostic) pools, so no raw trial is shared across train/test.
        Xtr, ytr = make_pseudotrials(Xtr_raw, ytr_raw, N_AVG, rng)
        Xte, yte = make_pseudotrials(Xte_raw, yte_raw, N_AVG, rng)
        if len(np.unique(ytr)) < 2 or len(np.unique(yte)) < 2:
            continue
        est = clone(sl).fit(Xtr, ytr)                        # learns rpCond-vs-swCond axis
        # P(swCond) for each diagnostic pseudotrial at each timepoint
        proba = est.predict_proba(Xte)[..., 1]               # (n_te, n_times)
        # AUC of diag_SC vs diag_RC using that score == how strongly the swCond
        # axis separates the two diagnostic lists in the predicted direction.
        # est.score with roc_auc computes exactly this given yte in {0,1}.
        curves.append(est.score(Xte, yte))                   # (n_times,) AUC
        pt_true.append(yte); pt_score.append(proba)

    if not curves:
        return None
    return {"scores": np.stack(curves),
            "preds": {"y_true": np.concatenate(pt_true),
                      "y_score": np.concatenate(pt_score, axis=0)},
            "n_times": n_times}


def decode_inducer_cv(epochs, rng):
    """SANITY/VALIDITY: within-inducer cross-validated rpCond-vs-swCond decode.
    Confirms the training axis is real for this subject before trusting transfer.
    DOES use cross-validation (train and test are the same inducer pool).
    Returns (n_times,) mean AUC or None if too thin.
    """
    meta = epochs.metadata
    y = _inducer_labels(meta)
    m = ~np.isnan(y)
    X, y = epochs.get_data()[m], y[m].astype(int)
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
        print("  none")


def plot_curve(times, g, clusters, pv, title, fname, ylabel="Decoding (roc_auc)"):
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
    ax.set(xlabel="Time from item cue (ms)", ylabel=ylabel,
           title=title, xlim=[times[0]*1000, times[-1]*1000])
    ax.legend(loc="upper right", fontsize=8); fig.tight_layout()
    fig.savefig(fname, dpi=200); plt.close(fig)
    print(f"  wrote {fname}")


def write_provenance(ids, cv_ids, times, dropped):
    prov = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "analysis": "model_b_diagnostic (train inducers rpCond-vs-swCond -> test diagnostic by list)",
        "direction": "positive class = diag_SC (episode 1, high list), scored by P(swCond); "
                     ">0.5 = list-consistent transfer",
        "episode_sc": EPISODE_SC, "episode_rc": EPISODE_RC,
        "confounds": {"visual": "defeated (same 4 diagnostic images in both lists)",
                      "context": "intrinsic/unavoidable (diagnostic never leaves its list); "
                                 "this is a LIST-level test, not item-level"},
        "n_subjects_transfer": len(ids), "subject_ids": [int(s) for s in ids],
        "n_subjects_inducer_cv": len(cv_ids), "inducer_cv_ids": [int(s) for s in cv_ids],
        "dropped_subjects": [int(s) for s in dropped],
        "params": {"n_avg": N_AVG, "n_repeats": N_REPEATS, "C_reg": C_REG,
                   "random_seed": RANDOM_SEED, "min_pt_per_class": MIN_PT_PER_CLASS,
                   "run_inducer_cv": RUN_INDUCER_CV},
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

    transfer_curves, cv_curves = [], []
    ids, cv_ids, dropped = [], [], []

    for sid in tqdm(subs, desc="Subjects", unit="subj"):
        gen_cached  = SCORES_DIR / f"S{sid}.npy"
        pred_cached = PREDS_DIR  / f"S{sid}.npz"
        cv_cached   = CV_DIR     / f"S{sid}.npy"

        have_gen_cache = RESUME and gen_cached.exists() and pred_cached.exists()
        have_cv_cache  = RUN_INDUCER_CV and RESUME and cv_cached.exists()
        need_gen = not have_gen_cache
        need_cv  = RUN_INDUCER_CV and not have_cv_cache

        # load primary cached result
        if have_gen_cache:
            cached = np.load(gen_cached)
            if cached.ndim != 2:
                raise ValueError(f"S{sid}: cached transfer scores shape {cached.shape}; "
                                 f"expected (n_repeats, n_times).")
            transfer_curves.append(cached.mean(axis=0)); ids.append(sid)

        # fully cached -> never touch the .set
        if not need_gen and not need_cv:
            if RUN_INDUCER_CV:
                cached_cv = np.load(cv_cached)
                if cached_cv.ndim != 1:
                    raise ValueError(f"S{sid}: cached inducer-CV curve shape {cached_cv.shape}; "
                                     f"expected (n_times,).")
                cv_curves.append(cached_cv); cv_ids.append(sid)
            continue

        # at least one analysis needs the EEG
        try:
            epochs, _ = load_subject(sid)
        except FileNotFoundError as exc:
            if need_gen:
                dropped.append(sid)
                tqdm.write(f"  S{sid}: required EEG/metadata missing -> dropped")
            else:
                tqdm.write(f"  S{sid}: primary cached, EEG/metadata missing; inducer-CV skipped")
            tqdm.write(f"  {exc}")
            continue

        # establish / verify the time vector
        if times is None:
            times = epochs.times.copy()
        elif len(times) != len(epochs.times) or not np.allclose(times, epochs.times):
            raise ValueError(f"S{sid}: epoch time vector differs from previous subjects "
                             f"or cached times.npy.")

        # subject-specific RNG streams so resuming never alters results
        gen_rng = np.random.default_rng([RANDOM_SEED, int(sid), 0])
        cv_rng  = np.random.default_rng([RANDOM_SEED, int(sid), 1])

        # primary diagnostic-transfer analysis
        if need_gen:
            res = decode_transfer(epochs, gen_rng, desc=f"S{sid}")
            if res is None:
                dropped.append(sid)
                tqdm.write(f"  S{sid}: too few inducer or diagnostic trials -> dropped")
                continue
            np.save(gen_cached, res["scores"])
            np.savez_compressed(pred_cached,
                                y_true=res["preds"]["y_true"],
                                y_score=res["preds"]["y_score"])
            transfer_curves.append(res["scores"].mean(axis=0)); ids.append(sid)

        # secondary inducer-CV sanity check
        if RUN_INDUCER_CV:
            if have_cv_cache:
                cached_cv = np.load(cv_cached)
                if cached_cv.ndim != 1:
                    raise ValueError(f"S{sid}: cached inducer-CV curve shape {cached_cv.shape}; "
                                     f"expected (n_times,).")
                cv_curves.append(cached_cv); cv_ids.append(sid)
            else:
                cv = decode_inducer_cv(epochs, cv_rng)
                if cv is not None:
                    np.save(cv_cached, cv)
                    cv_curves.append(cv); cv_ids.append(sid)
                else:
                    tqdm.write(f"  S{sid}: too few inducer trials for CV sanity check")

    if not transfer_curves:
        print("No subjects had enough diagnostic/inducer data. "
              "Consider N_AVG=2 or reporting the available trial counts.")
        return

    # all-cached edge case
    if times is None:
        epochs, _ = load_subject(ids[0]); times = epochs.times.copy()

    ids = np.asarray(ids, dtype=int)
    g = np.stack(transfer_curves)
    if g.shape[1] != len(times):
        raise ValueError(f"Group curves have {g.shape[1]} time points, "
                         f"time vector has {len(times)}.")

    np.save(OUT_DIR / "group_transfer_scores.npy", g)
    np.save(OUT_DIR / "subject_ids.npy", ids)
    np.save(OUT_DIR / "times.npy", times)

    # ---- primary result: diagnostic transfer ----
    print(f"\nDiagnostic transfer (train inducers -> test diagnostic by list): "
          f"N={len(ids)}, dropped={len(dropped)}")
    print("Positive direction = diag_SC scored more swCond-like than diag_RC "
          "(list-consistent transfer).")
    _, clusters, pv = group_cluster_test(g)
    report_clusters(times, clusters, pv, "Model B diagnostic transfer")
    plot_curve(times, g, clusters, pv,
               f"Diagnostic transfer: inducer axis applied to diagnostic by list (N={len(ids)})",
               OUT_DIR / f"decode_model_b_N{len(ids)}.png",
               ylabel="diag_SC vs diag_RC (roc_auc)")

    # ---- secondary result: inducer-CV sanity ----
    if RUN_INDUCER_CV and cv_curves:
        cv_ids = np.asarray(cv_ids, dtype=int)
        gcv = np.stack(cv_curves)
        np.save(OUT_DIR / "group_inducer_cv_scores.npy", gcv)
        np.save(OUT_DIR / "inducer_cv_ids.npy", cv_ids)
        _, cl2, pv2 = group_cluster_test(gcv)
        report_clusters(times, cl2, pv2, "Inducer-CV sanity (rpCond vs swCond, within-inducer)")
        plot_curve(times, gcv, cl2, pv2,
                   f"Inducer axis validity: within-inducer CV rpCond vs swCond (N={len(cv_ids)})",
                   OUT_DIR / f"decode_inducer_cv_N{len(cv_ids)}.png")

    write_provenance(ids, cv_ids if (RUN_INDUCER_CV and len(cv_curves)) else [], times, dropped)
    print("\nDone. Outputs in", OUT_DIR)


if __name__ == "__main__":
    main()
