"""
================================================================================
model_tg_v2.py -- temporal generalization, run for TWO contrasts sequentially
ISSC_wCue  |  FINAL analysis
================================================================================

Trains a classifier at each time point and tests it at every other, producing a
train-time x test-time AUC matrix (King & Dehaene, 2014, TICS). Run for two
contrasts, in this order:

  1. "A"          rpCond vs swCond, ALL inducer trials, cross-validated.
                  Best-powered estimate; its diagonal reproduces the published
                  time-resolved Model A curve and serves as a sanity check.
                  NOTE: in these blocks item contingency is entangled with block
                  reward context, so this is the temporal generalization of the
                  rpCond-vs-swCond DISCRIMINATIVE PATTERN, not of an isolated
                  reward representation.

  2. "A_neutral"  train on TRAIN_BLOCKS = [1,2,4,5] (entangled), test on block 3
                  (neutral context, both item types). Context-controlled, so it
                  is the interpretively cleaner matrix. Needs NO cross-validation
                  (train and test trials are disjoint), which makes it ~n_folds
                  times cheaper AND gives a LARGER test set per estimate than a
                  Model A CV fold (all of block 3 vs one fifth of the inducers).

The two run sequentially and independently: each has its own cache, its own
outputs, and its own try/except, so a failure or interrupt in one never destroys
the other. Group matrices are saved BEFORE the (slow) cluster statistics, so a
crash in the stats step cannot lose the decoding work.

WHAT CHANGED FROM v1
--------------------
 1. DECIMATION GUARD. v1 claimed decimation was anti-aliased. It is not: MNE's
    Epochs.decimate() states "Low-pass filtering is not performed, this simply
    selects [every nth sample]" and warns about aliasing. v2 verifies the data's
    low-pass sits below the post-decimation Nyquist and refuses to proceed
    otherwise (override with ALLOW_UNSAFE_DECIM, deliberately explicit).

 2. ADAPTIVE FOLDS + PER-SPLIT VERIFICATION. v1's feasibility floor allowed
    32 raw trials/class, which under 5-fold leaves ~6 test trials/class and,
    after 4-trial averaging, ONE test pseudotrial per class. AUC from 1v1 is
    meaningless and v1's split check (both labels present) would have passed it.
    v2 derives n_splits from the actual counts and verifies EVERY split.

 3. CONFIG-HASHED CACHING. v1's resume accepted any 3-D array, so changing
    C_REG / N_AVG / N_FOLDS and re-running could silently mix configurations
    across subjects. v2 hashes the numerically-relevant config into the cache
    filename, so a changed parameter simply misses the cache and recomputes.

 4. SIGNED CLUSTER REPORTING. v1 merged all significant clusters into one mask,
    so above- and below-chance clusters got identical contours. With tail=0 the
    sign is meaningful (below-chance off-diagonal = reversed discriminative
    geometry), so v2 separates them, draws them differently, and saves full
    per-cluster records (sign, p, extent, peak).

 5. A_NEUTRAL contrast added (see above).

INTERPRETING THE GEOMETRY (deliberately conservative)
-----------------------------------------------------
  broad square      -> a temporally STABLE discriminative pattern
  narrow diagonal   -> a rapidly CHANGING discriminative pattern
  off-diagonal      -> a sufficiently SIMILAR pattern reappears later
  below chance      -> reversed discriminative geometry / decision-axis sign
These are canonical diagnostic shapes, not one-to-one maps onto mechanisms;
filtering, autocorrelation, SNR and latency jitter all affect apparent width.
Cluster significance is a CLUSTER-level statement, not a per-cell one.

OUTPUTS (mvpa_results/model_tg/<analysis>/)
  scores_full/S<sid>_<cfghash>.npy   (n_repeats, n_t, n_t) per-subject matrices
  group_tg_scores.npy                (n_subjects, n_t, n_t)
  subject_ids.npy, times.npy, provenance.json
  clusters.json                      signed, per-cluster records
  t_obs.npy, sig_pos.npy, sig_neg.npy
  tg_matrix_N*.png                   publication figure (signed contours)
  tg_diagonal_N*.png                 diagonal companion panel
  tg_summary_N*.png                  matrix + diagonal, one file
================================================================================
"""

from pathlib import Path
import json, platform, hashlib, traceback, time
from datetime import datetime, timezone
import numpy as np
import mne, sklearn
from mne.decoding import GeneralizingEstimator
from mne.stats import permutation_cluster_1samp_test
from sklearn.base import clone
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from scipy.stats import t as t_dist
from tqdm import tqdm

from mvpa_io import load_subject, discover_subjects, ROOT

# ---- output root ----
TG_ROOT = ROOT / "mvpa_results" / "model_tg"
TG_ROOT.mkdir(parents=True, exist_ok=True)

# ==============================================================================
# CONFIG
# ==============================================================================
ANALYSIS_VERSION   = 2          # bump to invalidate every cache deliberately
RESUME             = True
KEEP_SUBJECTS      = None       # setting to None will run all (add subj nums here to only run those)

N_AVG              = 4
N_REPEATS          = 20         # the matrix already averages heavily
N_FOLDS_MAX        = 5          # ceiling; actual folds chosen per subject
C_REG              = 1.0
RANDOM_SEED        = 42
DECIM              = 2          # 2 -> ~125 timepoints. Guarded (see below).
ALLOW_UNSAFE_DECIM = False      # set True ONLY if you have verified aliasing is not an issue

KNOWN_LOWPASS_HZ = 30.0         # verified by PSD (see check_low_pass.py): ~33 dB roll-off 20->30Hz
                                # noise floor above; EEGLAB filter not carried into MNE, set to none if not known

MIN_PT_TEST        = 4          # min pseudotrials/class required in a TEST set
MIN_PT_TRAIN       = 8          # min pseudotrials/class required in a TRAIN set

TRAIN_BLOCKS       = [1, 2, 4, 5]
TEST_BLOCK         = 3
RP_CODE, SW_CODE   = 1, 3       # y = 1 for swCond

# group stats
CLUSTER_ALPHA      = 0.05
CLUSTER_TAIL       = 0          # two-sided; the sign of off-diagonal effects matters
N_PERMUTATIONS     = 10000
CHANCE_LEVEL       = 0.5

ANALYSES = ("A_neutral", "A")
PRETTY = {"A":         "rpCond vs swCond (all inducer blocks, cross-validated)",
          "A_neutral": f"train blocks {TRAIN_BLOCKS} -> test block {TEST_BLOCK} (context-controlled)"}


def config_hash(analysis):
    """Hash of every parameter that can change the numbers. Included in the
    cache filename so a changed parameter misses the cache instead of silently
    mixing configurations across subjects."""
    cfg = {"analysis": analysis, "analysis_version": ANALYSIS_VERSION,
           "n_avg": N_AVG, "n_repeats": N_REPEATS, "n_folds_max": N_FOLDS_MAX,
           "c_reg": C_REG, "decim": DECIM, "random_seed": RANDOM_SEED,
           "min_pt_test": MIN_PT_TEST, "min_pt_train": MIN_PT_TRAIN,
           "train_blocks": TRAIN_BLOCKS, "test_block": TEST_BLOCK,
           "clf": "standardscaler+logreg_l2_liblinear"}
    return hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:10], cfg


# ==============================================================================
# CORE PIECES
# ==============================================================================
def make_clf():
    return make_pipeline(StandardScaler(),
                         LogisticRegression(C=C_REG, max_iter=1000, solver="liblinear"))


def make_pseudotrials(X, y, n_avg, rng):
    """Average random groups of n_avg same-class trials. X is (n, ch, t)."""
    Xp, yp = [], []
    for cls in np.unique(y):
        idx = np.where(y == cls)[0]
        rng.shuffle(idx)
        for g in range(len(idx) // n_avg):
            Xp.append(X[idx[g*n_avg:(g+1)*n_avg]].mean(0))
            yp.append(cls)
    if not Xp:
        return np.empty((0,) + X.shape[1:]), np.empty((0,), int)
    return np.stack(Xp), np.asarray(yp, int)


def safe_decimate(epochs, decim):
    """Decimate ONLY if the data are low-passed below the new Nyquist.
    MNE's decimate() does NOT filter; it subselects samples, so decimating
    insufficiently-filtered data aliases high frequencies into the band."""
    if decim is None or decim <= 1:
        return epochs
    old_sfreq = float(epochs.info["sfreq"])
    new_nyq = (old_sfreq / decim) / 2.0
    lowpass = epochs.info.get("lowpass", None)
    if KNOWN_LOWPASS_HZ is not None:
            lowpass = KNOWN_LOWPASS_HZ
    ok = (lowpass is not None) and (float(lowpass) < new_nyq)
    if not ok and not ALLOW_UNSAFE_DECIM:
        raise ValueError(
            f"Unsafe decimation: data lowpass = {lowpass} Hz, post-decimation "
            f"Nyquist = {new_nyq:.2f} Hz (sfreq {old_sfreq:.1f} / DECIM {decim}). "
            f"MNE's decimate() does NOT low-pass filter. Either low-pass the data "
            f"below {new_nyq:.2f} Hz, reduce DECIM, or set ALLOW_UNSAFE_DECIM=True "
            f"if you have verified aliasing is not a concern.")
    if not ok:
        print(f"  WARNING: decimating with lowpass={lowpass} Hz vs Nyquist "
              f"{new_nyq:.2f} Hz (ALLOW_UNSAFE_DECIM=True).")
    epochs.decimate(decim)
    return epochs


def choose_n_splits(y):
    """Largest fold count (<= N_FOLDS_MAX) whose TEST sets can still yield
    MIN_PT_TEST pseudotrials per class. Returns 0 if even 2 folds cannot."""
    min_raw = int(np.bincount(y, minlength=2).min())
    need_test_raw = N_AVG * MIN_PT_TEST
    n = min(N_FOLDS_MAX, min_raw // need_test_raw)
    return int(n) if n >= 2 else 0


def _fold_is_usable(ytr_raw, yte_raw):
    """Verify a concrete split BEFORE spending a matrix on it."""
    if len(np.unique(ytr_raw)) < 2 or len(np.unique(yte_raw)) < 2:
        return False
    if np.bincount(yte_raw, minlength=2).min() < N_AVG * MIN_PT_TEST:
        return False
    if np.bincount(ytr_raw, minlength=2).min() < N_AVG * MIN_PT_TRAIN:
        return False
    return True


# ==============================================================================
# DECODERS
# ==============================================================================
def decode_tg_cv(X, y, rng, desc=""):
    """Model A: cross-validated TG. CV splits over RAW TRIALS and wraps the whole
    time x time matrix, so no trial is ever in train and test at any time-pair.
    Pseudotrials are formed independently inside each split."""
    n_splits = choose_n_splits(y)
    if n_splits == 0:
        return None, {}
    gen = GeneralizingEstimator(make_clf(), scoring="roc_auc", n_jobs=-1, verbose=False)
    rep_mats, fold_counts = [], []
    for _ in tqdm(range(N_REPEATS), desc=desc, leave=False, unit="rep"):
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True,
                              random_state=int(rng.integers(1e9)))
        fold_mats = []
        for tr_idx, te_idx in skf.split(X, y):
            if not _fold_is_usable(y[tr_idx], y[te_idx]):
                continue
            Xtr, ytr = make_pseudotrials(X[tr_idx], y[tr_idx], N_AVG, rng)
            Xte, yte = make_pseudotrials(X[te_idx], y[te_idx], N_AVG, rng)
            if len(np.unique(ytr)) < 2 or len(np.unique(yte)) < 2:
                continue
            fold_mats.append(clone(gen).fit(Xtr, ytr).score(Xte, yte))
        if fold_mats:
            rep_mats.append(np.mean(fold_mats, axis=0))
            fold_counts.append(len(fold_mats))
    if not rep_mats:
        return None, {}
    return np.stack(rep_mats), {"n_splits": n_splits,
                                "mean_usable_folds": float(np.mean(fold_counts))}


def decode_tg_generalize(Xtr_raw, ytr_raw, Xte_raw, yte_raw, rng, desc=""):
    """A-neutral: train on TRAIN_BLOCKS, test on the held-out neutral block.
    No cross-validation: the two trial pools are already disjoint, so the whole
    of each is used (which gives a LARGER test set than a Model A CV fold)."""
    if np.bincount(ytr_raw, minlength=2).min() < N_AVG * MIN_PT_TRAIN:
        return None, {}
    if np.bincount(yte_raw, minlength=2).min() < N_AVG * MIN_PT_TEST:
        return None, {}
    gen = GeneralizingEstimator(make_clf(), scoring="roc_auc", n_jobs=-1, verbose=False)
    rep_mats = []
    for _ in tqdm(range(N_REPEATS), desc=desc, leave=False, unit="rep"):
        Xtr, ytr = make_pseudotrials(Xtr_raw, ytr_raw, N_AVG, rng)
        Xte, yte = make_pseudotrials(Xte_raw, yte_raw, N_AVG, rng)
        if len(np.unique(ytr)) < 2 or len(np.unique(yte)) < 2:
            continue
        rep_mats.append(clone(gen).fit(Xtr, ytr).score(Xte, yte))
    if not rep_mats:
        return None, {}
    return np.stack(rep_mats), {"n_splits": 0,
                                "train_pt_per_class": int(np.bincount(ytr).min()),
                                "test_pt_per_class": int(np.bincount(yte).min())}


# ==============================================================================
# GROUP STATISTICS (signed)
# ==============================================================================
def group_cluster_test_2d(group, times):
    """Two-sided 2-D cluster permutation vs chance over the (train x test) map.
    Returns t_obs, positive mask, negative mask, and per-cluster records."""
    data = group - CHANCE_LEVEL
    n = data.shape[0]
    thr = t_dist.ppf(1 - CLUSTER_ALPHA / 2, df=n - 1)      # two-sided threshold
    t_obs, clusters, pv, _ = permutation_cluster_1samp_test(
        data, threshold=thr, tail=CLUSTER_TAIL, n_permutations=N_PERMUTATIONS,
        seed=RANDOM_SEED, out_type="mask", verbose=True)

    mean_map = group.mean(0)
    pos = np.zeros(data.shape[1:], bool)
    neg = np.zeros(data.shape[1:], bool)
    records = []
    for c, p in zip(clusters, pv):
        m = np.asarray(c, bool)
        if p >= CLUSTER_ALPHA or not m.any():
            continue
        sign = 1 if t_obs[m].mean() > 0 else -1
        (pos if sign > 0 else neg)[m] = True
        tr_i, te_i = np.where(m)
        vals = mean_map[m]
        pk = np.argmax(vals) if sign > 0 else np.argmin(vals)
        records.append({
            "sign": "above_chance" if sign > 0 else "below_chance",
            "p_value": float(p), "n_cells": int(m.sum()),
            "train_ms": [float(times[tr_i.min()]*1000), float(times[tr_i.max()]*1000)],
            "test_ms":  [float(times[te_i.min()]*1000), float(times[te_i.max()]*1000)],
            "peak_auc": float(vals[pk]),
            "peak_train_ms": float(times[tr_i[pk]]*1000),
            "peak_test_ms":  float(times[te_i[pk]]*1000),
        })
    records.sort(key=lambda r: r["p_value"])
    return t_obs, pos, neg, records


# ==============================================================================
# FIGURES
# ==============================================================================
def _matrix_panel(ax, times, mean_map, pos, neg, title):
    tmin, tmax = times[0]*1000, times[-1]*1000
    vmax = np.nanmax(np.abs(mean_map - CHANCE_LEVEL))
    im = ax.imshow(mean_map, origin="lower", extent=[tmin, tmax, tmin, tmax],
                   cmap="RdBu_r", vmin=CHANCE_LEVEL - vmax, vmax=CHANCE_LEVEL + vmax,
                   aspect="equal", interpolation="nearest")
    ext = [tmin, tmax, tmin, tmax]
    if pos.any():
        ax.contour(pos.astype(float), levels=[0.5], colors="k", linewidths=1.1,
                   linestyles="solid", extent=ext, origin="lower")
    if neg.any():
        ax.contour(neg.astype(float), levels=[0.5], colors="k", linewidths=1.1,
                   linestyles="dashed", extent=ext, origin="lower")
    ax.plot([tmin, tmax], [tmin, tmax], color="k", lw=0.8, ls="--", alpha=0.45)
    ax.axhline(0, color="k", lw=0.6, ls=":"); ax.axvline(0, color="k", lw=0.6, ls=":")
    ax.set_xlabel("Test time (ms)", fontsize=10)
    ax.set_ylabel("Train time (ms)", fontsize=10)
    ax.set_title(title, fontsize=10.5)
    ax.tick_params(labelsize=8.5)
    return im


def _diagonal_panel(ax, times, group):
    diag = np.diagonal(group, axis1=1, axis2=2)
    mean = diag.mean(0); sem = diag.std(0, ddof=1)/np.sqrt(diag.shape[0])
    ax.axhline(CHANCE_LEVEL, color="k", ls="--", lw=1, label="chance")
    ax.axvline(0, color="k", ls=":", lw=0.9)
    ax.axvline(152, color="grey", ls=":", lw=0.9)
    ax.fill_between(times*1000, mean-sem, mean+sem, color="#cfe0f2", alpha=0.85)
    ax.plot(times*1000, mean, color="#2c7fb8", lw=2)
    ax.set(xlabel="Time (ms)", ylabel="Decoding AUC",
           xlim=[times[0]*1000, times[-1]*1000])
    ax.set_title("Matrix diagonal (time-resolved decoding)", fontsize=10)
    ax.legend(fontsize=8, loc="upper right"); ax.tick_params(labelsize=8.5)


def make_figures(times, group, pos, neg, records, analysis, out_dir):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n = group.shape[0]
    mean_map = group.mean(0)
    note = ("solid = above-chance cluster, dashed = below-chance; "
            "significance is cluster-level, not per-cell")

    fig, ax = plt.subplots(figsize=(6.4, 5.8))
    im = _matrix_panel(ax, times, mean_map, pos, neg,
                       f"Temporal generalization (N={n})\n{PRETTY[analysis]}")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("Decoding AUC", fontsize=9); cb.ax.tick_params(labelsize=8)
    fig.text(0.5, -0.01, note, ha="center", fontsize=7.5, color="#555")
    fig.tight_layout()
    f1 = out_dir / f"tg_matrix_N{n}.png"
    fig.savefig(f1, dpi=250, bbox_inches="tight"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    _diagonal_panel(ax, times, group)
    fig.tight_layout()
    f2 = out_dir / f"tg_diagonal_N{n}.png"
    fig.savefig(f2, dpi=250, bbox_inches="tight"); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12.6, 5.0),
                             gridspec_kw={"width_ratios": [1, 1.15]})
    im = _matrix_panel(axes[0], times, mean_map, pos, neg, "Temporal generalization")
    cb = fig.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04)
    cb.set_label("Decoding AUC", fontsize=9); cb.ax.tick_params(labelsize=8)
    _diagonal_panel(axes[1], times, group)
    fig.suptitle(f"{PRETTY[analysis]}   |   N = {n}   |   {note}", fontsize=9.5, y=1.02)
    fig.tight_layout()
    f3 = out_dir / f"tg_summary_N{n}.png"
    fig.savefig(f3, dpi=250, bbox_inches="tight"); plt.close(fig)

    for f in (f1, f2, f3):
        print(f"  wrote {f}")


# ==============================================================================
# ONE ANALYSIS, END TO END
# ==============================================================================
def run_analysis(analysis, subs):
    cfghash, cfg = config_hash(analysis)
    out_dir = TG_ROOT / analysis
    scores_dir = out_dir / "scores_full"
    for d in (out_dir, scores_dir):
        d.mkdir(parents=True, exist_ok=True)
    print(f"\n{'='*74}\nANALYSIS '{analysis}'  [{PRETTY[analysis]}]\n"
          f"cache key {cfghash}\n{'='*74}")

    times_path = out_dir / "times.npy"
    times = np.load(times_path) if times_path.exists() else None

    mats, ids, dropped, diag_info = [], [], [], {}
    t_start = time.time()

    for sid in tqdm(subs, desc=f"[{analysis}] subjects", unit="subj"):
        cached = scores_dir / f"S{sid}_{cfghash}.npy"
        if RESUME and cached.exists():
            arr = np.load(cached)
            if arr.ndim == 3 and arr.shape[1] == arr.shape[2]:
                mats.append(arr.mean(0)); ids.append(sid); continue
            tqdm.write(f"  S{sid}: malformed cache {arr.shape}; recomputing")

        try:
            epochs, meta = load_subject(sid)
        except FileNotFoundError as exc:
            dropped.append(sid); tqdm.write(f"  S{sid}: missing -> dropped ({exc})")
            continue

        epochs = safe_decimate(epochs, DECIM)     # raises if aliasing would occur
        if times is None:
            times = epochs.times.copy()
        elif len(times) != len(epochs.times) or not np.allclose(times, epochs.times):
            raise ValueError(f"S{sid}: time vector differs from earlier subjects "
                             f"(check DECIM consistency / stale times.npy).")

        is_ind = ((meta.itemType.values == "inducer")
                  & np.isin(meta.triggerCode1.values, (RP_CODE, SW_CODE)))
        code = meta.triggerCode1.values
        blk  = meta.blockId.values
        Xall = epochs.get_data()
        rng = np.random.default_rng([RANDOM_SEED, int(sid), 4, ANALYSIS_VERSION])

        if analysis == "A":
            X = Xall[is_ind]; y = (code[is_ind] == SW_CODE).astype(int)
            arr, info = decode_tg_cv(X, y, rng, desc=f"S{sid}")
        else:
            m_tr = is_ind & np.isin(blk, TRAIN_BLOCKS)
            m_te = is_ind & (blk == TEST_BLOCK)
            arr, info = decode_tg_generalize(
                Xall[m_tr], (code[m_tr] == SW_CODE).astype(int),
                Xall[m_te], (code[m_te] == SW_CODE).astype(int), rng, desc=f"S{sid}")

        if arr is None:
            dropped.append(sid)
            tqdm.write(f"  S{sid}: insufficient trials for a usable split -> dropped")
            continue
        np.save(cached, arr)
        mats.append(arr.mean(0)); ids.append(sid); diag_info[int(sid)] = info

    if not mats:
        print(f"[{analysis}] no subjects yielded a matrix."); return
    if times is None:
        ep, _ = load_subject(ids[0]); times = safe_decimate(ep, DECIM).times.copy()

    ids = np.asarray(ids, int)
    group = np.stack(mats)
    # save the decoding work BEFORE the slow stats, so a stats crash costs nothing
    np.save(out_dir / "group_tg_scores.npy", group)
    np.save(out_dir / "subject_ids.npy", ids)
    np.save(out_dir / "times.npy", times)
    elapsed = time.time() - t_start
    print(f"[{analysis}] N={len(ids)}, dropped={len(dropped)}, "
          f"matrix {group.shape[1]}x{group.shape[2]}, decoding took {elapsed/60:.1f} min")

    print(f"[{analysis}] 2-D cluster permutation ({N_PERMUTATIONS} perms)...")
    t_obs, pos, neg, records = group_cluster_test_2d(group, times)
    np.save(out_dir / "t_obs.npy", t_obs)
    np.save(out_dir / "sig_pos.npy", pos)
    np.save(out_dir / "sig_neg.npy", neg)
    with open(out_dir / "clusters.json", "w") as f:
        json.dump(records, f, indent=2)

    n_pos = sum(1 for r in records if r["sign"] == "above_chance")
    n_neg = len(records) - n_pos
    print(f"[{analysis}] significant clusters: {n_pos} above chance, {n_neg} below "
          f"(cells: {int(pos.sum())} / {int(neg.sum())} of {pos.size})")
    for r in records[:6]:
        print(f"    {r['sign']:>13}  p={r['p_value']:.4f}  cells={r['n_cells']:>5}  "
              f"train {r['train_ms'][0]:.0f}-{r['train_ms'][1]:.0f} ms  "
              f"test {r['test_ms'][0]:.0f}-{r['test_ms'][1]:.0f} ms  "
              f"peak AUC {r['peak_auc']:.3f} @ (train {r['peak_train_ms']:.0f}, "
              f"test {r['peak_test_ms']:.0f}) ms")

    make_figures(times, group, pos, neg, records, analysis, out_dir)

    with open(out_dir / "provenance.json", "w") as f:
        json.dump({
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "analysis": analysis, "description": PRETTY[analysis],
            "config_hash": cfghash, "config": cfg,
            "cv_convention": ("Model A: StratifiedKFold over RAW trials wrapping the "
                              "whole time x time matrix, pseudotrials formed inside "
                              "each split, no trial shared between train and test at "
                              "any time-pair (King & Dehaene 2014). A_neutral: no CV; "
                              "train and test blocks are disjoint by design."),
            "decimation": {"decim": DECIM,
                           "note": "MNE decimate() does NOT low-pass filter; a guard "
                                   "verifies data lowpass < post-decimation Nyquist",
                           "allow_unsafe": ALLOW_UNSAFE_DECIM},
            "fold_selection": ("n_splits chosen per subject as the largest value "
                               "<= N_FOLDS_MAX whose test sets still yield MIN_PT_TEST "
                               "pseudotrials/class; every concrete split re-verified"),
            "n_subjects": len(ids), "subject_ids": [int(s) for s in ids],
            "dropped_subjects": [int(s) for s in dropped],
            "per_subject_split_info": diag_info,
            "decoding_minutes": round(elapsed/60, 1),
            "stats": {"test": "2-D two-sided cluster permutation vs 0.5 "
                              "(lattice adjacency over the train x test grid)",
                      "tail": CLUSTER_TAIL, "alpha": CLUSTER_ALPHA,
                      "n_permutations": N_PERMUTATIONS,
                      "note": "cluster-level inference; individual cells within a "
                              "significant cluster are not individually significant"},
            "versions": {"python": platform.python_version(), "numpy": np.__version__,
                         "mne": mne.__version__, "sklearn": sklearn.__version__},
        }, f, indent=2)
    print(f"[{analysis}] done. Outputs in {out_dir}")


def main():
    mne.set_log_level("WARNING")
    subs = discover_subjects(KEEP_SUBJECTS)
    print(f"Found {len(subs)} subjects. DECIM={DECIM}, N_REPEATS={N_REPEATS}, "
          f"N_FOLDS_MAX={N_FOLDS_MAX}")
    failed = []
    for analysis in ANALYSES:
        try:
            run_analysis(analysis, subs)
        except Exception:
            failed.append(analysis)
            print(f"\n!!! analysis '{analysis}' FAILED; continuing to the next one.\n")
            traceback.print_exc()
    print("\n" + "="*74)
    print("ALL DONE." if not failed else f"DONE with failures in: {failed}")
    print(f"Outputs under {TG_ROOT}")


if __name__ == "__main__":
    main()
