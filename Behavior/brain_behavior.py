"""
================================================================================
brain_behavior.py -- does neural decoding of reward condition predict the
                     behavioral reward-by-transition switch cost?
ISSC_wCue
================================================================================

CO-PRIMARY ANALYSIS. Model A shows the reward conditions are neurally decodable;
this asks whether that neural signal tracks the LEARNING that drove behavior. If
subjects who separate the conditions more strongly in the brain also show a larger
behavioral reward-by-transition effect, the decode reflects the learned
association rather than an incidental difference.

TWO MEASURES, CORRELATED ACROSS SUBJECTS
----------------------------------------
NEURAL (x): each subject's mean decoding AUC in a chosen window (default
    150-250 ms, the Model A peak), read from the SAVED per-subject arrays so it
    is exactly the quantity that produced the group figure -- not recomputed.

BEHAVIORAL (y): the reward-by-transition interaction in RT, computed per subject:
        SC_swCond   = RT(swCond, switch) - RT(swCond, repeat)
        SC_rpCond   = RT(rpCond, switch) - RT(rpCond, repeat)
        interaction = SC_swCond - SC_rpCond
    on cued-phase INDUCER trials, using RT_exclude == 0 (correct trials within
    each subject's own RT bounds). This matches the behavioral effect the study
    reports and is the swCond-vs-rpCond contrast that mirrors the decode.

WHY SPEARMAN (default): the per-subject interaction has a few negative outliers
    and heavy-ish tails; a rank correlation is robust to those. Pearson is also
    reported for completeness.

ROBUSTNESS / CORRECTNESS SAFEGUARDS
-----------------------------------
 * Join on subject ID EXPLICITLY. subject_ids.npy is in lexicographic (string)
   order, NOT ascending numeric order, so positional alignment would silently
   mispair subjects. We build a dict keyed by ID.
 * Use only subjects present in BOTH the neural arrays and the behavioral file,
   reporting any dropped.
 * The window is read from times.npy; if it does not match the decode's sampling
   the script errors rather than silently averaging the wrong samples.
 * Bootstrap CI on the correlation, and an influence check (leave-one-subject-out
   range of r) so a single subject cannot masquerade as the effect.

Inputs (from Model A outputs):
    mvpa_results/cv_group_scores.npy   (n_subjects, n_times)
    mvpa_results/subject_ids.npy       (n_subjects,)  row order of the above
    mvpa_results/times.npy             (n_times,)
Behavioral:
    E3_N70.xlsx
Outputs (mvpa_results/brain_behavior/):
    brain_behavior.csv        per-subject neural AUC + behavioral interaction
    brain_behavior.png        scatter with fit, CI, and both correlations
    brain_behavior.json       stats + provenance
================================================================================
"""

from pathlib import Path
import json, platform
from datetime import datetime, timezone
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr

from mvpa_io import ROOT   # reuse the project root only

# ---- paths ----
MODEL_A_DIR = ROOT / "mvpa_results"                     # holds cv_group_scores.npy etc.
BEH_FILE    = ROOT / "E3_N70.xlsx"
OUT_DIR     = ROOT / "mvpa_results" / "brain_behavior" / "full_window"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---- config ----
WIN_START_S = 0.130      # neural window (Model A peak); match extract_patterns
WIN_END_S   = 0.798
RP_CODE, SW_CODE = 1, 3
CUED_BLOCKS = [1, 2, 3, 4, 5]
N_BOOTSTRAP = 10000
RANDOM_SEED = 42


# ==============================================================================
# NEURAL SIDE
# ==============================================================================
def load_neural():
    """Return {sid: mean AUC in window} from the SAVED Model A arrays."""
    g   = np.load(MODEL_A_DIR / "cv_group_scores.npy")   # (n_subj, n_times)
    ids = np.load(MODEL_A_DIR / "subject_ids.npy")
    t   = np.load(MODEL_A_DIR / "times.npy")
    if g.shape[0] != len(ids):
        raise ValueError(f"cv_group_scores has {g.shape[0]} rows but subject_ids has "
                         f"{len(ids)}.")
    if g.shape[1] != len(t):
        raise ValueError(f"cv_group_scores has {g.shape[1]} time points but times has "
                         f"{len(t)}.")
    wm = (t >= WIN_START_S) & (t <= WIN_END_S)
    if wm.sum() == 0:
        raise ValueError(f"window {WIN_START_S}-{WIN_END_S}s selects no samples; "
                         f"times run {t[0]:.3f}-{t[-1]:.3f}s.")
    auc = g[:, wm].mean(axis=1)
    return {int(s): float(a) for s, a in zip(ids, auc)}, int(wm.sum())


# ==============================================================================
# BEHAVIORAL SIDE
# ==============================================================================
def behavioral_interaction():
    """Return {sid: reward-by-transition RT interaction (ms)}."""
    df = pd.read_excel(BEH_FILE)
    d = df[(df.blockId.isin(CUED_BLOCKS)) &
           (df.itemType == "inducer") &
           (df.triggerCode1.isin([RP_CODE, SW_CODE])) &
           (df.RT_exclude == 0)].copy()
    d["cond"] = np.where(d.triggerCode1 == SW_CODE, "swCond", "rpCond")

    # cell means; require all four cells present per subject
    cell = d.groupby(["sbjId", "cond", "trialType"]).RT.mean().unstack("trialType")
    if not {"switch", "repeat"}.issubset(cell.columns):
        raise ValueError("behavioral file lacks 'switch'/'repeat' trialType labels.")
    cell["sc"] = cell["switch"] - cell["repeat"]              # switch cost per cond
    sc = cell["sc"].unstack("cond")                           # cols: rpCond, swCond
    complete = sc[["rpCond", "swCond"]].notna().all(axis=1)
    inter = (sc["swCond"] - sc["rpCond"])[complete]
    # also keep the components for the CSV
    comps = sc[complete].rename(columns={"rpCond": "sc_rpCond", "swCond": "sc_swCond"})
    return {int(s): float(v) for s, v in inter.items()}, comps


# ==============================================================================
# STATS
# ==============================================================================
def bootstrap_ci(x, y, method, n_boot, seed):
    rng = np.random.default_rng(seed)
    n = len(x); rs = np.empty(n_boot)
    fn = spearmanr if method == "spearman" else pearsonr
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        # guard against a degenerate resample (no variance)
        if np.ptp(x[idx]) == 0 or np.ptp(y[idx]) == 0:
            rs[b] = np.nan
        else:
            rs[b] = fn(x[idx], y[idx])[0]
    rs = rs[~np.isnan(rs)]
    return float(np.percentile(rs, 2.5)), float(np.percentile(rs, 97.5))


def loso_influence(x, y, method):
    """Leave-one-subject-out range of r -- flags single-subject leverage."""
    fn = spearmanr if method == "spearman" else pearsonr
    rs = [fn(np.delete(x, i), np.delete(y, i))[0] for i in range(len(x))]
    return float(np.min(rs)), float(np.max(rs))


# ==============================================================================
# MAIN
# ==============================================================================
def main():
    neural, n_win = load_neural()
    behav, comps = behavioral_interaction()

    common = sorted(set(neural) & set(behav))
    only_neural = sorted(set(neural) - set(behav))
    only_behav  = sorted(set(behav) - set(neural))
    if only_neural:
        print(f"in neural but not behavioral (dropped): {only_neural}")
    if only_behav:
        print(f"in behavioral but not neural (dropped): {only_behav}")
    if len(common) < 3:
        print("Too few shared subjects."); return

    x = np.array([neural[s] for s in common])   # neural AUC
    y = np.array([behav[s]  for s in common])   # behavioral interaction (ms)

    # per-subject table
    tab = pd.DataFrame({"sbjId": common, "neural_auc": x, "behav_interaction_ms": y})
    tab = tab.merge(comps.reset_index().rename(columns={"index": "sbjId"}),
                    on="sbjId", how="left")
    tab.to_csv(OUT_DIR / "brain_behavior.csv", index=False)

    rho, p_s = spearmanr(x, y)
    r, p_p   = pearsonr(x, y)
    ci_s = bootstrap_ci(x, y, "spearman", N_BOOTSTRAP, RANDOM_SEED)
    ci_p = bootstrap_ci(x, y, "pearson",  N_BOOTSTRAP, RANDOM_SEED)
    loso_s = loso_influence(x, y, "spearman")

    print(f"\nBrain-behavior correlation  (N = {len(common)}, "
          f"neural window {WIN_START_S*1000:.0f}-{WIN_END_S*1000:.0f} ms, "
          f"{n_win} samples)")
    print(f"  Spearman rho = {rho:.3f}, p = {p_s:.4f}, 95% CI [{ci_s[0]:.3f}, {ci_s[1]:.3f}]")
    print(f"  Pearson  r   = {r:.3f}, p = {p_p:.4f}, 95% CI [{ci_p[0]:.3f}, {ci_p[1]:.3f}]")
    print(f"  leave-one-subject-out rho range: [{loso_s[0]:.3f}, {loso_s[1]:.3f}]")
    print(f"  behavioral interaction: M = {y.mean():.1f} ms, SD = {y.std(ddof=1):.1f}")
    if (ci_s[0] > 0) == (ci_s[1] > 0):
        print("  -> CI excludes zero: neural decoding tracks the behavioral effect.")
    else:
        print("  -> CI includes zero: no reliable brain-behavior link at this window.")

    # ---- figure ----
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    ax.scatter(x, y, s=42, color="#2c7fb8", alpha=0.8, edgecolor="white", lw=0.6, zorder=3)
    # OLS fit line + bootstrap band (for visualization only)
    b1, b0 = np.polyfit(x, y, 1)
    xs = np.linspace(x.min(), x.max(), 100)
    ax.plot(xs, b0 + b1*xs, color="#d6604d", lw=2, zorder=2)
    ax.axhline(0, color="#bbb", lw=0.8, ls=":")
    ax.axvline(0.5, color="#bbb", lw=0.8, ls=":")
    ax.set_xlabel(f"Neural decoding (mean AUC, {WIN_START_S*1000:.0f}-{WIN_END_S*1000:.0f} ms)",
                  fontsize=10)
    ax.set_ylabel("Behavioral reward-by-transition\ninteraction (ms)", fontsize=10)
    ax.set_title(f"Stronger decoding predicts a larger behavioral effect\n"
                 f"Spearman \u03c1 = {rho:.2f} (p = {p_s:.3f}), N = {len(common)}",
                 fontsize=10.5)
    ax.tick_params(labelsize=9)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "brain_behavior.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {OUT_DIR/'brain_behavior.png'}")

    # ---- provenance ----
    with open(OUT_DIR / "brain_behavior.json", "w") as f:
        json.dump({
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "n_subjects": len(common), "subject_ids": common,
            "dropped_only_neural": only_neural, "dropped_only_behavioral": only_behav,
            "neural_window_s": [WIN_START_S, WIN_END_S], "neural_window_samples": n_win,
            "neural_source": "cv_group_scores.npy (saved Model A per-subject curves)",
            "behavioral_measure": "reward-by-transition RT interaction "
                                  "(swCond switch-cost minus rpCond switch-cost), "
                                  "inducer trials, RT_exclude==0",
            "spearman": {"rho": float(rho), "p": float(p_s), "ci95": ci_s,
                         "loso_range": loso_s},
            "pearson":  {"r": float(r), "p": float(p_p), "ci95": ci_p},
            "behavioral_interaction_ms": {"mean": float(y.mean()),
                                          "sd": float(y.std(ddof=1))},
            "n_bootstrap": N_BOOTSTRAP, "random_seed": RANDOM_SEED,
            "versions": {"python": platform.python_version(),
                         "numpy": np.__version__, "pandas": pd.__version__},
        }, f, indent=2)
    print(f"  wrote {OUT_DIR/'brain_behavior.json'}\nDone.")


if __name__ == "__main__":
    main()
