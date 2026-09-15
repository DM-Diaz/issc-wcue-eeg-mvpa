"""
================================================================================
extract_patterns_v3.py -- Haufe forward patterns, electrode reliability,
                          and an INDEPENDENT train-vs-test pattern comparison
ISSC_wCue: Model A, Model A-neutral (train blocks), and block 3 (test block)
================================================================================

WHAT CHANGED FROM v2
--------------------
v2 correlated the Model A pattern with the A-neutral pattern and found r = 0.92.
That number is inflated: Model A is fit on ALL inducer trials and A-neutral on
blocks 1/2/4/5, so the two share roughly 62-68% of their training trials. A high
correlation is close to what the overlap alone produces. It is a useful internal
consistency check, but it is NOT independent evidence that the same signature
underlies both results.

v3 adds a third pattern fit on BLOCK 3 ONLY. Blocks 1/2/4/5 and block 3 are
DISJOINT sets of trials, so:

    corr( A_neutral pattern , A_block3 pattern )

is a genuine independent test. The A-neutral decode trains in the entangled
blocks and tests in the neutral block; if the discriminating pattern present in
block 3 matches the pattern learned in the training blocks, that is a direct
mechanistic explanation of WHY the decoder generalized. This is the headline
comparison in v3; the A vs A-neutral correlation is retained but explicitly
labelled as overlapping/internal-consistency.

Note on expectation: the block-3 pattern is estimated from far fewer trials
(~34 per class, i.e. ~8 pseudotrials per class), so it is noisy and the
correlation will be ATTENUATED relative to the overlapping comparison. Expect a
substantially lower r than 0.92. Anything reliably above zero is the meaningful
result; do not read a smaller r as a weaker finding.

Also in v3: subjects are tracked PER ANALYSIS rather than requiring all three to
succeed, so a subject too thin for the block-3 pattern no longer drops out of the
A and A-neutral results. Correlations use the intersection of the relevant
subject sets.

Retained from v2:
 * per-subject L2 scale normalization (Haufe patterns are defined only up to
   scale; magnitude tracks each subject's covariance, not the effect)
 * electrode reliability via one-sample cluster permutation over the sensor
   adjacency graph, two-sided (a channel may be reliably swCond- or rpCond-like)
 * colorbars and an explicit sign convention (y = 1 is swCond, so POSITIVE
   pattern values are swCond-like)

Outputs (mvpa_results/patterns/):
  patterns_<name>_raw.npy / _norm.npy / _stats.npz
  patterns_info.json
  topo_<name>.png                  per-analysis topography, reliable channels circled
  topo_comparison.png              three topographies + the independent scatter
================================================================================
"""

from pathlib import Path
import json, platform
from datetime import datetime, timezone
import numpy as np
import mne, sklearn
from mne.stats import permutation_cluster_1samp_test
from scipy.stats import t as t_dist, pearsonr, ttest_1samp
from sklearn.base import clone
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm

from mvpa_io import load_subject, discover_subjects, ROOT

# ---- output ----
OUT_DIR = ROOT / "mvpa_results" / "patterns"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---- config (matched to the decoding scripts) ----
KEEP_SUBJECTS    = None
WIN_START_S      = 0.150     # peak of the Model A cluster (peak AUC 0.591 @ 200 ms)
WIN_END_S        = 0.250
N_AVG            = 4
N_REPEATS        = 100
C_REG            = 1.0
RANDOM_SEED      = 42
MIN_PT_PER_CLASS = 4

TRAIN_BLOCKS     = [1, 2, 4, 5]   # A-neutral training blocks
TEST_BLOCK       = 3              # the neutral block A-neutral generalizes TO
RP_CODE, SW_CODE = 1, 3           # y = 1 for swCond -> POSITIVE pattern = swCond-like

# ---- reliability statistics ----
CLUSTER_ALPHA    = 0.05
CLUSTER_TAIL     = 0        # two-sided
N_PERMUTATIONS   = 10000

ANALYSES = ("A", "A_neutral", "A_block3")
PRETTY = {"A":         "Model A\n(all inducer blocks)",
          "A_neutral": f"A-neutral train\n(blocks {TRAIN_BLOCKS})",
          "A_block3":  f"Block {TEST_BLOCK} only\n(the neutral test block)"}


def make_clf():
    return make_pipeline(StandardScaler(),
                         LogisticRegression(C=C_REG, max_iter=1000, solver="liblinear"))


def make_pseudotrials_2d(X2d, y, n_avg, rng):
    """Average random groups of n_avg same-class rows. X2d is (n_trials, n_channels)."""
    Xp, yp = [], []
    for cls in np.unique(y):
        idx = np.where(y == cls)[0]
        rng.shuffle(idx)
        for g in range(len(idx) // n_avg):
            Xp.append(X2d[idx[g*n_avg:(g+1)*n_avg]].mean(0))
            yp.append(cls)
    if not Xp:
        return np.empty((0, X2d.shape[1])), np.empty((0,), int)
    return np.stack(Xp), np.asarray(yp, int)


def haufe_pattern(clf, Xp):
    """Forward pattern A = Cov(Xs) @ w in the scaled space the linear model sees.
    Interpretable as source strength; raw weights are not."""
    scaler = clf.named_steps["standardscaler"]
    lin    = clf.named_steps["logisticregression"]
    Xs   = scaler.transform(Xp)
    w    = lin.coef_.ravel()
    cov  = np.cov(Xs, rowvar=False)
    return cov @ w


def subject_pattern(Xwin, y, rng):
    """Mean Haufe pattern over N_REPEATS pseudotrial re-draws, or None if too thin."""
    if len(y) == 0 or np.bincount(y, minlength=2).min() < N_AVG * MIN_PT_PER_CLASS:
        return None
    pats = []
    for _ in range(N_REPEATS):
        Xp, yp = make_pseudotrials_2d(Xwin, y, N_AVG, rng)
        if len(np.unique(yp)) < 2:
            continue
        pats.append(haufe_pattern(clone(make_clf()).fit(Xp, yp), Xp))
    return np.mean(pats, axis=0) if pats else None


def l2_normalize(p):
    """Scale-normalize so subjects contribute equally in SHAPE, not amplitude."""
    n = np.linalg.norm(p)
    return p / n if n > 0 else p


# ==============================================================================
# RELIABILITY
# ==============================================================================
def channel_reliability(patterns, info):
    """patterns: (n_subjects, n_channels), scale-normalized. Tests each channel
    against 0 with cluster correction over the sensor adjacency graph."""
    adjacency, adj_names = mne.channels.find_ch_adjacency(info, ch_type="eeg")
    if len(adj_names) != patterns.shape[1]:
        raise ValueError(f"adjacency has {len(adj_names)} channels but patterns have "
                         f"{patterns.shape[1]}; check the montage.")
    n = patterns.shape[0]
    thr = t_dist.ppf(1 - CLUSTER_ALPHA / 2, df=n - 1)      # two-sided
    t_obs, clusters, pv, _ = permutation_cluster_1samp_test(
        patterns, adjacency=adjacency, threshold=thr, tail=CLUSTER_TAIL,
        n_permutations=N_PERMUTATIONS, seed=RANDOM_SEED, out_type="mask",
        verbose=False)
    sig = np.zeros(patterns.shape[1], bool)
    for c, p in zip(clusters, pv):
        if p < CLUSTER_ALPHA:
            m = np.asarray(c)
            sig |= (m.astype(bool) if m.dtype == bool
                    else np.isin(np.arange(len(sig)), m))
    return t_obs, np.asarray(pv), sig


# ==============================================================================
# PATTERN COMPARISON
# ==============================================================================
def _align(patterns, ids, common):
    idx = {int(s): i for i, s in enumerate(ids)}
    return np.stack([patterns[idx[int(s)]] for s in common])


def compare_patterns(res_a, res_b):
    """Spatial correlation between two analyses, on their common subjects.
    Returns group-mean r plus a per-subject test (Fisher z, one-sample t)."""
    common = np.intersect1d(res_a["ids"], res_b["ids"])
    if len(common) < 3:
        return None
    A = _align(res_a["norm"], res_a["ids"], common)
    B = _align(res_b["norm"], res_b["ids"], common)
    r_group = pearsonr(A.mean(0), B.mean(0))[0]
    r_subj  = np.array([pearsonr(A[i], B[i])[0] for i in range(len(common))])
    z       = np.arctanh(np.clip(r_subj, -0.999, 0.999))
    t, p    = ttest_1samp(z, 0)
    return dict(n=int(len(common)), r_group=float(r_group),
                r_subj_mean=float(r_subj.mean()), r_subj_sd=float(r_subj.std(ddof=1)),
                t=float(t), p=float(p), r_subj=r_subj,
                A_mean=A.mean(0), B_mean=B.mean(0))


# ==============================================================================
# PLOTTING
# ==============================================================================
MASK_PARAMS = dict(marker="o", markerfacecolor="w", markeredgecolor="k",
                   linewidth=0, markersize=4)


def plot_topo(pattern, info, sig_mask, title, fname):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(4.4, 4.4))
    vmax = np.max(np.abs(pattern))
    im, _ = mne.viz.plot_topomap(pattern, info, axes=ax, show=False, cmap="RdBu_r",
                                 vlim=(-vmax, vmax), mask=sig_mask,
                                 mask_params=MASK_PARAMS, contours=6)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.06)
    cbar.set_label("Haufe pattern (a.u.)", fontsize=8); cbar.ax.tick_params(labelsize=7)
    ax.set_title(title, fontsize=9.5)
    fig.tight_layout(); fig.savefig(fname, dpi=200, bbox_inches="tight"); plt.close(fig)
    print(f"  wrote {fname}")


def plot_comparison(results, info, comp_indep, fname):
    """Three topographies plus the INDEPENDENT (disjoint-data) scatter."""
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 4, figsize=(15.5, 4.0))

    for ax, name in zip(axes[:3], ANALYSES):
        if name not in results:
            ax.axis("off"); continue
        pat = results[name]["norm"].mean(0)
        vmax = np.max(np.abs(pat))
        im, _ = mne.viz.plot_topomap(pat, info, axes=ax, show=False, cmap="RdBu_r",
                                     vlim=(-vmax, vmax), mask=results[name]["sig"],
                                     mask_params=MASK_PARAMS, contours=6)
        ax.set_title(f"{PRETTY[name]}\nN = {len(results[name]['ids'])}", fontsize=8.8)
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.05)
        cb.ax.tick_params(labelsize=6.5)

    ax = axes[3]
    if comp_indep is None:
        ax.axis("off")
    else:
        a, b = comp_indep["A_mean"], comp_indep["B_mean"]
        ax.scatter(a, b, s=16, color="#2c7fb8", alpha=0.75, edgecolor="none")
        lim = np.max(np.abs(np.r_[a, b])) * 1.1
        ax.plot([-lim, lim], [-lim, lim], ls="--", lw=1, color="#888")
        ax.axhline(0, color="#ccc", lw=0.7); ax.axvline(0, color="#ccc", lw=0.7)
        ax.set(xlim=(-lim, lim), ylim=(-lim, lim),
               xlabel=f"train blocks {TRAIN_BLOCKS} pattern",
               ylabel=f"block {TEST_BLOCK} pattern")
        ax.set_title("INDEPENDENT comparison (disjoint trials)\n"
                     f"r = {comp_indep['r_group']:.3f}   |   per-subject "
                     f"mean r = {comp_indep['r_subj_mean']:.3f}, p = {comp_indep['p']:.1e}",
                     fontsize=8.5)
        ax.tick_params(labelsize=7)
        for lbl in (ax.xaxis.label, ax.yaxis.label): lbl.set_fontsize(8)

    fig.suptitle(f"Haufe forward patterns, {WIN_START_S*1000:.0f}-{WIN_END_S*1000:.0f} ms   |   "
                 f"positive = swCond-like, negative = rpCond-like   |   "
                 f"circled electrodes reliable across subjects (p < {CLUSTER_ALPHA}, cluster-corrected)",
                 fontsize=9, y=1.03)
    fig.tight_layout(); fig.savefig(fname, dpi=200, bbox_inches="tight"); plt.close(fig)
    print(f"  wrote {fname}")


# ==============================================================================
# MAIN
# ==============================================================================
def main():
    mne.set_log_level("WARNING")
    subs = discover_subjects(KEEP_SUBJECTS)
    print(f"Found {len(subs)} subjects. Window {WIN_START_S*1000:.0f}-"
          f"{WIN_END_S*1000:.0f} ms.\n")

    raw = {k: [] for k in ANALYSES}
    ids = {k: [] for k in ANALYSES}          # per-analysis subject tracking
    info, ch_names, missing = None, None, []

    for sid in tqdm(subs, desc="Subjects", unit="subj"):
        try:
            epochs, meta = load_subject(sid)
        except FileNotFoundError as exc:
            missing.append(sid); tqdm.write(f"  S{sid}: missing -> skipped\n  {exc}")
            continue

        if info is None:
            if epochs.get_montage() is None:          # topoplot needs sensor positions
                epochs.set_montage("standard_1020", match_case=False, on_missing="warn")
            info, ch_names = epochs.info, epochs.ch_names

        times = epochs.times
        tmask = (times >= WIN_START_S) & (times <= WIN_END_S)
        Xwin_all = epochs.get_data()[:, :, tmask].mean(axis=2)     # (n_ep, n_ch)

        is_ind = meta.itemType.values == "inducer"
        code   = meta.triggerCode1.values
        blk    = meta.blockId.values
        base   = is_ind & np.isin(code, (RP_CODE, SW_CODE))

        sel = {"A":         base,
               "A_neutral": base & np.isin(blk, TRAIN_BLOCKS),
               "A_block3":  base & (blk == TEST_BLOCK)}

        rng = np.random.default_rng([RANDOM_SEED, int(sid), 3])
        for name, m in sel.items():
            y = (code[m] == SW_CODE).astype(int)        # 1 = swCond
            p = subject_pattern(Xwin_all[m], y, rng)
            if p is None:
                tqdm.write(f"  S{sid}: too few trials for '{name}' -> excluded from that analysis")
                continue
            raw[name].append(p); ids[name].append(sid)

    if not ids["A"]:
        print("No subjects yielded patterns."); return

    # ---- per-analysis: normalize, test reliability, plot ----
    results = {}
    for name in ANALYSES:
        if not ids[name]:
            print(f"\n[{name}] no subjects -> skipped"); continue
        raw_arr  = np.stack(raw[name])
        norm_arr = np.stack([l2_normalize(p) for p in raw_arr])
        sid_arr  = np.asarray(ids[name], int)
        np.save(OUT_DIR / f"patterns_{name}_raw.npy",  raw_arr)
        np.save(OUT_DIR / f"patterns_{name}_norm.npy", norm_arr)

        t_obs, pv, sig = channel_reliability(norm_arr, info)
        np.savez_compressed(OUT_DIR / f"patterns_{name}_stats.npz",
                            t_obs=t_obs, cluster_pvals=pv, sig_mask=sig,
                            ch_names=np.array(ch_names), subject_ids=sid_arr)
        results[name] = dict(raw=raw_arr, norm=norm_arr, ids=sid_arr,
                             t=t_obs, pv=pv, sig=sig)

        mean_pat = norm_arr.mean(0)
        n_sig = int(sig.sum())
        print(f"\n[{name}]  N = {len(sid_arr)}   reliable electrodes: "
              f"{n_sig}/{len(ch_names)}  (two-sided cluster test, p < {CLUSTER_ALPHA})")
        if n_sig:
            pos = [c for i, c in enumerate(ch_names) if sig[i] and mean_pat[i] > 0]
            neg = [c for i, c in enumerate(ch_names) if sig[i] and mean_pat[i] < 0]
            print(f"   swCond-like (+): {', '.join(pos) if pos else 'none'}")
            print(f"   rpCond-like (-): {', '.join(neg) if neg else 'none'}")
        else:
            print("   none survived correction (the pattern may still be consistent in "
                  "shape; see the group map and the correlations below)")

        plot_topo(mean_pat, info, sig,
                  f"{PRETTY[name].replace(chr(10), ' ')}\n"
                  f"{WIN_START_S*1000:.0f}-{WIN_END_S*1000:.0f} ms, N={len(sid_arr)}  |  "
                  f"positive = swCond-like\ncircled = reliable (p<{CLUSTER_ALPHA})",
                  OUT_DIR / f"topo_{name}.png")

    # ---- comparisons ----
    print("\n" + "="*74)
    comp = {}

    # (1) THE INDEPENDENT TEST: disjoint trials
    if "A_neutral" in results and "A_block3" in results:
        c = compare_patterns(results["A_neutral"], results["A_block3"])
        comp["A_neutral_vs_block3"] = c
        if c:
            print(f"INDEPENDENT  train blocks {TRAIN_BLOCKS}  vs  block {TEST_BLOCK}   "
                  f"(DISJOINT trials, N = {c['n']})")
            print(f"   group-mean pattern r = {c['r_group']:.3f}")
            print(f"   per-subject r: mean = {c['r_subj_mean']:.3f} "
                  f"(SD {c['r_subj_sd']:.3f}), t({c['n']-1}) = {c['t']:.2f}, p = {c['p']:.2e}")
            print("   -> a reliably positive r means the discriminating pattern present in the")
            print("      neutral block is the SAME pattern learned in the entangled blocks,")
            print("      which is the mechanistic explanation for why the decoder generalized.")
            print("   NOTE: the block-3 pattern rests on ~8 pseudotrials/class, so this r is")
            print("      attenuated by measurement noise. A smaller r than the overlapping")
            print("      comparison below is EXPECTED and is not a weaker result.")

    # (2) overlapping consistency check (retained, explicitly labelled)
    if "A" in results and "A_neutral" in results:
        c = compare_patterns(results["A"], results["A_neutral"])
        comp["A_vs_A_neutral"] = c
        if c:
            print(f"\nOVERLAPPING  Model A  vs  A-neutral train blocks   (N = {c['n']})")
            print(f"   group-mean pattern r = {c['r_group']:.3f}, "
                  f"per-subject mean r = {c['r_subj_mean']:.3f}")
            print("   NOTE: these two are fit on ~62-68% of the SAME trials, so a high r is")
            print("      largely produced by the overlap. Report as internal consistency,")
            print("      NOT as independent convergence.")
    print("="*74)

    plot_comparison(results, info, comp.get("A_neutral_vs_block3"),
                    OUT_DIR / "topo_comparison.png")

    # ---- provenance ----
    def _ser(c):
        if c is None: return None
        return {k: v for k, v in c.items() if k not in ("r_subj", "A_mean", "B_mean")}

    with open(OUT_DIR / "patterns_info.json", "w") as f:
        json.dump({
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "analyses": {
                "A": "axis from all inducer trials (matches model_a_decode)",
                "A_neutral": f"axis from TRAIN_BLOCKS {TRAIN_BLOCKS} inducers "
                             f"(the axis that generalized to block {TEST_BLOCK})",
                "A_block3": f"axis fit within block {TEST_BLOCK} only (the neutral test "
                            f"block); DISJOINT from A_neutral, enabling an independent "
                            f"train-vs-test pattern comparison",
            },
            "sign_convention": "y=1 is swCond, so positive pattern values are swCond-like",
            "scale_normalization": "per-subject L2 norm before averaging/statistics",
            "window_s": [WIN_START_S, WIN_END_S],
            "channels": ch_names,
            "subject_ids_per_analysis": {k: results[k]["ids"].tolist()
                                         for k in results},
            "n_subjects_per_analysis": {k: int(len(results[k]["ids"])) for k in results},
            "subjects_missing_data": missing,
            "params": {"n_avg": N_AVG, "n_repeats": N_REPEATS, "C_reg": C_REG,
                       "random_seed": RANDOM_SEED, "min_pt_per_class": MIN_PT_PER_CLASS},
            "reliability_stats": {"test": "one-sample cluster permutation over sensor adjacency",
                                  "tail": CLUSTER_TAIL, "alpha": CLUSTER_ALPHA,
                                  "n_permutations": N_PERMUTATIONS},
            "pattern_comparisons": {
                "A_neutral_vs_block3": {**(_ser(comp.get("A_neutral_vs_block3")) or {}),
                                        "data_overlap": "none (disjoint trials)",
                                        "role": "independent test"},
                "A_vs_A_neutral": {**(_ser(comp.get("A_vs_A_neutral")) or {}),
                                   "data_overlap": "~62-68% shared trials",
                                   "role": "internal consistency only"},
            },
            "versions": {"python": platform.python_version(), "numpy": np.__version__,
                         "mne": mne.__version__, "sklearn": sklearn.__version__},
        }, f, indent=2)

    print(f"\nDone. Outputs in {OUT_DIR}")


if __name__ == "__main__":
    main()
