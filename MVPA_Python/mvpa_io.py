"""
mvpa_io.py  --  shared loader for the ISSC_wCue MVPA models.

this loads an artifact-clean epoched .set produced by step3_createERP_mvpa.m and
attaches the per-epoch metadata CSV as epochs.metadata, so every model is a
column filter:

    Model A (rpCond vs swCond)      : itemType=='inducer', split on itSRProb (10 vs 90)
    Model A-neutral (train 1,2,4,5 / test block 3) : split on blockId
    Model B (diagnostic, by list)   : itemType=='diagnostic', split on episode (1 vs 2)
    Model C (leave-one-item-out)    : group by stimId within each condition

Note tht the metadata rows are aligned 1:1 with the saved epochs (the MATLAB export writes
one row per surviving epoch, in saved order, with an epoch_index column as a check).

Usage:
    from mvpa_io import load_subject
    epochs, meta = load_subject(3)
    inducers = epochs[meta.itemType == 'inducer']      # boolean indexing via metadata
"""

from pathlib import Path
import numpy as np
import pandas as pd
import mne

ROOT      = Path("/home/moon/Desktop/ISSC_wCue")
DATA_DIR  = ROOT / "preprocessed_mvpa"
META_DIR  = ROOT / "mvpa_results" / "metadata"

# item-cue tag in the MNE event label (e.g. 'B1,3(S1)/4'); (S2)=diagnostic here
TAG_RP   = "(S1)"    # rpCond  / lowSR
TAG_SW   = "(S3)"    # swCond  / highSR
TAG_DIAG = "(S2)"    # diagnostic / noR


def load_subject(sid, verify=True):
    """Return (epochs, meta_df) with metadata attached and row-aligned.

    Raises if the epoch count and metadata row count disagree, which would mean
    the MATLAB alignment and the saved set are out of sync.
    """
    set_path  = DATA_DIR / f"S{sid}_preprocessed.set"
    meta_path = META_DIR / f"S{sid}_metadata.csv"
    if not set_path.exists():
        raise FileNotFoundError(set_path)
    if not meta_path.exists():
        raise FileNotFoundError(f"{meta_path} (run step3_createERP_mvpa.m first)")

    epochs = mne.io.read_epochs_eeglab(set_path, verbose="ERROR")
    meta = pd.read_csv(meta_path)

    if len(epochs) != len(meta):
        raise ValueError(f"S{sid}: {len(epochs)} epochs but {len(meta)} metadata rows. "
                         f"The .set and metadata CSV are out of sync -- re-run the "
                         f"MATLAB export.")

    # cross-check: the class tag in each epoch's label must match the metadata's
    # triggerCode1 (1->S1, 2->S2, 3->S3). Catches any residual ordering bug.
    if verify:
        code_from_label = np.full(len(epochs), -1)
        inv = {v: k for k, v in epochs.event_id.items()}
        for i, ev_code in enumerate(epochs.events[:, 2]):
            lbl = inv[ev_code]
            if TAG_RP in lbl:   code_from_label[i] = 1
            elif TAG_DIAG in lbl: code_from_label[i] = 2
            elif TAG_SW in lbl: code_from_label[i] = 3
        mism = np.sum(code_from_label != meta["triggerCode1"].values)
        if mism:
            raise ValueError(f"S{sid}: {mism} epochs where the label's item code "
                             f"disagrees with metadata triggerCode1 -- alignment bug.")

    epochs.metadata = meta.reset_index(drop=True)
    return epochs, epochs.metadata


def get_XY(epochs, mask, label_col, pos_value):
    """Convenience: return (X, y, meta_subset) for a boolean mask over epochs.
      X : (n, n_ch, n_times)
      y : 1 where meta[label_col]==pos_value else 0
    """
    ep = epochs[mask]
    X = ep.get_data()
    y = (ep.metadata[label_col].values == pos_value).astype(int)
    return X, y, ep.metadata


def discover_subjects(keep=None):
    subs = []
    for f in sorted(DATA_DIR.glob("S*_preprocessed.set")):
        try:
            sid = int(f.stem.replace("S", "").replace("_preprocessed", ""))
        except ValueError:
            continue
        if keep is None or sid in keep:
            subs.append(sid)
    return subs


if __name__ == "__main__":
    # smoke test on the first available subject
    subs = discover_subjects()
    if not subs:
        print("No subjects in", DATA_DIR)
    else:
        sid = 1
        ep, m = load_subject(sid)
        print(f"S{sid}: {len(ep)} epochs, metadata cols: {list(m.columns)}")
        print("  itemType counts:\n", m.itemType.value_counts().to_string())
        print("  inducer itSRProb counts:\n",
              m[m.itemType == 'inducer'].itSRProb.value_counts().to_string())
        print("  diagnostic by episode (1=high/SC list, 2=low/RC list):\n",
              m[m.itemType == 'diagnostic'].episode.value_counts().to_string())
        print("  block 3 by triggerCode1:\n",
              m[m.blockId == 3].triggerCode1.value_counts().to_string())
