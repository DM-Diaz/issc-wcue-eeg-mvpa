import numpy as np

import mne

import matplotlib.pyplot as plt
from scipy.io import loadmat
from mne.viz import plot_topomap

from sklearn import svm
from sklearn.utils import resample
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from matplotlib.colors import TwoSlopeNorm, LinearSegmentedColormap
from scipy.stats import ttest_1samp
import pickle
from tqdm import tqdm

mne.set_log_level("ERROR")


class EEG_Class:
    pass


def get_args():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--E", help="E number", required=True)
    parser.add_argument(
        "--baseline_or_penalty",
        help="Select baseline or penalty if E = 2",
        required=False,
    )
    return parser.parse_args()


def main():
    args = get_args()
    E_num = int(args.E)
    baseline_or_penalty = args.baseline_or_penalty

    global eegs
    if E_num == 1:
        eegs = np.load("eegs_E1.npy", allow_pickle=True)
    elif E_num == 2:
        eegs = np.load(f"eegs_E2_{baseline_or_penalty}.npy", allow_pickle=True)
    else:
        raise ValueError("Unsupported E number: must be 1 or 2")

    train_accs, val_accs, test_accs, all_weights = run_experiment(
        train_bins=["low", "high"], test_bins=["low", "high"]
    )

    if E_num == 1:
        prefix = f"eegs_E{E_num}"
    elif E_num == 2:
        prefix = f"eegs_E{E_num}_{baseline_or_penalty}"

    np.save(f"../results/multi_{prefix}_train_accs.npy", train_accs)
    np.save(f"../results/multi_{prefix}_val_accs.npy", val_accs)
    np.save(f"../results/multi_{prefix}_test_accs.npy", test_accs)
    np.save(f"../results/multi_{prefix}_weights.npy", all_weights)


def run_svm(x_train, y_train, x_test, y_test):
    val_accs = []
    train_accs = []
    test_accs = []
    weights = []

    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    for train_index, val_index in kf.split(x_train, y_train):
        X_tr, X_val = x_train[train_index], x_train[val_index]
        y_tr, y_val = y_train[train_index], y_train[val_index]

        scaler = StandardScaler()
        X_tr_scaled = scaler.fit_transform(X_tr)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(x_test)

        clf = svm.LinearSVC()
        clf.fit(X_tr_scaled, y_tr)

        val_accs.append(np.mean(clf.predict(X_val_scaled) == y_val))
        train_accs.append(np.mean(clf.predict(X_tr_scaled) == y_tr))
        test_accs.append(np.mean(clf.predict(X_test_scaled) == y_test))
        weights.append(clf.coef_[0])

    return (
        np.mean(train_accs),
        np.mean(val_accs),
        np.mean(test_accs),
        np.mean(weights, axis=0),
    )


# Per-time SVM
def run_svm_per_time(X_train, y_train, X_test, y_test, time_indexes):
    train_accs = []
    val_accs = []
    test_accs = []
    weights = []

    for t in tqdm(time_indexes):
        x_train = X_train[:, :, t]
        x_test = X_test[:, :, t]

        train_acc, val_acc, test_acc, weight = run_svm(x_train, y_train, x_test, y_test)

        train_accs.append(train_acc)
        val_accs.append(val_acc)
        test_accs.append(test_acc)
        weights.append(weight)

    return (
        np.array(train_accs),
        np.array(val_accs),
        np.array(test_accs),
        np.array(weights),
    )


# Main loop
def run_experiment(train_bins, test_bins, kfolds=10):
    train_accs = []
    val_accs = []
    test_accs = []
    all_weights = []
    time_indexes = np.arange(0, 250)

    all_train_bin_data = {bin_name: [] for bin_name in train_bins}
    all_test_bin_data = {bin_name: [] for bin_name in test_bins}

    subject_numbers = []

    for eeg in eegs:
        subject_numbers.append(eeg.subject_number)

        for bin_name in train_bins:
            bin_data = getattr(eeg, f"{bin_name}_data")
            averaged_data = np.mean(bin_data, axis=0)[:, :250]
            all_train_bin_data[bin_name].append(averaged_data)

        for bin_name in test_bins:
            bin_data = getattr(eeg, f"{bin_name}_data")
            averaged_data = np.mean(bin_data, axis=0)[:, :250]
            all_test_bin_data[bin_name].append(averaged_data)

    all_train_bin_data = {k: np.array(v) for k, v in all_train_bin_data.items()}
    all_test_bin_data = {k: np.array(v) for k, v in all_test_bin_data.items()}

    n_subjects = len(subject_numbers)

    # Perform subject-wise KFold
    kf = KFold(n_splits=kfolds, shuffle=True, random_state=None)
    for train_subj_idx, test_subj_idx in kf.split(range(n_subjects)):
        test_mask = np.zeros(n_subjects, dtype=bool)
        test_mask[test_subj_idx] = True

        X_train = np.concatenate(
            [all_train_bin_data[bin_name][~test_mask] for bin_name in train_bins],
            axis=0,
        ).reshape(-1, 64, 250)
        y_train = np.concatenate(
            [
                np.full(X_train.shape[0] // len(train_bins), i)
                for i in range(len(train_bins))
            ]
        )

        X_test = np.concatenate(
            [all_test_bin_data[bin_name][test_mask] for bin_name in test_bins], axis=0
        ).reshape(-1, 64, 250)
        y_test = np.concatenate(
            [
                np.full(X_test.shape[0] // len(test_bins), i)
                for i in range(len(test_bins))
            ]
        )

        train_acc, val_acc, test_acc, weights = run_svm_per_time(
            X_train, y_train, X_test, y_test, time_indexes
        )

        train_accs.append(train_acc)
        val_accs.append(val_acc)
        test_accs.append(test_acc)
        all_weights.append(weights)

    train_accs = np.array(train_accs)
    val_accs = np.array(val_accs)
    test_accs = np.array(test_accs)
    all_weights = np.array(all_weights)

    return train_accs, val_accs, test_accs, all_weights


if __name__ == "__main__":
    main()
