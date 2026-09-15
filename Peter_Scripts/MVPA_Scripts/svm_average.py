import numpy as np
from tqdm import tqdm

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
    parser.add_argument(
        "--test",
        help="Test on baseline or penalty",
        required=False,
    )
    return parser.parse_args()


def main():
    args = get_args()
    E_num = int(args.E)
    baseline_or_penalty = args.baseline_or_penalty
    test = args.test

    train_bins = ["low", "high"]
    test_bins = ["low", "high"]
    use_kfold = True

    if E_num == 1:
        eegs = np.load("eegs_E1.npy", allow_pickle=True)
    elif E_num == 2:
        eegs = np.load(f"eegs_E2_{baseline_or_penalty}.npy", allow_pickle=True)
        if test != None:
            use_kfold = False
            test_eegs = np.load(f"eegs_E2_{test}.npy", allow_pickle=True)
            for i in range(len(eegs)):
                eeg = eegs[i]
                test_eeg = test_eegs[i]

                eeg.low_train_data = eeg.low_data
                eeg.high_train_data = eeg.high_data
                eeg.low_test_data = test_eeg.low_data
                eeg.high_test_data = test_eeg.high_data

            train_bins = ["low_train", "high_train"]
            test_bins = ["low_test", "high_test"]

    else:
        raise ValueError("Unsupported E number: must be 1 or 2")

    eeg_val_accs = []
    eeg_test_accs = []
    eeg_train_accs = []
    eeg_weights = []

    time_indexes = np.arange(0, 250, 1)
    for i, eeg in tqdm(enumerate(eegs), total=len(eegs)):
        time_val_accs, time_test_accs, time_train_accs, time_weights = run_subject_svm(
            eeg, train_bins, test_bins, time_indexes, use_kfold=use_kfold
        )

        eeg_val_accs.append(time_val_accs)
        eeg_test_accs.append(time_test_accs)
        eeg_train_accs.append(time_train_accs)
        eeg_weights.append(time_weights)

    eeg_val_accs = np.array(eeg_val_accs)
    eeg_test_accs = np.array(eeg_test_accs)
    eeg_train_accs = np.array(eeg_train_accs)
    eeg_weights = np.array(eeg_weights)

    if E_num == 1:
        prefix = f"eegs_E{E_num}"
    elif E_num == 2:
        prefix = f"eegs_E{E_num}_{baseline_or_penalty}"
        if test != None:
            prefix += test

    np.save(
        f'../results/average_{prefix}_{"_".join(train_bins)}_{"_".join(test_bins)}_train_accs.npy',
        eeg_train_accs,
    )
    np.save(
        f'../results/average_{prefix}_{"_".join(train_bins)}_{"_".join(test_bins)}_val_accs.npy',
        eeg_val_accs,
    )
    np.save(
        f'../results/average_{prefix}_{"_".join(train_bins)}_{"_".join(test_bins)}_test_accs.npy',
        eeg_test_accs,
    )
    np.save(
        f'../results/average_{prefix}_{"_".join(train_bins)}_{"_".join(test_bins)}_weights.npy',
        eeg_weights,
    )


def balance_classes(X, y):
    """Ensures an equal number of samples from each class by undersampling."""
    class_0 = np.where(y == 0)[0]
    class_1 = np.where(y == 1)[0]

    min_samples = min(len(class_0), len(class_1))

    class_0_balanced = resample(class_0, replace=False, n_samples=min_samples)
    class_1_balanced = resample(class_1, replace=False, n_samples=min_samples)

    balanced_indices = np.concatenate([class_0_balanced, class_1_balanced])
    np.random.shuffle(balanced_indices)

    return X[balanced_indices], y[balanced_indices]


def get_bin_data(eeg, bin_names):
    averaged_data = []
    y = []
    for i, bin_name in enumerate(bin_names):
        data = getattr(eeg, f"{bin_name}_data")
        n_trials = len(data)
        n_samples = (n_trials // 10) * 10
        resampled = resample(data, replace=False, n_samples=n_samples)
        reshaped = resampled.reshape(-1, 10, *data.shape[1:])
        averaged = np.mean(reshaped, axis=1)
        averaged_data.append(averaged)

        y.append([i] * averaged.shape[0])

    return np.concatenate(averaged_data, axis=0), np.concatenate(y, axis=0)


def run_subject_svm(eeg, train_bins, test_bins, time_indexes, use_kfold=True):
    X_train, y_train = get_bin_data(eeg, train_bins)
    X_test, y_test = get_bin_data(eeg, test_bins)

    time_val_accs, time_test_accs, time_train_accs, time_weights = run_svm_per_time(
        X_train, y_train, X_test, y_test, time_indexes, use_kfold=use_kfold
    )
    return time_val_accs, time_test_accs, time_train_accs, time_weights


def run_svm(X_train, y_train, X_test, y_test, use_kfold=True):
    """If use_kfold is True perform 10-fold CV on X_train; otherwise train once on full X_train and evaluate on X_test."""
    if use_kfold:
        val_accs = []
        test_accs = []
        train_accs = []
        weights = []

        kf = KFold(n_splits=10, shuffle=True, random_state=42)

        for train_index, val_index in kf.split(X_train):
            X_tr, X_val = X_train[train_index], X_train[val_index]
            y_tr, y_val = y_train[train_index], y_train[val_index]

            scaler = StandardScaler()
            X_tr_scaled = scaler.fit_transform(X_tr)
            X_val_scaled = scaler.transform(X_val)
            X_test_scaled = scaler.transform(X_test)

            clf = svm.LinearSVC(max_iter=5000)
            clf.fit(X_tr_scaled, y_tr)

            # Validation accuracy on fold
            val_acc = np.mean(clf.predict(X_val_scaled) == y_val)
            val_accs.append(val_acc)

            # Train accuracy on fold
            train_acc = np.mean(clf.predict(X_tr_scaled) == y_tr)
            train_accs.append(train_acc)

            # Accuracy on separate test set
            test_acc = np.mean(clf.predict(X_test_scaled) == y_test)
            test_accs.append(test_acc)

            weights.append(clf.coef_[0])

        return (
            np.mean(val_accs, axis=0),
            np.mean(test_accs, axis=0),
            np.mean(train_accs, axis=0),
            np.mean(weights, axis=0),
        )
    else:
        # Train once on the entire training set and test on the provided test set
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        clf = svm.LinearSVC(max_iter=5000)
        clf.fit(X_train_scaled, y_train)

        train_acc = np.mean(clf.predict(X_train_scaled) == y_train)
        test_acc = np.mean(clf.predict(X_test_scaled) == y_test)
        weights = clf.coef_[0]

        # No validation in no-CV mode; return np.nan for val_acc to preserve shape
        val_acc = np.nan
        return val_acc, test_acc, train_acc, weights


def run_svm_per_time(X_train, y_train, X_test, y_test, time_indexes, use_kfold=True):
    time_val_accs = []
    time_test_accs = []
    time_train_accs = []
    time_weights = []

    for t in time_indexes:
        val_accs, test_accs, train_accs, weights = run_svm(
            X_train[:, :, t], y_train, X_test[:, :, t], y_test, use_kfold=use_kfold
        )

        time_val_accs.append(val_accs)
        time_test_accs.append(test_accs)
        time_train_accs.append(train_accs)
        time_weights.append(weights)

    time_val_accs = np.array(time_val_accs)
    time_test_accs = np.array(time_test_accs)
    time_train_accs = np.array(time_train_accs)
    time_weights = np.array(time_weights)

    return time_val_accs, time_test_accs, time_train_accs, time_weights


if __name__ == "__main__":
    main()
