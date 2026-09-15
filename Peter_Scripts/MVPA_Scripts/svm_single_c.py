import argparse
import numpy as np
from tqdm import tqdm

import mne

from sklearn import svm
from sklearn.utils import resample
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV

mne.set_log_level("ERROR")


class EEG_Class:
    pass


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
    data = [getattr(eeg, f"{bin_name}_data") for bin_name in bin_names]
    return np.concatenate(data, axis=0)


def run_subject_svm(eeg, train_bins, test_bins, time_indexes, C_values):
    X_train = get_bin_data(eeg, train_bins)
    X_test = get_bin_data(eeg, test_bins)

    y_train = np.concatenate(
        [
            np.zeros(getattr(eeg, f"{train_bins[0]}_data").shape[0]),
            np.ones(getattr(eeg, f"{train_bins[1]}_data").shape[0]),
        ]
    )
    y_test = np.concatenate(
        [
            np.zeros(getattr(eeg, f"{test_bins[0]}_data").shape[0]),
            np.ones(getattr(eeg, f"{test_bins[1]}_data").shape[0]),
        ]
    )

    X_train, y_train = balance_classes(X_train, y_train)
    X_test, y_test = balance_classes(X_test, y_test)

    time_val_accs, time_test_accs, time_train_accs, time_weights, time_best_Cs = (
        run_svm_per_time(X_train, y_train, X_test, y_test, time_indexes, C_values)
    )
    return time_val_accs, time_test_accs, time_train_accs, time_weights, time_best_Cs


def run_svm(X_train, y_train, X_test, y_test, C_values):
    """
    Finds the best C for a given time point using cross-validation.
    Returns accuracies, weights, and the best C.
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    param_grid = {"C": C_values}
    clf = GridSearchCV(svm.LinearSVC(max_iter=10000), param_grid, cv=5)
    clf.fit(X_train_scaled, y_train)

    best_model = clf.best_estimator_
    train_acc = best_model.score(X_train_scaled, y_train)
    test_acc = best_model.score(X_test_scaled, y_test)
    val_acc = np.max(clf.cv_results_["mean_test_score"])
    weights = best_model.coef_[0]
    best_C = best_model.C

    return val_acc, test_acc, train_acc, weights, best_C


def run_svm_per_time(X_train, y_train, X_test, y_test, time_indexes, C_values):
    time_val_accs = []
    time_test_accs = []
    time_train_accs = []
    time_weights = []
    time_best_Cs = []

    for t in time_indexes:
        val_acc, test_acc, train_acc, weights, best_C = run_svm(
            X_train[:, :, t], y_train, X_test[:, :, t], y_test, C_values
        )

        time_val_accs.append(val_acc)
        time_test_accs.append(test_acc)
        time_train_accs.append(train_acc)
        time_weights.append(weights)
        time_best_Cs.append(best_C)

    return (
        np.array(time_val_accs),
        np.array(time_test_accs),
        np.array(time_train_accs),
        np.array(time_weights),
        np.array(time_best_Cs),
    )


def get_args():
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

    if E_num == 1:
        eegs = np.load("eegs_E1.npy", allow_pickle=True)
    elif E_num == 2:
        eegs = np.load(f"eegs_E2_{baseline_or_penalty}.npy", allow_pickle=True)

    train_bins = ["low", "high"]
    test_bins = ["low", "high"]

    eeg_val_accs = []
    eeg_test_accs = []
    eeg_train_accs = []
    eeg_weights = []
    eeg_best_Cs = []

    time_indexes = np.arange(0, 250, 1)
    # Grid of C values to search
    C_values = [0.0001, 0.001, 0.01, 0.1, 1]

    for i, eeg in tqdm(enumerate(eegs), total=len(eegs)):
        time_val_accs, time_test_accs, time_train_accs, time_weights, time_best_Cs = (
            run_subject_svm(eeg, train_bins, test_bins, time_indexes, C_values)
        )

        eeg_val_accs.append(time_val_accs)
        eeg_test_accs.append(time_test_accs)
        eeg_train_accs.append(time_train_accs)
        eeg_weights.append(time_weights)
        eeg_best_Cs.append(time_best_Cs)

    eeg_val_accs = np.array(eeg_val_accs)
    eeg_test_accs = np.array(eeg_test_accs)
    eeg_train_accs = np.array(eeg_train_accs)
    eeg_weights = np.array(eeg_weights)
    eeg_best_Cs = np.array(eeg_best_Cs)

    if E_num == 1:
        prefix = f"eegs_E{E_num}"
    elif E_num == 2:
        prefix = f"eegs_E{E_num}_{baseline_or_penalty}"

    #    np.save(
    #        f'../results/single_{prefix}_{"-".join(train_bins)}_{"-".join(test_bins)}_train_accs.npy',
    #        eeg_train_accs,
    #    )
    #    np.save(
    #        f'../results/single_{prefix}_{"-".join(train_bins)}_{"-".join(test_bins)}_val_accs.npy',
    #        eeg_val_accs,
    #    )
    #    np.save(
    #        f'../results/single_{prefix}_{"-".join(train_bins)}_{"-".join(test_bins)}_test_accs.npy',
    #        eeg_test_accs,
    #    )
    #    np.save(
    #        f'../results/single_{prefix}_{"-".join(train_bins)}_{"-".join(test_bins)}_weights.npy',
    #        eeg_weights,
    #    )
    np.save(
        f'../results/single_{prefix}_{"-".join(train_bins)}_{"-".join(test_bins)}_best_Cs.npy',
        eeg_best_Cs,
    )


if __name__ == "__main__":
    main()
