import argparse
import re
import numpy as np
import mne

from scipy.io import loadmat
from tqdm import tqdm

mne.set_log_level("ERROR")

ACTUAL_ORDER = np.array(
    [
        "Fp1",
        "Fp2",
        "AF7",
        "AF3",
        "AFz",
        "AF4",
        "AF8",
        "F7",
        "F5",
        "F3",
        "F1",
        "Fz",
        "F2",
        "F4",
        "F6",
        "F8",
        "FT9",
        "FT7",
        "FT8",
        "FT10",
        "FC5",
        "FC3",
        "FC1",
        "FCz",
        "FC2",
        "FC4",
        "FC6",
        "T7",
        "T8",
        "C5",
        "C3",
        "C1",
        "Cz",
        "C2",
        "C4",
        "C6",
        "TP9",
        "TP7",
        "TP8",
        "TP10",
        "CP5",
        "CP3",
        "CP1",
        "CPz",
        "CP2",
        "CP4",
        "CP6",
        "P7",
        "P5",
        "P3",
        "P1",
        "Pz",
        "P2",
        "P4",
        "P6",
        "P8",
        "PO7",
        "PO3",
        "POz",
        "PO4",
        "PO8",
        "O1",
        "Oz",
        "O2",
    ]
)


class EEG_Class:
    def __init__(self, subject_number, E_num, low_bins, high_bins):
        self.subject_number = subject_number
        filename = f"/home/yuchin/Desktop/Sambashare/ISSP_mvpa/E{E_num}/preprocessed/S{subject_number}_preprocessed.set"
        data = mne.io.read_epochs_eeglab(filename)

        data.reorder_channels(ACTUAL_ORDER)

        # Get rid of bad data
        eeg = loadmat(filename, squeeze_me=True, struct_as_record=False)
        self.rejected_indices = np.where(eeg["reject"].rejmanual)[0]
        data.drop(self.rejected_indices)

        def collect(bin_num):
            pattern = re.compile(rf"^B.*{bin_num}(?=\()")
            return [v for k, v in data.event_id.items() if pattern.search(k)]

        low_codes = []
        for bin_num in low_bins:
            low_codes.extend(collect(bin_num))

        high_codes = []
        for bin_num in high_bins:
            high_codes.extend(collect(bin_num))

        self.low_events = np.isin(data.events[:, 2], low_codes)
        self.high_events = np.isin(data.events[:, 2], high_codes)

        self.low_data = data[self.low_events]._data
        self.high_data = data[self.high_events]._data

        self.data = data


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
        # fmt: off
        subject_list = [2, 3, 4, 6, 7, 8, 10, 11, 12, 15, 17, 18, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 42, 43, 44, 46, 47, 49, 52, 53, 54, 55, 56, 57, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 73, 74, 75, 76, 102, 502, 802]
        # fmt: on
        low_bins = [1, 2, 5, 6]
        high_bins = [3, 4, 7, 8]
    elif E_num == 2:
        # fmt: off
        subject_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 31, 32, 33, 35, 36, 37, 38, 39, 40, 111, 141, 142, 143, 144, 145, 146, 147, 148, 150, 151, 152, 153, 155, 156, 157, 162, 572]
        # fmt: on
        if baseline_or_penalty == "baseline":
            low_bins = [1, 2]
            high_bins = [3, 4]
        elif baseline_or_penalty == "penalty":
            low_bins = [5, 6]
            high_bins = [7, 8]
        else:
            raise Exception("E2 needs baseline or penalty")

    eegs = []
    for num in tqdm(subject_list):
        eeg = EEG_Class(num, E_num, low_bins, high_bins)
        if eeg.low_data.shape[0] <= 50 or eeg.high_data.shape[0] <= 50:
            print(num)
            continue

        eegs.append(eeg)

    if E_num == 1:
        filename = f"eegs_E{E_num}.npy"
    elif E_num == 2:
        filename = f"eegs_E{E_num}_{baseline_or_penalty}.npy"

    np.save(filename, eegs)


if __name__ == "__main__":
    main()
