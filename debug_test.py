#!/usr/bin/env python3
"""Debug test for segment issue"""
import mne
import numpy as np

file_path = "/work/2024/tanzunsheng/ProcessedData/HBN_EEG/HBN_cmi_bids_NC/bids/derivatives/preprocessing/sub-NDARAA075AMK/eeg/sub-NDARAA075AMK_task-DespicableMe_eeg.vhdr"
raw = mne.io.read_raw_brainvision(file_path, preload=True, verbose="ERROR")

print("Before pick:")
print("  Channels:", len(raw.ch_names))
sfreq = raw.info["sfreq"]
print("  Sfreq:", sfreq)
print("  Samples:", raw.n_times)
print("  Duration:", round(raw.times[-1], 2), "s")

# Pick EEG channels
picks = mne.pick_types(raw.info, eeg=True, meg=True, exclude="bads")
print("\nPicks found:", len(picks))
raw.pick(picks)

print("\nAfter pick:")
print("  Channels:", len(raw.ch_names))
print("  Samples:", raw.n_times)

# Resample
raw.resample(256, verbose="ERROR")

print("\nAfter resample to 256 Hz:")
print("  Sfreq:", raw.info["sfreq"])
print("  Samples:", raw.n_times)
print("  Duration:", round(raw.times[-1], 2), "s")

# Get data
data = raw.get_data()
print("\nData shape:", data.shape)

# Test segmentation
segment_samples = 10 * 256  # 2560
print("Segment samples needed:", segment_samples)
print("Data has enough:", data.shape[1] >= segment_samples)
