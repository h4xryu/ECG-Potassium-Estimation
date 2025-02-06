from .dataset.Severance import process_peaks_crop_PQRST,SignalCollector
import numpy as np
from tqdm import tqdm

def one_cycle(ecg,spl,pat, fs=500):
    ecg_collected = []
    spl_collected = []
    pat_collected = []
    for i in tqdm(range(0, len(ecg)), total=len(ecg), desc="Cropping"):
        opt = 0
        ecg_signal = ecg[i]  # ecg[i]가 유효한 데이터인지 확인 필요

        # ecg[i], opt = process_peaks_crop_old(ecg_signal)
        ecg[i], opt = process_peaks_crop_PQRST(ecg_signal, sampling_rate=fs)


        ecg_collected.append(ecg[i])

        spl_collected.append(spl[i])
        pat_collected.append(pat[i])

    ecg = ecg_collected
    spl = spl_collected
    pat = pat_collected
    return ecg,spl,pat

def signal_collector(ecg,spl,pat):
    selected_ecg, selected_pat, selected_spl = [], [], []

    collector = SignalCollector()
    for i in range(len(ecg)):
        selected_ecg, selected_pat, selected_spl = collector.plot_signal(i, ecg, pat, spl)
        if selected_ecg is not None:
            print(f"Selected ECG for Patient: {selected_pat}, Potassium Level: {selected_spl}")

    print("\nFinal Selected Data:")
    for i in range(len(collector.selected_signals)):
        print(
            f"Patient ID: {collector.selected_patient_ids[i]}, Potassium Level: {collector.selected_potassium_levels[i]}")
    collector.save_temp_data()
    ecg = selected_ecg
    pat = selected_pat
    spl = selected_pat
    return ecg, spl, pat
