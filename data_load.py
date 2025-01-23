from torch.utils.data import Dataset, DataLoader
from utils.dataset.Severance import *
import torch

cls = {'A':0, 'B':1}
leads = {'v1':0,'v2':1,'v3':2,'v4':3,'v5':4,'v6':5}

def sel_ecg_data(ECG, PAT, c, l):
    return ECG[cls[c]][leads[l]], PAT[cls[c]][leads[l]]

def get_ecg_from_lead(root='./dataset', cls='A', lead='v1'):
    # Load data
    ecg, patients_from_signals = load_datas(root=root)  # ECG data and patient IDs
    spl, patients_from_spl = load_xlsx_datas(root=root)  # Additional data and patient IDs

    # Filter patients and data for the given class
    spl_filtered = [x for i, x in enumerate(spl) if cls in patients_from_spl[i]]
    patients_from_spl_filtered = [x for i, x in enumerate(patients_from_spl) if cls in patients_from_spl[i]]

    ecg_filtered, patients_from_signals_filtered = sel_ecg_data(ecg, patients_from_signals, cls, lead)

    # Ensure both lists contain valid entries only
    ecg_filtered = [x for i, x in enumerate(ecg_filtered) if cls in patients_from_signals_filtered[i]]
    patients_from_signals_filtered = [x for i, x in enumerate(patients_from_signals_filtered) if cls in patients_from_signals_filtered[i]]

    # Match patient IDs between the two datasets
    matched_ecg = []
    matched_patients = []
    matched_spl = []

    for i, patient in enumerate(patients_from_signals_filtered):
        if patient in patients_from_spl_filtered:
            matched_ecg.append(ecg_filtered[i])
            matched_patients.append(patient)
            spl_index = patients_from_spl_filtered.index(patient)
            matched_spl.append(spl_filtered[spl_index])

    return matched_ecg, matched_patients, matched_spl



class ECGDataset(Dataset):
    def __init__(self, signals, labels):
        self.signals = torch.tensor(signals, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.signals)

    def __getitem__(self, idx):
        return self.signals[idx], self.labels[idx]

def data_info():
    ecg_a, pat_a, spl_a = get_ecg_from_lead(root='./dataset/exp1', cls='A', lead='v1')
    ecg_b, pat_b, spl_b = get_ecg_from_lead(root='./dataset/exp1', cls='B', lead='v1')
    print(f'lead v1, A ECG: {len(ecg_a)}, SPL: {len(spl_a)}   ', end='')
    print(f'B ECG: {len(ecg_b)}, SPL: {len(spl_b)}')

    ecg_a, pat_a, spl_a = get_ecg_from_lead(root='./dataset/exp1', cls='A', lead='v2')
    ecg_b, pat_b, spl_b = get_ecg_from_lead(root='./dataset/exp1', cls='B', lead='v2')
    print(f'lead v2, A ECG: {len(ecg_a)}, SPL: {len(spl_a)}   ', end='')
    print(f'B ECG: {len(ecg_b)}, SPL: {len(spl_b)}')

    ecg_a, pat_a, spl_a = get_ecg_from_lead(root='./dataset/exp1', cls='A', lead='v3')
    ecg_b, pat_b, spl_b = get_ecg_from_lead(root='./dataset/exp1', cls='B', lead='v3')
    print(f'lead v3, A ECG: {len(ecg_a)}, SPL: {len(spl_a)}   ', end='')
    print(f'B ECG: {len(ecg_b)}, SPL: {len(spl_b)}')

    ecg_a, pat_a, spl_a = get_ecg_from_lead(root='./dataset/exp1', cls='A', lead='v4')
    ecg_b, pat_b, spl_b = get_ecg_from_lead(root='./dataset/exp1', cls='B', lead='v4')
    print(f'lead v4, A ECG: {len(ecg_a)}, SPL: {len(spl_a)}  ', end='')
    print(f'B ECG: {len(ecg_b)}, SPL: {len(spl_b)}')

    ecg_a, pat_a, spl_a = get_ecg_from_lead(root='./dataset/exp1', cls='A', lead='v5')
    ecg_b, pat_b, spl_b = get_ecg_from_lead(root='./dataset/exp1', cls='B', lead='v5')
    print(f'lead v5, A ECG: {len(ecg_a)}, SPL: {len(spl_a)}   ', end='')
    print(f'B ECG: {len(ecg_b)}, SPL: {len(spl_b)}')

    ecg_a, pat_a, spl_a = get_ecg_from_lead(root='./dataset/exp1', cls='A', lead='v6')
    ecg_b, pat_b, spl_b = get_ecg_from_lead(root='./dataset/exp1', cls='B', lead='v6')
    print(f'lead v6, A ECG: {len(ecg_a)}, SPL: {len(spl_a)}   ', end='')
    print(f'B ECG: {len(ecg_b)}, SPL: {len(spl_b)}')