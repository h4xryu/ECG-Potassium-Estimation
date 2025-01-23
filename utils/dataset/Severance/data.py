import os
import csv
import numpy as np 
from .manage_dirs import *
from numpy.typing import NDArray
from .preprocessing import *
from tqdm import tqdm

def save_npy_data(npy_path, arr):
    np.save(npy_path, np.array(arr))


def preprocess_datas(arr, samp_rate=150):
    arr = remove_baseline_wander(np.array(arr))
    arr = highpass_filter(arr, 150)
    arr = bandstop_filter(arr, 150)

    # r_peaks = detect_qrs(arr, 150)
    # arr = extract_one_cycle(arr, r_peaks, 150)
    return arr

@DirectoryProcess
def generate_datas(root="./dataset/"):
    '''
    :param root: 데이터 셋이 있는 디렉토리 입력
    :return: csv로부터 넘파이 및 텍스트 파일 생성
    '''
    files = sorted([f for f in os.listdir(root) if f.endswith(".csv")])
    for file in tqdm(files, total=len(files), desc="Data generating"):
        path = os.path.join(root, file)
        txt_path = path[:-4]+".txt"
        npy_path = path[:-4]+".npy"
        with open(txt_path,'w') as reset:
            reset.write('')
            
        with open(path, 'r') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            arr = []
            pat = []
            for row in reader:

                result = []
                for x in row:
                    stripped = x.strip()
                    if stripped.count('.') > 1:
                        stripped = stripped.rsplit('.', 1)[0]

                    if stripped.isalnum():
                        if stripped != '' and stripped != '\t':
                            pat.append(stripped)
                            with open(txt_path,'a') as txt:
                                txt.write(stripped)
                                txt.write('\n')

                    else :
                        if stripped != '' and stripped != '\t':
                            try:
                                result.append(float(stripped))
                            except:
                                print("확장자문제 ",stripped)

                if result:
                    arr.append(result)

                # Find the max length of rows

                arr = [[value for value in series if value != 0] for series in arr]

                lengths = [len(series) for series in arr]
                min_length = min(lengths)
                arr = [series[:min_length] for series in arr]

        arr = preprocess_datas(arr)
        save_npy_data(npy_path, arr)
    return [],[]
        

@DirectoryProcess
def load_datas(root="./dataset/") -> (list, list):
    '''
    :param root: 데이터 셋이 있는 디렉토리 입력
    :return: 환자의 한 사이클 ecg 배열 반환 (QRS, T 정보)
    '''
    ECG_arr = []
    patient_arr = []
    tmp_ecg = []
    tmp_pat = []
    npy_files = sorted([f for f in os.listdir(root) if f.endswith(".npy")])
    txt_files = sorted([f for f in os.listdir(root) if f.endswith(".txt")])
    for npy, txt in zip(npy_files, txt_files):

        npy_path = os.path.join(root, npy)
        txt_path = os.path.join(root, txt)

        with open(txt_path, 'r') as t:
            lines = t.readlines()

        # \n 제거
        lines = [x.strip() for x in lines]
        arr = np.load(npy_path)

        for ECG, patient  in zip(arr, lines):
            tmp_pat.append(patient)
            tmp_ecg.append(ECG)
        patient_arr.append(tmp_pat)
        ECG_arr.append(tmp_ecg)
        tmp_ecg = []
        tmp_pat = []
    return ECG_arr, patient_arr



def data_sample_by_SPL(ecg,spl,pat):
    samples_1 = [(i, e, s, p) for i, (e, s, p) in enumerate(zip(ecg, spl, pat)) if s < 5.0]
    samples_2 = [(i, e, s, p) for i, (e, s, p) in enumerate(zip(ecg, spl, pat)) if 6.0 < s < 7.0]
    samples_3 = [(i, e, s, p) for i, (e, s, p) in enumerate(zip(ecg, spl, pat)) if 7.0 < s < 8.0]
    samples_4 = [(i, e, s, p) for i, (e, s, p) in enumerate(zip(ecg, spl, pat)) if 8.0 < s < 9.0]
    samples_5 = [(i, e, s, p) for i, (e, s, p) in enumerate(zip(ecg, spl, pat)) if s > 9.0]

    return samples_1, samples_2, samples_3, samples_4, samples_5

        