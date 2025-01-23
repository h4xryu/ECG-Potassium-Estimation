import matplotlib.pyplot as plt
import pywt
import numpy as np
import scipy.signal as signal
import neurokit2 as nk
import pandas as pd
from nbformat.validator import isvalid
from tests.tests_ppg import sampling_rates
from scipy.signal import resample
from scipy.signal import butter, filtfilt, find_peaks
from .plot_sample import *

def zero_pad_to_length(data, target_length=500):
    """
    리스트 또는 NumPy 배열을 제로 패딩하여 지정된 길이로 맞춥니다.

    Args:
        data (list or np.ndarray): 원본 데이터.
        target_length (int): 목표 길이 (기본값: 150).

    Returns:
        np.ndarray: 제로 패딩된 데이터.
    """
    # NumPy 배열로 변환 (리스트도 지원)
    data = np.array(data)
    current_length = len(data)

    if current_length > target_length:
        # 데이터가 목표 길이를 초과하면 자르기
        return data[:target_length]
    elif current_length < target_length:
        # 데이터가 목표 길이보다 짧으면 패딩 추가
        padding_length = target_length - current_length
        return np.pad(data, (0, padding_length), 'constant', constant_values=0)
    else:
        # 이미 목표 길이면 그대로 반환
        return data


def remove_baseline_wander(ecg_signal, wavelet='db4', level=9):
    """
    Removes baseline wander noise from an ECG signal using DWT.

    Parameters:
        ecg_signal (array-like): The input ECG signal.
        wavelet (str): Wavelet type to use for DWT. Default is 'db4'.
        level (int): Decomposition level. Default is 9.

    Returns:
        np.ndarray: ECG signal with baseline wander removed.
    """
    # Perform DWT decomposition
    coeffs = pywt.wavedec(ecg_signal, wavelet, level=level)

    # Set the approximation coefficients (level 9) to zero
    coeffs[0] = np.zeros_like(coeffs[0])

    # Reconstruct the signal using the modified coefficients
    corrected_signal = pywt.waverec(coeffs, wavelet)

    return corrected_signal


def highpass_filter(ecg_signal, sampling_rate, cutoff_frequency=0.5):
    """
    Apply highpass filter to remove baseline wander.
    """
    nyquist_rate = sampling_rate / 2
    normal_cutoff = cutoff_frequency / nyquist_rate
    b, a = signal.butter(1, normal_cutoff, btype='highpass')
    filtered_signal = signal.filtfilt(b, a, ecg_signal)
    return filtered_signal

def bandstop_filter(ecg_signal, sampling_rate, cutoff_freqs=50):
    """
    Apply bandstop filter to remove powerline interference.
    """
    nyquist_rate = sampling_rate / 2
    low = (cutoff_freqs - 1) / nyquist_rate
    high = (cutoff_freqs + 1) / nyquist_rate
    b, a = signal.butter(2, [low, high], btype='bandstop')
    filtered_signal = signal.filtfilt(b, a, ecg_signal)
    return filtered_signal


def detect_qrs(ecg_signal, sampling_rate):
    """
    QRS detection using differentiation, squaring, and moving window integration.
    """
    diff_signal = np.diff(ecg_signal)
    squared_signal = diff_signal ** 2
    window_size = int(0.12 * sampling_rate)  # 120 ms window
    integrated_signal = np.convolve(squared_signal, np.ones(window_size), mode='same')
    peaks, _ = signal.find_peaks(integrated_signal, distance=int(0.6 * sampling_rate))  # 600 ms distance
    return peaks

def filter_peak_indices(peaks, signal_length, peak_name="Peak"):
    """
    유효한 피크 인덱스를 필터링하는 함수.

    Args:
        peaks (array-like): 피크 배열 (0과 1로 이루어진 배열).
        signal_length (int): 신호의 길이.
        peak_name (str): 디버깅용 피크 이름.

    Returns:
        array: 유효한 피크 인덱스.
    """
    peak_indices = np.where(peaks == 1)[0]  # 피크의 인덱스 추출
    peak_indices = peak_indices[peak_indices < signal_length]  # 유효한 범위로 제한
    # print(f"[DEBUG] {peak_name} Peaks: {len(peak_indices)} detected at indices: {peak_indices}")
    return peak_indices


def align_r_peak(signal_segment, sampling_rate=500, target_r_position=140, padded_length=512):
    """
    Align the R-peak to the target position by adding padding to the signal segment.

    Args:
        signal_segment (numpy.ndarray): Signal segment containing the R-peak.
        r_center (int): Current R-peak position in the signal segment.
        target_r_position (int): Desired R-peak position after alignment (default: 187).
        padded_length (int): Total length of the signal after padding (default: 374).

    Returns:
        numpy.ndarray: Processed signal with R-peak aligned at the target position.
    """
    # Calculate current R position relative to the start of the segment
    if len(signal_segment) > padded_length:
        return resample(signal_segment,padded_length)

    current_r_position = detect_qrs(signal_segment, sampling_rate=sampling_rate)[0]
    # print("Current R position:", current_r_position)

    left_padding = max(0, target_r_position - current_r_position)
    # Calculate the required padding
    # if current_r_position > target_r_position:
    #     signal_segment = signal_segment[current_r_position-target_r_position:]


    right_padding = max(0, padded_length - len(signal_segment) - left_padding)

    # print("Left padding:", left_padding)
    # print("Right padding:", right_padding)

    # Pad the signal segment
    processed_signal = np.pad(
        signal_segment,
        (left_padding, right_padding),
        'constant',
        constant_values=0
    )

    # Ensure the processed signal has the desired length
    if len(processed_signal) != padded_length:
        # print(f"Processed signal length mismatch: {len(processed_signal)} != {padded_length}")
        signal_segment = resample(signal_segment, padded_length // 2)
        current_r_position = detect_qrs(signal_segment, sampling_rate=sampling_rate)[0]
        # print("Current R position:", current_r_position)

        left_padding = max(0, target_r_position - current_r_position)
        # Calculate the required padding
        # if current_r_position > target_r_position:
        #     signal_segment = signal_segment[current_r_position-target_r_position:]

        right_padding = max(0, padded_length - len(signal_segment) - left_padding)

        # print("Left padding:", left_padding)
        # print("Right padding:", right_padding)

        # Pad the signal segment
        processed_signal = np.pad(
            signal_segment,
            (left_padding, right_padding),
            'constant',
            constant_values=0
        )
        return processed_signal

    return processed_signal

def _process_ecg_signal(ecg_signal, fs, p_peaks, q_peaks, r_peaks, s_peaks, t_peaks):
    """
    ECG 신호를 조건에 따라 제로 패딩 및 자르기 처리.

    Args:
        ecg_signal (array-like): 원본 ECG 신호.
        fs (int): 샘플링 주파수.
        p_peaks, q_peaks, r_peaks, s_peaks, t_peaks (array-like): P, Q, R, S, T 피크 배열.

    Returns:
        processed_signal (array-like): 처리된 신호.
    """
    isv = 0
    # P와 T 피크의 시작점 추출
    p_indices = filter_peak_indices(p_peaks, len(ecg_signal), peak_name="P")
    t_indices = filter_peak_indices(t_peaks, len(ecg_signal), peak_name="T")
    r_indices = r_peaks[r_peaks < len(ecg_signal)]

    if len(p_indices) > 1 and len(t_indices) > 1 and len(r_indices) > 1:
        # P, R, T 피크가 모두 여러 개인 경우

        p_segments = []  # P 구간 저장 리스트
        qrs_segments = []  # QRS 구간 저장 리스트
        t_segments = []  # T 구간 저장 리스트

        for i in range(1, len(p_indices)):
            # 현재 P, R, T 피크 인덱스
            p_start = p_indices[i - 1]
            r_peak = r_indices[i - 1]
            t_start = t_indices[i - 1]

            # R-P 간격 및 R-T 간격 확인 후 자르기
            if p_start < r_peak < t_start:
                # P 구간
                p_segment = ecg_signal[p_start:r_peak]
                p_segments.append(p_segment)

                # QRS 구간 (R 중심으로 Q와 S를 포함)
                qrs_start = max(r_peak - 30, 0)  # R 앞쪽으로 Q 추정
                qrs_end = min(r_peak + 30, len(ecg_signal))  # R 뒤쪽으로 S 추정
                qrs_segment = ecg_signal[qrs_start:qrs_end]
                qrs_segments.append(qrs_segment)

                # T 구간
                t_segment = ecg_signal[r_peak:t_start]
                t_segments.append(t_segment)

        # P, QRS, T 구간 각각 평균 계산
        def average_segments(segments):
            max_length = max(len(seg) for seg in segments)
            # 모든 구간을 동일 길이로 패딩 후 평균 계산
            padded_segments = [np.pad(seg, (0, max_length - len(seg)), 'constant') for seg in segments]
            return np.mean(padded_segments, axis=0)

        avg_p_segment = average_segments(p_segments) if len(p_segments) > 0 else []
        avg_qrs_segment = average_segments(qrs_segments) if len(qrs_segments) > 0 else []
        avg_t_segment = average_segments(t_segments) if len(t_segments) > 0 else []

        # 평균 구간 연결하여 One Cycle ECG 생성
        one_cycle_ecg = np.concatenate((avg_p_segment, avg_qrs_segment, avg_t_segment))
        # plot_signal_from_data(one_cycle_ecg)

        isv = 1

    elif len(p_indices) > 0 and len(t_indices) > 0 and len(r_indices) > 0:
        # PQRST 피크를 모두 찾은 경우

        if len(p_indices) > 2:
            r_indices = r_peaks[r_peaks < p_indices[1]]
            t_indices = t_indices[t_indices < p_indices[1]]

        p_start = p_indices[0]
        # print(p_indices)
        # print(t_indices)
        # print(r_indices)


        if(len(t_indices)> 0):
            t_start = t_indices[-1]
        else:
            t_start = p_indices[1] - 30

        # 신호 길이의 30% 계산

        signal_target = ecg_signal[p_start:t_start + 1]
        padding_length = int(len(signal_target) * 1.25)
        # P의 시작점에서 T의 시작점까지 패딩주고 자르기
        p_start = p_start - (padding_length // 2)
        t_start = t_start + 1 +(padding_length//2)


        if p_start < 0:
            p_start = 0
        if t_start > len(ecg_signal):
            t_start = len(ecg_signal) - 1
        # print("crop 시작점 : ", p_start)
        # print("crop 끝점 : ", t_start)

        signal_segment = ecg_signal[p_start:t_start]


        r_center = r_indices[0]
        # print("정렬 전 r_center:",r_center)
        # plot_signal_from_data(signal_segment)

        processed_signal = align_r_peak(signal_segment,r_center)


        isv = 1



    elif len(r_indices) > 0:
        # R 피크만 찾은 경우
        r_center = r_indices[0]  # 첫 번째 R 피크 기준으로 처리

        start_idx = max(0, r_center - 75)
        end_idx = min(len(ecg_signal), r_center + 75)
        signal_segment = ecg_signal[start_idx:end_idx]

        # 신호 길이 초과 시 잘라내기
        processed_signal = align_r_peak(signal_segment,r_center)
        isv = 0
    else:
        # 피크를 찾지 못한 경우, 신호 반환
        processed_signal = ecg_signal
        isv = 0

    return processed_signal, isv



def process_peaks_crop_PQRST(ecg_signal, sampling_rate=500):
    """
    Process an ECG signal, detect peaks, and plot the results.

    Args:
        ecg_signal (array-like): Input ECG signal.
        sampling_rate (int): Sampling rate (default: 150).

    Returns:
        np.ndarray: Processed ECG signal.
    """
    isv = 0
    time = np.arange(0, len(ecg_signal)) / sampling_rate
    temp = ecg_signal
    try:
        # 피크 탐지 (예: detect_peaks 함수 사용)
        p_peaks, q_peaks, r_peaks, s_peaks, t_peaks = detect_peaks(ecg_signal, sampling_rate=sampling_rate)
        r_peaks = r_peaks[r_peaks < len(ecg_signal)]

        p_peak_indices = filter_peak_indices(p_peaks, len(ecg_signal), peak_name="P")
        q_peak_indices = filter_peak_indices(q_peaks, len(ecg_signal), peak_name="Q")
        s_peak_indices = filter_peak_indices(s_peaks, len(ecg_signal), peak_name="S")
        t_peak_indices = filter_peak_indices(t_peaks, len(ecg_signal), peak_name="T")



        # plt.plot(time, ecg_signal, label="ECG Signal")

        if t_peak_indices.any() and q_peak_indices.any() and s_peak_indices.any() and r_peaks.any() and p_peak_indices.any():

            # for peaks, color, label in zip(
            #         [p_peak_indices, r_peaks, t_peak_indices],
            #         ["blue", "green", "red"],
            #         ["P Peaks", "R Peaks", "T Peaks"]
            # ):
            #     # plt.scatter(time[peaks], ecg_signal[peaks], color=color, label=label, zorder=3)

            temp, isv = _process_ecg_signal(ecg_signal, sampling_rate, p_peaks, q_peaks, r_peaks, s_peaks, t_peaks)
            temp = (temp - np.mean(temp)) / np.std(temp)

            ecg_signal = temp
        else:

            ecg_signal, isv =process_peaks_crop(temp,fs=sampling_rate)

    except:
        ecg_signal, isv =process_peaks_crop(temp,fs=sampling_rate)


    return ecg_signal, isv

def process_peaks_crop(ecg_signal, fs=500):
    """
    Process an ECG signal, detect peaks, and plot the results.

    Args:
        ecg_signal (array-like): Input ECG signal.
        sampling_rate (int): Sampling rate (default: 150).

    Returns:
        np.ndarray: Processed ECG signal.
    """
    if np.all(ecg_signal == 0) or not ecg_signal.any():
        isv = 0
        return ecg_signal, isv
    target_length = 512  # 고정된 샘플 수

    # 밴드패스 필터 설계 (0.5-50Hz)


    # 2. R-피크 검출
    peaks, _ = find_peaks(ecg_signal, height=0.6 * np.max(ecg_signal), distance=0.6 * fs)


    # 3. 한 주기 ECG 생성
    r_peak_index = peaks[0]  # 첫 번째 R-피크 위치
    start_idx = max(r_peak_index - round(0.2 * fs), 0)  # P 시작점 (R 이전 200ms)
    end_idx = min(r_peak_index + round(0.8 * fs), 512)  # T 종료점 (R 이후 800ms)

    one_cycle = ecg_signal[start_idx:end_idx]

    if len(one_cycle) == 0:
        isv = 0
        return ecg_signal, isv
    # 4. R-피크 동기화
    max_idx = np.argmax(one_cycle) # 자른 신호에서 R-피크 위치 찾기
    shift_amount = len(one_cycle) // 2 - max_idx  # 중앙으로 이동
    aligned_signal = np.roll(one_cycle, shift_amount)

    # 5. 샘플 수 조정 (270개, 양옆 패딩)
    deficit = target_length - len(aligned_signal)
    pad_left = deficit // 2
    pad_right = deficit - pad_left
    aligned_signal = np.pad(aligned_signal, (pad_left, pad_right), 'constant', constant_values=0)
    isv = 1
    # 결과 저장

    # plot_signal_from_data(aligned_signal)
    return aligned_signal, isv

def detect_peaks(signal, sampling_rate=500):
    # 데이터 길이 확인 및 확장
    if len(signal) < sampling_rate * 3:  # 최소 3초 이상의 데이터가 필요
        signal = np.tile(signal, int(np.ceil(sampling_rate * 3 / len(signal))))

    # NaN 값 처리
    signal = np.nan_to_num(signal, nan=np.nanmean(signal))  # NaN을 평균값으로 대체

    try:
        # R 피크 탐지
        ecg_signals, ecg_info = nk.ecg_process(signal, sampling_rate=sampling_rate)

        # QRS 및 P, T 피크 탐지
        waves_peak, waves_signal = nk.ecg_delineate(
            signal, ecg_info['ECG_R_Peaks'], sampling_rate=sampling_rate, method="dwt"
        )

        # 각 피크를 추출하고, 없을 경우 공백 리스트를 반환
        r_peaks = ecg_info.get('ECG_R_Peaks', [])
        q_peaks = waves_peak.get('ECG_Q_Peaks', [])
        s_peaks = waves_peak.get('ECG_S_Peaks', [])
        p_peaks = waves_peak.get('ECG_P_Peaks', [])
        t_peaks = waves_peak.get('ECG_T_Peaks', [])

        return (
            np.array(p_peaks if p_peaks is not None else []),
            np.array(q_peaks if q_peaks is not None else []),
            np.array(r_peaks if r_peaks is not None else []),
            np.array(s_peaks if s_peaks is not None else []),
            np.array(t_peaks if t_peaks is not None else [])
        )

    except Exception as e:
        # 오류 발생 시 빈 배열 반환
        return np.array([]), np.array([]), np.array([]), np.array([]), np.array([])



def extract_one_cycle(ecg_signal, r_peaks, sampling_rate):
    """
    Extract one-cycle ECG based on detected R-peaks.
    """
    one_cycles = []
    for i in range(len(r_peaks) - 1):
        start_idx = r_peaks[i]
        end_idx = r_peaks[i + 1]
        one_cycle = ecg_signal[start_idx:end_idx]
        one_cycles.append(one_cycle)
    return one_cycles

# 밴드패스 필터 설계 (0.5-50Hz)
def bandpass_filter(signal, low_cutoff, high_cutoff, fs, order=2):
    nyquist = 0.5 * fs
    low = low_cutoff / nyquist
    high = high_cutoff / nyquist
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, signal)