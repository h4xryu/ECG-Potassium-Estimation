import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.stats import skew, kurtosis
from utils.dataset.Severance.preprocessing import *

class HandcraftFeatureExtractor:
    """ECG 신호에서 수동 특징을 추출하는 클래스"""

    def __init__(self):
        pass

    def calculate_t_wave_gradient(self, signal, t_peaks, s_peaks, window_size=50):
        """
        T-wave 하강 기울기 계산. S-T 구간부터 T-wave 끝까지의 기울기를 분석

        Args:
            signal: Single lead ECG signal
            t_peaks: T-wave peak indices
            s_peaks: S-wave peak indices
            window_size: Number of points to consider after T peak
        Returns:
            mean_gradient: Average T-wave descent gradient
        """
        gradients = []

        # T-wave와 S-wave 피크 쌍을 찾아서 분석
        for t_peak in t_peaks:
            if t_peak + window_size >= len(signal):
                continue

            # T-wave 이전의 가장 가까운 S-wave 찾기
            s_peak_before_t = s_peaks[s_peaks < t_peak]
            if len(s_peak_before_t) == 0:
                continue
            s_peak = s_peak_before_t[-1]

            # ST 구간 분석
            st_segment = signal[s_peak:t_peak]
            st_gradient = np.polyfit(np.arange(len(st_segment)), st_segment, 1)[0]

            # T-wave 하강 구간 분석
            t_descent = signal[t_peak:t_peak + window_size]
            t_descent_gradient = np.polyfit(np.arange(window_size), t_descent, 1)[0]

            gradients.extend([st_gradient, t_descent_gradient])

        return np.mean(gradients) if gradients else 0

    def calculate_qrs_features(self, signal, q_peaks, r_peaks, s_peaks):
        """
        QRS complex 관련 특징 계산

        Args:
            signal: Single lead ECG signal
            q_peaks: Q-wave peak indices
            r_peaks: R-wave peak indices
            s_peaks: S-wave peak indices
        Returns:
            qrs_features: QRS complex related features
        """
        qrs_durations = []
        qrs_amplitudes = []

        for i in range(min(len(q_peaks), len(r_peaks), len(s_peaks))):
            # QRS duration
            if i < len(q_peaks) and i < len(s_peaks):
                qrs_duration = s_peaks[i] - q_peaks[i]
                qrs_durations.append(qrs_duration)

            # QRS amplitude
            if i < len(r_peaks):
                r_amplitude = signal[r_peaks[i]]
                if i < len(q_peaks):
                    q_amplitude = signal[q_peaks[i]]
                    qrs_amplitudes.append(r_amplitude - q_amplitude)

        return [
            np.mean(qrs_durations) if qrs_durations else 0,
            np.std(qrs_durations) if qrs_durations else 0,
            np.mean(qrs_amplitudes) if qrs_amplitudes else 0,
            np.std(qrs_amplitudes) if qrs_amplitudes else 0
        ]

    def extract_features(self, ecg_signal, sampling_rate=500):
        """
        ECG 신호에서 다양한 통계적, 형태학적 특징 추출

        Args:
            ecg_signal: shape (batch_size, 12, sequence_length)
            sampling_rate: Signal sampling rate
        Returns:
            features: 추출된 특징 벡터
        """
        if torch.is_tensor(ecg_signal):
            ecg_signal = ecg_signal.cpu().numpy()

        batch_size, num_leads, seq_length = ecg_signal.shape
        features_list = []

        for i in range(batch_size):
            signal = ecg_signal[i]  # (12, sequence_length)
            batch_features = []

            # Process each lead separately and concatenate features
            for lead in range(num_leads):
                lead_signal = signal[lead]  # (sequence_length,)

                # 기본 통계적 특징
                stats_features = [
                    np.mean(lead_signal),
                    np.std(lead_signal),
                    skew(lead_signal),
                    kurtosis(lead_signal),
                    np.max(lead_signal),
                    np.min(lead_signal),
                    np.ptp(lead_signal),  # peak-to-peak
                ]

                # 시간 영역 특징
                rms = np.sqrt(np.mean(np.square(lead_signal)))
                crossing_rate = np.sum(np.diff(np.signbit(lead_signal).astype(int)))

                time_features = [
                    rms,
                    crossing_rate,
                ]

                # Combine all features for this lead
                lead_features = stats_features + time_features
                batch_features.extend(lead_features)

            features_list.append(batch_features)

        # Convert to numpy array with explicit dtype
        features_array = np.array(features_list, dtype=np.float32)

        # Convert to PyTorch tensor
        features_tensor = torch.tensor(features_array, dtype=torch.float32)

        return features_tensor


          # (batch_size, num_features)


class PotassiumRegressor(nn.Module):
    """칼륨 수치 예측을 위한 회귀 모델"""

    def __init__(self, latent_dim, handcraft_feature_dim, hidden_dims=[512,256,128]):
        super().__init__()

        self.feature_dim = latent_dim + handcraft_feature_dim

        # 특징 결합 및 회귀를 위한 MLP
        layers = []
        input_dim = self.feature_dim + 9
        self.ext = nn.Linear(1,512)
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            input_dim = hidden_dim

        # 출력층
        layers.append(nn.Linear(hidden_dims[-1], 1))

        self.mlp = nn.Sequential(*layers)

    def forward(self, vae_features, handcraft_features):
        """
        Args:
            vae_features: (batch_size, feature_dim, seq_len) = (64, 64, 512)
            handcraft_features: (batch_size, feature_dim) = (64, 18)
        """
        # handcraft_features를 3D로 확장: (64, 18) -> (64, 18, 1)
        handcraft_features = handcraft_features.unsqueeze(-1)

        # handcraft_features를 sequence_length 차원으로 반복
        handcraft_features = handcraft_features.expand(-1, -1, vae_features.size(1))  # (64, 18, 512)
        handcraft_features = handcraft_features.permute(0,2,1)

        # 이제 feature 차원(dim=1)으로 concatenate
        combined_features = torch.cat([vae_features, handcraft_features], dim=2)
        # combined_features: (64, 82, 512)  # 64 + 18 = 82

        # For debugging
        # print(f"VAE features shape: {vae_features.shape}")
        # print(f"Expanded handcraft features shape: {handcraft_features.shape}")
        # print(f"Combined features shape: {combined_features.shape}")
        combined_features = combined_features.mean(dim=1)
        potassium = self.mlp(combined_features)
        return potassium





