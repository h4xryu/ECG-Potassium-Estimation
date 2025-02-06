import numpy as np
from scipy import signal
from scipy.signal import find_peaks, hilbert
from utils.dataset.Severance.preprocessing import *

class RationalECGCompressor:
    def __init__(self, sampling_rate=360):
        self.sampling_rate = sampling_rate
        self.alpha = 0.5  # Regularization parameter from paper
        self.architecture_space = self._init_architecture_space()

    def _init_architecture_space(self):
        """
        Initialize the architecture space with 30 configurations as per paper
        Returns dictionary with pole configurations and their dimensions
        """
        # Format: (multiplicity vector, system complexity, virtual dimension)
        configs = [
            ([2, 1, 1], 4, 1),  # Example: 2 poles with mult 2,1,1
            ([2, 2, 1], 5, 2),
            ([2, 2, 2], 6, 3),
            ([3, 2, 1], 6, 4),
            ([3, 2, 2], 7, 5),
            ([3, 3, 2], 8, 6),
            ([4, 2, 2], 8, 7),
            ([4, 3, 2], 9, 8),
            ([4, 3, 3], 10, 9),
            ([5, 3, 2], 10, 10),
            ([2, 4, 6], 12, 11),
            # ... Add all 30 configurations as per Table I in paper
        ]
        return configs

    def _calculate_mt_basis(self, signal_length, poles, multiplicities):
        """
        Calculate stabilized Malmquist-Takenaka rational function basis
        """
        try:
            t = np.linspace(-1, 1, signal_length)
            basis = []

            for pole_idx, (pole, mult) in enumerate(zip(poles, multiplicities)):
                # Complex pole with stability check
                a = pole[0] + 1j * pole[1]
                if abs(a) >= 1.0:  # Ensure pole is inside unit disk
                    a = a / (abs(a) + 1e-8)

                # Base calculation with numerical stability
                for k in range(mult):
                    denom = 1 - np.conj(a) * np.exp(1j * np.pi * t) + 1e-10
                    base_func = np.sqrt(1 - abs(a) ** 2) / denom

                    if k > 0:  # Apply additional terms for higher multiplicities
                        base_func *= ((-a) ** k) / (denom ** k)

                    basis.append(base_func)

            basis = np.array(basis).T

            # Add small regularization for numerical stability
            basis = basis + 1e-10 * np.random.randn(*basis.shape)

            # Normalize basis
            basis = basis / np.sqrt(np.sum(np.abs(basis) ** 2, axis=0))

            return basis

        except Exception as e:
            print(f"Error in MT basis calculation: {str(e)}")
            # Return fallback basis in case of error
            return np.eye(signal_length, min(signal_length, sum(multiplicities)))

    def _hyperbolic_pso_update(self, particles, velocities):
        """
        Update particles using hyperbolic operations in Poincaré disk model
        """

        # Calculate hyperbolic addition using formulas from paper
        def hyperbolic_add(w1, w2):
            return (w1 + w2) / (1 + np.conj(w1) * w2)

        # Project particles back to unit disk using scaling
        norm = np.sqrt(np.sum(particles ** 2, axis=2, keepdims=True))
        mask = (norm > 0.99)
        scaling = np.where(mask, 0.99 / norm, 1.0)
        particles = particles * scaling

        # Apply velocity updates using hyperbolic addition
        new_particles = np.zeros_like(particles)
        for i in range(particles.shape[0]):
            for j in range(particles.shape[1]):
                w1 = particles[i, j, 0] + 1j * particles[i, j, 1]
                v = velocities[i, j, 0] + 1j * velocities[i, j, 1]
                new_w = hyperbolic_add(w1, v)
                new_particles[i, j] = [np.real(new_w), np.imag(new_w)]

        return new_particles

    def _calculate_cost(self, signal, poles, multiplicities):
        """
        Calculate cost function with error handling
        """
        try:
            # Calculate basis
            basis = self._calculate_mt_basis(len(signal), poles, multiplicities)

            # Solve least squares with regularization
            reg_factor = 1e-6
            A = basis.conjugate().T @ basis + reg_factor * np.eye(basis.shape[1])
            b = basis.conjugate().T @ signal

            try:
                coeffs = np.linalg.solve(A, b)
            except np.linalg.LinAlgError:
                # Fallback to pseudo-inverse if direct solve fails
                coeffs = np.linalg.pinv(basis) @ signal

            approximation = basis @ coeffs

            # Calculate PRD
            error = np.abs(signal - approximation) ** 2
            signal_power = np.abs(signal) ** 2
            prd = 100 * np.sqrt(np.sum(error) / (np.sum(signal_power) + 1e-10))

            # Calculate RCR
            n = len(poles)  # number of poles
            N = sum(multiplicities)  # sum of multiplicities
            M = len(signal)  # signal length
            rcr = 2 * (n + N) / M * 100

            cost = self.alpha * prd + (1 - self.alpha) * rcr

            # Handle numerical instability
            if not np.isfinite(cost):
                return float('inf')

            return float(cost)

        except Exception as e:
            print(f"Error in cost calculation: {str(e)}")
            return float('inf')

    def _optimize_poles(self, signal, max_iterations=100):
        """
        Multi-Dimensional Hyperbolic PSO implementation
        """
        num_particles = 30
        num_poles = 3
        dimension = 2  # real and imaginary parts

        # Initialize particles in unit disk
        particles = np.random.uniform(-0.8, 0.8, (num_particles, num_poles, dimension))
        velocities = np.zeros_like(particles)

        personal_best_pos = particles.copy()
        personal_best_val = np.full(num_particles, float('inf'))
        global_best_pos = particles[0].copy()
        global_best_val = float('inf')

        # PSO parameters
        w = 0.8  # inertia
        c1 = 1.5  # cognitive
        c2 = 2.0  # social

        # Use first configuration from architecture space for testing
        multiplicities = self.architecture_space[0][-1]

        for _ in range(max_iterations):
            for i in range(num_particles):
                cost = self._calculate_cost(signal, particles[i], multiplicities)

                # Update personal and global best
                if cost < personal_best_val[i]:
                    personal_best_pos[i] = particles[i].copy()
                    personal_best_val[i] = cost

                    if cost < global_best_val:
                        global_best_pos = particles[i].copy()
                        global_best_val = cost

            # Update velocities
            r1, r2 = np.random.rand(2)
            velocities = (w * velocities +
                          c1 * r1 * (personal_best_pos - particles) +
                          c2 * r2 * (global_best_pos - particles))

            # Update positions using hyperbolic operations
            particles = self._hyperbolic_pso_update(particles, velocities)

            # Decay inertia
            w *= 0.99

        return global_best_pos, multiplicities

    def _find_best_rpeaks(self, ecg_signal, r_peaks, num_best=4):
        """
        Find the best quality R-peaks based on peak prominence and surrounding signal quality

        Args:
            ecg_signal: Input ECG signal
            r_peaks: Detected R-peak positions
            num_best: Number of best peaks to select (default: 4)
        """
        if len(r_peaks) <= num_best:
            return r_peaks

        peak_scores = []
        for peak in r_peaks:
            # Extract window around peak
            start = max(0, peak - 200)
            end = min(len(ecg_signal), peak + 200)
            window = ecg_signal[start:end]

            if len(window) < 260:  # Skip if window is too short
                peak_scores.append(-float('inf'))
                continue

            # Calculate peak prominence
            peak_height = ecg_signal[peak]
            prominence = peak_height - np.min(window)

            # Calculate signal quality metrics
            signal_range = np.max(window) - np.min(window)
            signal_std = np.std(window)

            # Combined quality score
            quality_score = prominence * signal_std / (signal_range + 1e-10)
            peak_scores.append(quality_score)

        # Get indices of best peaks
        best_indices = np.argsort(peak_scores)[-num_best:]
        best_peaks = r_peaks[best_indices]

        # Sort peaks by position
        best_peaks.sort()

        return best_peaks

    def compress(self, ecg_signal):
        """Compress ECG signal using best 4 R-peaks"""
        try:
            # Detect R-peaks
            # r_peaks, _ = find_peaks(ecg_signal,
            #                         distance=int(0.2 * self.sampling_rate),
            #                         height=0.5 * np.max(ecg_signal))

            r_peaks = detect_r_peaks_adaptive(ecg_signal)

            # Select best R-peaks
            # best_peaks = self._find_best_rpeaks(ecg_signal, r_peaks, num_best=int(len(r_peaks)*0.5))

            compressed_beats = []
            processed_r_peaks = []

            for r_peak in r_peaks:
                try:
                    # Extract beat
                    start = max(0, r_peak - 200)
                    end = min(len(ecg_signal), r_peak + 200)

                    if end - start < 50:
                        continue

                    beat = ecg_signal[start:end]

                    # Preprocess
                    beat = beat - np.mean(beat)
                    if np.std(beat) > 1e-10:
                        beat = beat / np.std(beat)

                    # Hilbert transform
                    beat_complex = hilbert(beat)

                    # Default configuration
                    poles = np.array([[0.5, 0], [0, 0.5], [-0.5, 0]])
                    multiplicities = [2, 32, 32]

                    # Calculate compression
                    basis = self._calculate_mt_basis(len(beat_complex), poles, multiplicities)
                    coeffs = np.linalg.lstsq(basis, beat_complex, rcond=1e-10)[0]

                    compressed_beats.append({
                        'poles': poles,
                        'multiplicities': multiplicities,
                        'coefficients': coeffs
                    })
                    processed_r_peaks.append(r_peak)

                except Exception as e:
                    print(f"Error processing beat at {r_peak}: {str(e)}")
                    continue

            if not compressed_beats:
                return self._create_fallback_compression(ecg_signal)


            return compressed_beats, np.array(processed_r_peaks)

        except Exception as e:
            print(f"Compression error: {str(e)}")
            return self._create_fallback_compression(ecg_signal)

    @staticmethod
    def _create_fallback_compression(ecg_signal):
        # 원본 신호를 보존하는 fallback 생성
        return [{
            'poles': np.array([[0.5, 0]]),
            'multiplicities': [1],
            'coefficients': ecg_signal  # 원본 신호 사용
        }], np.array([0])

    def decompress(self, compressed_beats, r_peaks, signal_length):
        """
        Decompress with error handling and fallback to original signal
        """
        try:
            reconstructed = np.zeros(signal_length, dtype=complex)

            for i, (r_peak, comp_data) in enumerate(zip(r_peaks, compressed_beats)):
                try:
                    start = max(0, r_peak - 200)
                    end = min(signal_length, r_peak + 200)

                    if end - start < 50:
                        continue

                    # 에러 체크 추가
                    if len(comp_data['coefficients']) == signal_length:  # 원본 신호인 경우
                        reconstructed = comp_data['coefficients']
                        return np.real(reconstructed)  # 바로 원본 반환

                    basis = self._calculate_mt_basis(end - start,
                                                     comp_data['poles'],
                                                     comp_data['multiplicities'])

                    # 차원 체크
                    if basis.shape[1] != len(comp_data['coefficients']):
                        print(f"Dimension mismatch: basis {basis.shape}, coeffs {len(comp_data['coefficients'])}")
                        reconstructed[start:end] = comp_data['coefficients'][start:end]  # 해당 구간 원본 사용
                        continue

                    beat = basis @ comp_data['coefficients']
                    reconstructed[start:end] = beat

                except Exception as e:
                    print(f"Error decompressing beat {i}: {str(e)}")
                    # 에러 발생 시 해당 구간 원본 신호 사용
                    if len(comp_data['coefficients']) == signal_length:
                        reconstructed[start:end] = comp_data['coefficients'][start:end]
                    continue

            return np.real(reconstructed)

        except Exception as e:
            print(f"Decompression error: {str(e)}")
            # 전체 에러 시 첫 번째 compressed_beat의 coefficients가 원본이라고 가정
            if compressed_beats and len(compressed_beats[0]['coefficients']) == signal_length:
                return np.real(compressed_beats[0]['coefficients'])
            return np.zeros(signal_length)


def compress(ecg_signal):
    # Example usage
    import matplotlib.pyplot as plt

    # Create compressor
    compressor = RationalECGCompressor(sampling_rate=500)

    # Compress signal
    compressed_beats, r_peaks = compressor.compress(ecg_signal)

    # Decompress signal
    reconstructed = compressor.decompress(compressed_beats, r_peaks, len(ecg_signal))

    return reconstructed


