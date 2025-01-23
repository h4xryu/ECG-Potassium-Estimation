import numpy as np
import matplotlib.pyplot as plt


def plot_signal_from_data(data1, data2=None, sampling_frequency=150, multimode=False):
    """
    Plots a signal from given data and sampling frequency.

    Parameters:
        data (array-like): Signal data (e.g., amplitude values).
        sampling_frequency (float): Sampling frequency (Hz).
    """
    # Calculate time axis based on sampling frequency and data length
    num_samples = len(data1)
    time = np.arange(0, num_samples) / sampling_frequency

    if multimode:
        fig, axes = plt.subplots(2, 1, figsize=(8, 6))  # 2행, 1열의 subplot

        # 2. 첫 번째 subplot
        axes[0].plot(time, data1, label="signal-1", color="blue")
        axes[0].set_title("ECG Wave")
        axes[0].set_xlabel("x")
        axes[0].set_ylabel("signal-1")
        axes[0].legend()
        axes[0].grid(True)

        # 3. 두 번째 subplot
        axes[1].plot(time, data2, label="signal-2", color="red")
        axes[1].set_title("ECG Wave")
        axes[1].set_xlabel("x")
        axes[1].set_ylabel("signal-2")
        axes[1].legend()
        axes[1].grid(True)

        # 4. 레이아웃 조정 및 출력
        plt.tight_layout()
        plt.show()

    else:
        # Plot
        plt.figure(figsize=(10, 4))
        plt.plot(time, data1, label="Input Signal")
        plt.title("Signal Plot")
        plt.xlabel("Time (seconds)")
        plt.ylabel("Amplitude")
        plt.grid(True)
        plt.legend()
        plt.show()


