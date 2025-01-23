import matplotlib.pyplot as plt
import numpy as np


class SignalCollector:
    def __init__(self):
        self.selected_signals = []
        self.selected_patient_ids = []
        self.selected_potassium_levels = []

    def save_temp_data(self):
        """임시 데이터를 넘파이 파일로 저장."""
        np.savez("temp_selected_data_2.npz",
                 signals=np.array(self.selected_signals, dtype=object),
                 patient_ids=np.array(self.selected_patient_ids, dtype=object),
                 potassium_levels=np.array(self.selected_potassium_levels))
        print("Temporary data saved to temp_selected_data.npz.")

    def load_temp_data(self):
        """임시 저장된 데이터를 불러옵니다."""
        try:
            data = np.load("temp_selected_data.npz", allow_pickle=True)
            self.selected_signals = list(data["signals"])
            self.selected_patient_ids = list(data["patient_ids"])
            self.selected_potassium_levels = list(data["potassium_levels"])
            print("Temporary data loaded successfully.")
        except FileNotFoundError:
            print("No temporary data file found.")

    def plot_signal(self, index, signals, patient_ids, potassium_levels):
        """신호를 플롯하고 키 입력에 따라 행동을 수행."""
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(signals[index], label=f"Patient: {patient_ids[index]}, K: {potassium_levels[index]}")
        ax.set_title(f"Signal for Patient {patient_ids[index]}")
        ax.set_xlabel("Time")
        ax.set_ylabel("Amplitude")
        ax.legend()

        selected = False

        def on_key(event):

            nonlocal selected
            if event.key == 'r':
                self.selected_signals.append(signals[index])
                self.selected_patient_ids.append(patient_ids[index])
                self.selected_potassium_levels.append(potassium_levels[index])
                print("Signal selected.")
                self.save_temp_data()
                selected = True
                plt.close()
            elif event.key == 'e':
                print("Signal skipped.")
                plt.close()

        fig.canvas.mpl_connect('key_press_event', on_key)
        plt.show()


        if selected:
            return signals[index], patient_ids[index], potassium_levels[index]
        else:
            return None, None, None