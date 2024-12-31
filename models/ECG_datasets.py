import torch




# from torch.utils.data import Dataset, DataLoader
#
# class ECGDataset(Dataset):
#     def __init__(self, ecg_data, k_values, bins=60, min_k=1.5, max_k=7.5):
#         self.ecg_data = ecg_data
#         self.k_values = k_values
#         self.bins = bins
#         self.min_k = min_k
#         self.max_k = max_k
#
#     def __len__(self):
#         return len(self.ecg_data)
#
#     def __getitem__(self, idx):
#         ecg = self.ecg_data[idx]
#         k_value = self.k_values[idx]
#         # label = encode_k_concentration(k_value, self.bins, self.min_k, self.max_k)
#         return ecg, torch.tensor(label, dtype=torch.float32)



# # 데이터 확인
# for batch in dataloader:
#     ecg_batch, label_batch = batch
#     print("ECG Batch Shape:", ecg_batch.shape)
#     print("Label Batch Shape:", label_batch.shape)
#     break

# import matplotlib.pyplot as plt
#
# # 첫 번째 데이터 포인트의 라벨 시각화
# plt.plot(labels[0].numpy())
# plt.title("Encoded K+ Concentration (Label)")
# plt.xlabel("Bins")
# plt.ylabel("Value")
# plt.show()