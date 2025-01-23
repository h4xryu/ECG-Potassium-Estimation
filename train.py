import random

import matplotlib.pyplot as plt
import numpy as np
import torch
from IPython.core.pylabtools import figsize
from torch.distributed import group
from torch.utils.data import ConcatDataset
import seaborn as sns
from data_load import *
from utils import visualize
from utils.dataset.Severance import *
from model import *
from sklearn.model_selection import train_test_split, KFold
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def test_model(ecg,spl, model, criterion):
    """
    테스트 데이터셋을 사용하여 모델을 평가하는 함수.

    Args:
        model: 훈련된 모델.
        test_loader: 테스트 데이터 로더.
        device: 실행 디바이스 (CPU 또는 GPU).
        criterion: 손실 함수.

    Returns:
        평균 손실과 실제값 및 예측값 리스트.
    """
    min_len = min(len(signal) for signal in ecg)

    x_train, x_test, y_train, y_test = train_test_split(ecg, spl, test_size=0.2, random_state=42)
    x_train = [series[0:min_len] for series in x_train]
    x_test = [series[0:min_len] for series in x_test]
    train_dataset = ECGDataset(x_train, y_train)
    test_dataset = ECGDataset(x_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()  # 모델을 평가 모드로 전환
    test_loss = 0.0
    predictions = []
    ground_truths = []
    with torch.no_grad():
        for sig_batch, lab_batch in test_loader:
            sig_batch = sig_batch.unsqueeze(1).to(device)
            lab_batch = lab_batch.unsqueeze(1).to(device)

            outputs = model(sig_batch)
            loss = criterion(outputs, lab_batch)
            test_loss += loss.item()

            # 저장
            predictions.extend(outputs.cpu().numpy())
            ground_truths.extend(lab_batch.cpu().numpy())

    avg_loss = test_loss / len(test_loader)
    print(f"Test Loss: {avg_loss:.4f}")
    return avg_loss, predictions, ground_truths


def train(model, device, ecg, spl, criterion, num_epochs=3000):
    min_len = min(len(signal) for signal in ecg)
    print(min_len)
    print()
    print()
    k_folds = 5

    torch.manual_seed(42)
    kfold = KFold(n_splits=k_folds, shuffle=True)
    x_train, x_test, y_train, y_test = train_test_split(ecg, spl, test_size=0.2, random_state=42)
    x_train = [series[0:min_len] for series in x_train]
    x_test = [series[0:min_len] for series in x_test]
    train_dataset = ECGDataset(x_train, y_train)
    test_dataset = ECGDataset(x_test, y_test)
    dataset = ConcatDataset([train_dataset, test_dataset])
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    writer = SummaryWriter(log_dir='./logs')

    input_size = min_len
    # model = DCRNNModel(input_size).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.00005)
    best_loss = float('inf')
    best_model_state = None

    # 7. 학습 루프

        # Initialize optimize

        # for epoch in (tqdm(range(num_epochs),desc='training',total=num_epochs)):
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0

        for sig_batch, lab_batch in tqdm(train_loader,desc='training',total=len(train_loader)):
            sig_batch = sig_batch.unsqueeze(1).to(device)
            lab_batch = lab_batch.unsqueeze(1).to(device)

                # for signal, label in zip(sig_batch, lab_batch):
                #     signal, label = signal.to(device), label.to(device)
            optimizer.zero_grad()
            outputs = model(sig_batch)
            loss = criterion(outputs, lab_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        writer.add_scalar("Loss/train", train_loss / len(train_loader), epoch)
        print(f" Epoch {epoch + 1}/{num_epochs}, Loss: {train_loss / len(train_loader):.4f}", end='')

            # 8. 테스트
        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for sig_batch, lab_batch in test_loader:
                sig_batch = sig_batch.unsqueeze(1).to(device)
                lab_batch = lab_batch.unsqueeze(1).to(device)


                outputs = model(sig_batch)
                loss = criterion(outputs, lab_batch)
                test_loss += loss.item()
        avg_test_loss = test_loss / len(test_loader)
        print(f" Validation Loss: {test_loss / len(test_loader):.4f}")
        writer.add_scalar("Loss/Valid", test_loss / len(test_loader), epoch)

            # 베스트 모델 갱신
        if avg_test_loss < best_loss:
            best_loss = avg_test_loss
            best_model_state = model.state_dict()

    writer.close()
    # best_model = DCRNNModel().to(device)
    # best_model.load_state_dict(best_model_state)
    # print(f"Best Validation Loss: {best_loss:.4f}")
    return model