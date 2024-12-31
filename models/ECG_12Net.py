import torch
import torch.nn as nn
import torch.nn.functional as F
from dask.array.overlap import nearest
from networkx.utils.random_sequence import weighted_choice
import numpy as np


def encode_k_class(k_value):
    """
    Encodes the K+ concentration into one of 5 classes.
    """
    if k_value <= 2.5:
        return 0
    elif 2.5 < k_value <= 3.5:
        return 1
    elif 3.5 < k_value < 5.5:
        return 2
    elif 5.5 <= k_value < 6.5:
        return 3
    else:
        return 4


# K+ 농도 인코딩 함수
def encode_k_concentration(k_value, bins=60, min_k=1.5, max_k=7.5):
    step = (max_k - min_k) / bins
    encoded = np.zeros(bins)
    cutoff = int((k_value - min_k) / step)
    encoded[:cutoff] = 1
    return encoded

def get_labels(k_value=4.0):
    # 테스트 데이터

    encoded_label = encode_k_concentration(k_value)

    # 샘플 ECG 데이터 생성 (배치 크기=100, 채널=12, 시퀀스 길이=500)
    ecg_data = torch.randn(100, 12, 1024)

    # 샘플 K+ 농도 생성 (100개의 데이터 포인트)
    k_values = np.random.uniform(1.5, 7.5, size=100)

    # K+ 농도를 60-포인트 시그모이드 벡터로 변환
    labels = np.array([encode_k_concentration(k) for k in k_values])

    # Tensor로 변환
    labels = torch.tensor(labels, dtype=torch.float32)

    return labels

class DenseUnit(nn.Module):
    def __init__(self, in_channels, growth_rate,k):
        super(DenseUnit, self).__init__()
        self.growth_rate = growth_rate
        self.conv1 = nn.Conv1d(in_channels, 128, kernel_size=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm1d(128)
        self.conv2 = nn.Conv1d(128, 128, kernel_size=3, stride=1*k, bias=False)
        self.bn2 = nn.BatchNorm1d(128)
        self.conv3 = nn.Conv1d(128,32,kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm1d(32)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = F.relu(self.bn3(self.conv3(out)))

        return out


class DenseBlock(nn.Module):
    def __init__(self, num_layers, in_channels, growth_rate):
        super(DenseBlock, self).__init__()
        self.layers = nn.ModuleList([
            DenseUnit(in_channels + i * growth_rate, growth_rate,1) for i in range(num_layers)
        ])

    def forward(self, x):
        for layer in self.layers:

            temp = x
            x = layer(x)
            if temp.size(2) != x.size(2):
                diff = abs(x.size(2) - temp.size(2))
                # Pad the last dimension of `out`
                x = F.pad(x, (0, diff))
            x = torch.cat([temp,x],dim=1)
        return x

class TransitionLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(TransitionLayer, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=1, bias=False)
        self.bn = nn.BatchNorm1d(out_channels)
        self.pool = nn.AvgPool1d(kernel_size=2, stride=2)

    def forward(self, x):
        x = F.relu(self.bn(self.conv(x)))
        x2 = self.pool(x)
        return torch.cat([x,x2],1)

class PoolingBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(PoolingBlock, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=1, bias=False)
        self.bn = nn.BatchNorm1d(out_channels)
        self.du = DenseUnit(in_channels,out_channels,2)
        self.pool = nn.AvgPool1d(kernel_size=2, stride=2)

    def forward(self, x):
        # x = F.relu(self.bn(self.conv(x)))
        out1 = self.du(x)

        out2 = self.pool(x)
        out1 = F.interpolate(out1, size=out2.size(2),mode='nearest')
        return torch.cat([out1,out2],dim=1)

class ECGLeadBlock(nn.Module):
    def __init__(self):
        super(ECGLeadBlock, self).__init__()
        self.init_conv = nn.Conv1d(1, 32, kernel_size=7, stride=2, padding=3, bias=False)
        self.init_bn = nn.BatchNorm1d(32)
        self.bn = nn.BatchNorm1d(864)
        self.pooling1 = PoolingBlock(32, 64)
        self.dense1 = DenseBlock(3, 64, 32)
        self.pooling2 = PoolingBlock(160, 320)
        self.dense2 = DenseBlock(3, 192, 32)
        self.pooling3 = PoolingBlock(288, 512)
        self.dense3 = DenseBlock(6, 320, 32)
        self.pooling4 = PoolingBlock(512, 768)
        self.dense4 = DenseBlock(6, 544, 32)
        self.pooling5 = PoolingBlock(736, 768)
        self.dense5 = DenseBlock(3, 768, 32)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(864,1)

    def forward(self, x):

        x = F.relu(self.init_bn(self.init_conv(x)))
        x = self.pooling1(x)
        x = self.dense1(x)
        x = self.pooling2(x)
        x = self.dense2(x)
        x = self.pooling3(x)
        x = self.dense3(x)
        x = self.pooling4(x)
        x = self.dense4(x)
        x = self.pooling5(x)
        x = self.dense5(x)
        x = F.relu(self.bn(x))
        x = self.global_pool(x)
        att = x
        x = x.view(x.squeeze(-1).size())
        x = self.fc(x)
        x = x.view(1,1,1)
        return x, att

class AttentionBlock(nn.Module):
    def __init__(self, in_channels):
        super(AttentionBlock, self).__init__()
        self.fc1 = nn.Linear(in_channels, 8)
        self.bn1 = nn.BatchNorm1d(864)
        self.fc2 = nn.Linear(8, 1)

    def forward(self, x):
        x = x.view(x.squeeze(-1).size())
        attention_scores = F.relu(self.fc1(x))
        # attention_scores = attention_scores.view(1,1,1)
        attention_scores = torch.sigmoid(self.fc2(attention_scores))
        return F.softmax(attention_scores, dim=1)

class ECG12Net(nn.Module):
    def __init__(self):
        super(ECG12Net, self).__init__()
        self.lead_blocks = nn.ModuleList([ECGLeadBlock() for _ in range(12)])
        self.attention = AttentionBlock(864)
        self.fc = nn.Linear(864, 1)
        self.fc1 = nn.Linear(12, 1)

    def forward(self, x):
        # 각 리드 블록에서 (output, att) 값을 함께 가져옴
        weighted_outputs = []
        # attentions = []

        for i, block in enumerate(self.lead_blocks):
            output, att = block(x[:, i:i + 1, :])  # 블록에서 (output, att) 반환
            # lead_outputs.append(output)
            # attentions.append(att)
            #
            # lead_outputs = torch.stack(lead_outputs, dim=1)  # 리드 블록 출력 스택
            # attentions = torch.stack(attentions, dim=1)  # att 값 스택

            attention_scores = self.attention(att)
            weighted_output = output * attention_scores.unsqueeze(-1)
            weighted_outputs.append(weighted_output)

        summed_output = F.sigmoid(self.fc1(torch.tensor(weighted_outputs))).sum()

        return summed_output


class EMPNet(nn.Module):
    def __init__(self):
        super(EMPNet, self).__init__()
        self.fc1 = nn.Linear(8, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class ECG12Net2(nn.Module):
    def __init__(self):
        super(ECG12Net2, self).__init__()
        self.ecg12net1 = ECG12Net()  # 기존 ECG12Net 모델
        self.empnet = EMPNet()
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(1, 60)

    def forward(self, ecg_input, emp_input):
        ecg_output = self.ecg12net1(ecg_input)
        emp_output = self.empnet(emp_input)
        combined = ecg_output + emp_output
        combined = self.dropout(combined)

        return torch.sigmoid(self.fc(combined))

