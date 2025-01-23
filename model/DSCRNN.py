from unicodedata import bidirectional

import torch
import torch.nn as nn
import torch.nn.functional as F




def calculate_accuracy(outputs, labels):
    # 다중 클래스 분류
    _, predicted = torch.max(outputs, 1)
    correct = (predicted == labels).sum().item()
    return correct / len(labels)

class DenseUnit(nn.Module):
    def __init__(self, in_channels, growth_rate,k):
        super(DenseUnit, self).__init__()
        self.growth_rate = growth_rate
        self.conv1 = nn.Conv1d(in_channels, 128, kernel_size=3, stride=1, bias=False)
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

# Depthwise Convolution in PyTorch
class DepthwiseConv1D(nn.Module):
    def __init__(self, in_channels, kernel_size, padding=1):
        super(DepthwiseConv1D, self).__init__()
        self.depthwise = nn.Conv1d(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size, padding=padding,
                                   groups=in_channels)

    def forward(self, x):
        return self.depthwise(x)
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
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=1, bias=False)
        self.bn = nn.BatchNorm1d(out_channels)
        self.du = DenseUnit(in_channels,out_channels,2)
        self.pool = nn.AvgPool1d(kernel_size=2, stride=2)

    def forward(self, x):
        # x = F.relu(self.bn(self.conv(x)))
        out1 = self.du(x)

        out2 = self.pool(x)
        out1 = F.interpolate(out1, size=out2.size(2),mode='nearest')
        return torch.cat([out1,out2],dim=1)

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

class DCRNNModel_longseq(nn.Module):
    def __init__(self, in_channel=1):
        super(DCRNNModel_longseq, self).__init__()

        self.init_conv = nn.Conv1d(1, 32, kernel_size=7, stride=2, padding=32, bias=False)
        self.init_bn = nn.BatchNorm1d(32)

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
        self.bn = nn.BatchNorm1d(864)

        # Attention
        # self.fc0 = nn.Linear(169 * 896, 512)
        self.attention = AttentionBlock(in_channels=864)

        # Fully Connected Layer
        self.fc = nn.Linear(864, 1)

        self.dropout = nn.Dropout(0.5)

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
        att = att.permute(0,2,1)
        x = x.view(x.squeeze(-1).size())
        output = self.fc(x)
        attention_scores = self.attention(att)

        weighted_output = output * attention_scores.unsqueeze(-1)

        out = F.sigmoid(weighted_output)
        return out

# Complete Model
class DCRNNModel(nn.Module):
    '''
    Groups가 입력채널과 동일해야 DSCRNN임
    '''
    def __init__(self, in_channel=1):
        super(DCRNNModel, self).__init__()

        # SeparableConv1
        self.depthwise1 = DepthwiseConv1D(in_channels=in_channel, kernel_size=368,padding=32)
        self.pointwise1 = nn.Conv1d(in_channels=in_channel, out_channels=64, kernel_size=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.pool1 = nn.MaxPool1d(kernel_size=2)

        # SeparableConv2
        self.depthwise2= DepthwiseConv1D(in_channels=64, kernel_size=128,padding=32)
        self.pointwise2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=1)
        self.bn2 = nn.BatchNorm1d(128)
        self.pool2 = nn.MaxPool1d(kernel_size=2)

        # SeparableConv3
        self.depthwise3= DepthwiseConv1D(in_channels=128, kernel_size=32,padding=32)
        self.pointwise3 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=1)
        self.bn3 = nn.BatchNorm1d(256)

        # LSTM
        self.lstm = nn.LSTM(input_size=256, hidden_size=64, num_layers=3, batch_first=True, bidirectional=False)

        # Fully Connected Layer
        self.fc = nn.Linear(53 * 64, 1)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        # SeparableConv1
        x = self.depthwise1(x)
        x = F.relu(self.pointwise1(x))
        x = self.bn1(x)
        x = self.pool1(x)

        # SeparableConv2
        x = self.depthwise2(x)
        x = F.relu(self.pointwise2(x))
        x = self.bn2(x)
        x = self.pool2(x)

        # SeparableConv3
        x = self.depthwise3(x)
        x = F.relu(self.pointwise3(x))
        x = self.bn3(x)

        # LSTM
        x = x.permute(0, 2, 1)  # Adjust dimensions for LSTM (batch, seq, feature)
        x, _ = self.lstm(x)

        # Reshape for fully connected layer
        x = x.reshape(-1, 53 * 64)
        # Fully Connected
        x = self.dropout(x)
        out = self.fc(x)
        return out

