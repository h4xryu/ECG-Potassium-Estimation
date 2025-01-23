# Deep Learning based Serum potassium estimation
***

This repository contains implementations and preprocessing methods for deep learning-based potassium level prediction from ECG signals. The dataset consists of 1,567 patients' ECG signals (Lead-II) sampled at 500 Hz from Wonju Severance Hospital.

**Dataset**

| Potassium Concentration Range (Lead-II) (mEq/L) | Number of Samples |
|---------------------------------------|-------------------|
| `< 4.0`                               | 52                |
| `4.0 - 5.0`                           | 190               |
| `5.0 - 6.0`                           | 496               |
| `6.0 - 7.0`                           | 336               |
| `7.0 - 8.0`                           | 341               |
| `> 8.0`                               | 152               |

**Serum Potassium level Distribution**  
   ![Potassium Distribution](https://ifh.cc/g/cFKT1Q.png)

Preprocessing Overview: Refer to the included visualizations and preprocessing function for insights into the signal preparation process.

---

## Preprocessing

**Signal extraction**
  ![Signal extraction](https://ifh.cc/g/lRoHf9.png)

Used viterbi algorithm

**PQRST Detection**  
   ![PQRST Annotation](https://ifh.cc/g/mWLP8Q.png)

In case, whenever i run into the problem difficult to find PQRST parameter, I used to crop by R-peak-wise
   
**Signal Cycle**  
   ![Signal Cycle](https://ifh.cc/g/LX4y0y.png) 


## Preprocessing Pipeline

The following preprocessing steps were applied to the ECG signals before feeding them into the model:
1. **Baseline Wander Removal**: Removes low-frequency noise.
2. **High-Pass Filtering**: Eliminates DC components and slow drift.
3. **Bandstop Filtering**: Suppresses powerline interference.

## Model
Depth-wise separable convolution + LSTM
  ![nn](https://ifh.cc/g/gmlLSQ.png)
  ![nn](https://ifh.cc/g/gl5D4f.png)

---

## DCRNNModel Architecture

The `DCRNNModel` architecture consists of the following layers:

| Layer                 | Description                                                                                   | Parameters                     |
|-----------------------|-----------------------------------------------------------------------------------------------|--------------------------------|
| **DepthwiseConv1D**   | Depthwise convolution with kernel size 368 and padding 32                                     | Input: `in_channel`, Output: `in_channel` |
| **PointwiseConv1D**   | Pointwise convolution with kernel size 1, output channels: 64                                 | Input: `in_channel`, Output: 64 |
| **BatchNorm1d**       | Batch normalization after pointwise convolution                                               | Input: 64, Output: 64          |
| **MaxPool1d**         | Max pooling with kernel size 2                                                               | Input: 64, Output: 64          |
| **DepthwiseConv1D**   | Depthwise convolution with kernel size 128 and padding 32                                     | Input: 64, Output: 64          |
| **PointwiseConv1D**   | Pointwise convolution with kernel size 1, output channels: 128                                | Input: 64, Output: 128         |
| **BatchNorm1d**       | Batch normalization after pointwise convolution                                               | Input: 128, Output: 128        |
| **MaxPool1d**         | Max pooling with kernel size 2                                                               | Input: 128, Output: 128        |
| **DepthwiseConv1D**   | Depthwise convolution with kernel size 32 and padding 32                                      | Input: 128, Output: 128        |
| **PointwiseConv1D**   | Pointwise convolution with kernel size 1, output channels: 256                                | Input: 128, Output: 256        |
| **BatchNorm1d**       | Batch normalization after pointwise convolution                                               | Input: 256, Output: 256        |
| **LSTM**              | LSTM with input size 256, hidden size 64, 3 layers                                           | Input: 256, Output: 64         |
| **Fully Connected**   | Fully connected layer with input size `53 * 64`, output size: 1                              | Input: `53 * 64`, Output: 1    |
| **Dropout**           | Dropout with probability 0.2                                                                 | Applied to fully connected layer |


---
## Perfomance

**Model Loss Curve (Huber Loss)**  

  ![loss](https://ifh.cc/g/8zbZwt.png)

**Heatmap**

  ![heatmap](https://ifh.cc/g/movOD8.png)

**Boxplots**

  ![boxplot](https://ifh.cc/g/wVvvQc.png)
