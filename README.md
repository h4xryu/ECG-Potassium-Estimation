# Variational Auto Encoder Based Two stage Potassium Level Prediction Framework
***

This repository contains implementations and preprocessing methods for deep learning-based potassium level prediction from ECG signals. The dataset consists of 826 patients' ECG signals (Lead-II, V5) sampled at 500 Hz from Wonju Severance Hospital.


A two-stage learning framework for non-invasive potassium level prediction using electrocardiogram (ECG) signals. This framework combines traditional ECG feature extraction with deep learning-based variational autoencoder for robust representation learning.

## Overview

This project implements a novel approach for predicting potassium levels from ECG signals using:
- Beta-VAE for learning robust ECG signal representations
- Handcrafted feature extraction for capturing domain knowledge
- A fusion mechanism combining both learned and handcrafted features

  ![nn](https://ifh.cc/g/fpKc08.png)

**Dataset**

Distribution of serum potassium levels across 826 patients:

![nn](https://ifh.cc/g/6MPfhp.png)

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




## Architecture

### Stage 1: Feature Learning
- **Deep Learning Path**: Beta-VAE architecture for learning latent representations
- **Traditional Path**: Extraction of clinically relevant ECG features
- Input: Median beat ECG signals (leads II and V5)

### Stage 2: Potassium Prediction
- Feature fusion module combining deep and handcrafted features
- Multi-layer perceptron for final potassium level prediction

## Requirements

```bash
torch
numpy
scipy
matplotlib
```

## Usage

### Training the Model

```python
from model import train_two_stage_model

# Example hyperparameters
hyper_param = {
    'reconstruction_loss': 'mae',
    'in_channel': 2,
    'channels': 128,
    'depth': 7,
    'width': 512,
    'kernel_size': 5,
    'encoder_dropout': 0.3,
    'decoder_dropout': 0.0,
    'softplus_eps': 1.0e-4,
    'num_epochs_vae': 1000,
    'num_epochs_regressor': 1000,
    'batch_size': 64,
    'hidden_dim': 128,
    'latent_dim': 64,
    'beta': 1.1,
    'learning_rate': 1e-3
}

# Train the model
vae_model, regressor, feature_extractor, val_loader = train_two_stage_model(
    root='./dataset',
    **hyper_param
)
```

### Testing Reconstruction

```python
def test_reconstruction(model, test_loader, device='cuda', num_examples=5):
    """
    Visualize original and reconstructed ECG signals.
    
    Args:
        model: Trained VAE model
        test_loader: DataLoader containing test data
        device: Computing device (cuda/cpu)
        num_examples: Number of examples to visualize
    """
    # Visualization code for comparing original and reconstructed signals
```

## Model Architecture

### Model Components

| Component | Description | Architecture |
|-----------|-------------|--------------|
| Beta-VAE | ECG Signal Representation Learning | - Input: (batch_size, 2, sequence_length) ECG signals<br>- Encoder: Conv1D layers with increasing channels<br>- Latent space: 64-dimensional<br>- Decoder: Transposed Conv1D layers |
| HandcraftFeatureExtractor | Traditional ECG Feature Extraction | - Statistical features<br>- Morphological features<br>- Time-domain features<br>- Output: 9-dimensional feature vector |
| PotassiumRegressor | Feature Fusion & Prediction | - Input: Combined VAE (64-dim) and handcrafted (9-dim) features<br>- Hidden layers: [512, 256, 128]<br>- Output: Single potassium value |

### Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| reconstruction_loss | 'mae' | VAE reconstruction loss type |
| in_channel | 2 | Number of input ECG leads |
| channels | 128 | Base number of convolutional channels |
| depth | 7 | Number of convolutional layers |
| width | 512 | Input sequence length |
| kernel_size | 5 | Convolutional kernel size |
| encoder_dropout | 0.3 | Encoder dropout rate |
| decoder_dropout | 0.0 | Decoder dropout rate |
| softplus_eps | 1.0e-4 | Softplus epsilon value |
| num_epochs_vae | 1000 | VAE training epochs |
| num_epochs_regressor | 1000 | Regressor training epochs |
| batch_size | 64 | Training batch size |
| hidden_dim | 128 | Hidden layer dimension |
| latent_dim | 64 | VAE latent dimension |
| beta | 1.1 | VAE beta parameter |
| learning_rate | 1e-3 | Initial learning rate |

### Detailed Model Architectures

#### 1. Beta-VAE Architecture

| Layer | Output Shape | Parameters |
|-------|-------------|------------|
| Input | (B, 2, 512) | - |
| Conv1D | (B, 128, 512) | channels=128, kernel_size=5, stride=1 |
| BatchNorm + ReLU + Dropout(0.3) | (B, 128, 512) | - |
| Conv1D Blocks × depth | (B, 128, 512) | depth=7 layers with residual connections |
| Flatten | (B, 65536) | - |
| Linear (μ) | (B, 64) | Latent mean |
| Linear (σ) | (B, 64) | Latent variance |
| Sampling | (B, 64) | Reparameterization trick |
| Linear | (B, 65536) | - |
| Reshape | (B, 128, 512) | - |
| TransConv1D Blocks × depth | (B, 128, 512) | depth=7 layers with residual connections |
| TransConv1D | (B, 2, 512) | Output reconstruction |

#### 2. HandcraftFeatureExtractor Features

| Feature Category | Features | Dimension |
|-----------------|----------|----------|
| Statistical Features | Mean, Std, Skewness, Kurtosis, Max, Min, Peak-to-peak | 7 per lead |
| Time Domain Features | RMS, Zero-crossing rate | 2 per lead |
| Total Features | All features × 12 leads | 9 dimensions |

#### 3. PotassiumRegressor Architecture

| Layer | Output Shape | Parameters |
|-------|-------------|------------|
| Input (VAE Features) | (B, 64) | VAE latent representation |
| Input (Handcraft) | (B, 9) | Handcrafted features |
| Concatenate | (B, 73) | Combined features |
| Linear + BatchNorm + ReLU + Dropout | (B, 512) | dropout=0.2 |
| Linear + BatchNorm + ReLU + Dropout | (B, 256) | dropout=0.2 |
| Linear + BatchNorm + ReLU + Dropout | (B, 128) | dropout=0.2 |
| Linear | (B, 1) | Final potassium prediction |

Notes:
- B: Batch size
- All convolutional layers use padding='same'
- Dropout rates: Encoder=0.3, Decoder=0.0, Regressor=0.2
- BatchNorm is applied after each linear/conv layer
- ReLU activation is used throughout the network

### Training Process

1. **Stage 1 - VAE Training**:
   - Loss = Reconstruction Loss + β * KL Divergence
   - Reconstruction options: MAE, MSE, Huber, Smooth L1
   - Learning rate scheduling with ReduceLROnPlateau
   - Best model saving based on validation loss

2. **Stage 2 - Regressor Training**:
   - Combined feature learning
   - MSE loss for potassium level prediction
   - Early stopping with patience
   - Gradient clipping for stability


- Extracts traditional ECG features including:
  - Statistical features (mean, std, skewness, kurtosis)
  - Morphological features (QRS complex, T-wave gradients)
  - Time-domain features (RMS, zero-crossing rate)

### BetaVAE
- Variational autoencoder with β-VAE objective
- Configurable architecture depth and width
- Multiple reconstruction loss options (MAE, MSE, Huber, Smooth L1)

### PotassiumRegressor
- Multi-layer perceptron for feature fusion and regression
- Combines VAE latent features with handcrafted features
- Dropout and batch normalization for regularization

### Evaluation

   - reconstruction test
      ![nn](https://ifh.cc/g/QNbCpg.png)
      Reconstruction Error - MSE: 39.744705, MAE: 3.202912
     
   - confusion matrix
      ![nn](https://ifh.cc/g/frC8vK.png)
     
## Model Performance

### Classification Metrics by Potassium Range

| Range (mEq/L) | Precision | Recall | F1-Score | Support |
|---------------|-----------|---------|----------|----------|
| < 4.0         | 0.00      | 0.00    | 0.00     | 4        |
| 4.0-5.0       | 0.25      | 0.04    | 0.06     | 28       |
| 5.0-6.0       | 0.51      | 0.42    | 0.46     | 52       |
| 6.0-7.0       | 0.29      | 0.50    | 0.36     | 36       |
| 7.0-8.0       | 0.27      | 0.46    | 0.34     | 28       |
| > 8.0         | 0.43      | 0.18    | 0.25     | 17       |

### Overall Performance Metrics

| Metric        | Score |
|---------------|-------|
| Accuracy      | 0.35  |
| Macro Avg     | 0.25  |
| Weighted Avg  | 0.32  |

### Key Observations:
1. Best performance in the 5.0-6.0 mEq/L range (F1-score: 0.46)
2. Limited performance in extreme ranges (< 4.0 and > 8.0 mEq/L)
3. Model shows moderate recall for 6.0-7.0 and 7.0-8.0 ranges
4. Overall accuracy of 35% across all ranges

### Dataset Distribution:
- Total samples: 165
- Largest class: 5.0-6.0 mEq/L (52 samples)
- Smallest class: < 4.0 mEq/L (4 samples)
