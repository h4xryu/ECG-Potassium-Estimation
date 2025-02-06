from torch.utils.data import ConcatDataset
import seaborn as sns
from data_load import *
from utils.delineation import *
from utils.dataset.Severance import *
from model import *
from sklearn.model_selection import train_test_split, KFold
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from logger import *
torch.manual_seed(37)

##############################################################################################
#                                         Beta-VAE                                           #
##############################################################################################

def train_two_stage(vae_model, regressor, feature_extractor, train_loader, val_loader,
                    num_epochs=100, device='cuda', learning_rate=1e-3):
    """Two-stage 모델 학습"""
    optimizer = torch.optim.Adam(regressor.parameters(), lr=learning_rate)
    criterion = nn.HuberLoss()  # Huber Loss 사용

    vae_model.eval()  # VAE는 고정
    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        # Training
        regressor.train()
        train_loss = 0

        for batch_idx, (data, k_level) in enumerate(train_loader):
            data = data.to(device)
            k_level = k_level.to(device)

            # 1. VAE 특징 추출
            with torch.no_grad():
                mu, _ = vae_model.encoder(data)

                vae_features = mu  # 잠재 벡터 사용

            # 2. Handcraft 특징 추출
            handcraft_features = feature_extractor.extract_features(data).to(device)

            # 3. 칼륨 수치 예측
            optimizer.zero_grad()
            pred_k = regressor(vae_features, handcraft_features)
            loss = criterion(pred_k, k_level.unsqueeze(1))

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

        # Validation
        regressor.eval()
        val_loss = 0
        predictions = []
        ground_truths = []

        with torch.no_grad():
            for data, k_level in val_loader:
                data = data.to(device)
                k_level = k_level.to(device)

                # 특징 추출
                mu, _ = vae_model.encoder(data)
                handcraft_features = feature_extractor.extract_features(data).to(device)

                # 예측
                pred_k = regressor(mu, handcraft_features)
                loss = criterion(pred_k, k_level.unsqueeze(1))

                val_loss += loss.item()
                predictions.extend(pred_k.cpu().numpy())
                ground_truths.extend(k_level.cpu().numpy())

        avg_val_loss = val_loss / len(val_loader)

        # 메트릭 계산
        predictions = np.array(predictions)
        ground_truths = np.array(ground_truths)
        mae = np.mean(np.abs(predictions - ground_truths))

        print(f'Epoch {epoch + 1}/{num_epochs}:')
        print(f'Train Loss: {avg_train_loss:.4f}')
        print(f'Val Loss: {avg_val_loss:.4f}')
        print(f'MAE: {mae:.4f}')

        # 모델 저장
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                'regressor_state_dict': regressor.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': avg_val_loss,
            }, 'best_regressor_model.pth')

    return regressor

##############################################################################################
#                                        Train with regressor                                #
##############################################################################################

def train_two_stage_model(root='./dataset',
    device = None,
    in_channel = 12,
    channels=128,
    depth=7,
    width=600,
    kernel_size=5,
    encoder_dropout=0.3,
    decoder_dropout=0.0,
    softplus_eps=1.0e-4,
    reconstruction_loss='mae',
    num_epochs_vae=500,
    num_epochs_regressor=100, # 하이퍼파라미터
    batch_size = 64,
    hidden_dim = 128,
    latent_dim = 64,
    beta = 1.1,
    learning_rate = 1e-3):


    dataset = MultiLeadECGDataset(root=root)
    train_size = int(0.7 * len(dataset))
    val_size = int(0.2 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # 1. Beta-VAE 학습
    vae_model = BetaVAE(in_channels=in_channel,
                 channels=channels,
                 depth=depth,
                 latent_dim=latent_dim,
                 width=width,
                 beta=beta,
                 kernel_size=kernel_size,
                 encoder_dropout=encoder_dropout,
                 decoder_dropout=decoder_dropout,
                 softplus_eps=softplus_eps).to(device)
    vae_optimizer = torch.optim.Adam(vae_model.parameters(), lr=learning_rate)

    print("Training Beta-VAE...")
    best_vae_loss = float('inf')
    recon_loss = 0

    # Training

    optimizer = torch.optim.Adam(vae_model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10, verbose=True
    )

    for epoch in range(num_epochs_vae):

        vae_model.train()
        train_loss = 0
        train_recon_loss = 0
        train_kl_loss = 0
        for batch_idx, (data, _) in enumerate(train_loader):
            data = data.to(device)
            optimizer.zero_grad()

            # Forward pass
            recon_batch, mu, logvar = vae_model(data)

            # Reconstruction loss
            if reconstruction_loss == 'mse':
                recon_loss = F.mse_loss(recon_batch, data, reduction='sum')
            elif reconstruction_loss == 'mae':
                recon_loss = F.l1_loss(recon_batch, data, reduction='sum')
            elif reconstruction_loss == 'huber':
                recon_loss = F.huber_loss(recon_batch, data, reduction='sum')
            elif reconstruction_loss == 'smooth_l1':
                recon_loss = F.smooth_l1_loss(recon_batch, data, reduction='sum')

            # KL divergence loss
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

            # Total loss with beta weighting
            loss = recon_loss + beta * kl_loss

            # Backward pass
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_recon_loss += recon_loss.item()
            train_kl_loss += kl_loss.item()

        avg_train_loss = train_loss / len(train_loader.dataset)
        avg_train_recon = train_recon_loss / len(train_loader.dataset)
        avg_train_kl = train_kl_loss / len(train_loader.dataset)

        # Validation
        vae_model.eval()
        val_loss = 0
        val_recon_loss = 0
        val_kl_loss = 0

        with torch.no_grad():
            for data, _ in val_loader:
                data = data.to(device)
                recon_batch, mu, logvar = vae_model(data)

                kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
                loss = recon_loss + beta * kl_loss

                val_loss += loss.item()
                val_recon_loss += recon_loss.item()
                val_kl_loss += kl_loss.item()

        avg_val_loss = val_loss / len(val_loader.dataset)
        avg_val_recon = val_recon_loss / len(val_loader.dataset)
        avg_val_kl = val_kl_loss / len(val_loader.dataset)

        # Learning rate scheduling
        scheduler.step(avg_val_loss)

        # Save best model
        # Compare the loss value, not the loss object
        current_loss = avg_val_loss

        # Save best model
        if current_loss < best_vae_loss:
            best_vae_loss = current_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': vae_model.state_dict(),
                'optimizer_state_dict': vae_optimizer.state_dict(),
                'loss': current_loss,
            }, 'best_beta_vae.pth')

        # Print progress
        print(f'Epoch {epoch + 1}/{num_epochs_vae}:')
        print(f'Train Loss: {avg_train_loss:.4f} '
                f'(Recon: {avg_train_recon:.4f}, KL: {avg_train_kl:.4f})')
        print(f'Val Loss: {avg_val_loss:.4f} '
                f'(Recon: {avg_val_recon:.4f}, KL: {avg_val_kl:.4f})')
        print('-' * 80)

    # 2. Handcraft Feature Extractor 초기화
    feature_extractor = HandcraftFeatureExtractor()

    # 3. Regressor 학습
    handcraft_feature_dim = 9  # 12 leads * 14 features per lead = ?
    regressor = PotassiumRegressor(latent_dim, handcraft_feature_dim).to(device)

    print("\nTraining Regressor...")
    regressor = train_two_stage(vae_model, regressor, feature_extractor,
                                train_loader, val_loader, num_epochs_regressor, device)

    return vae_model, regressor, feature_extractor, val_loader, test_loader
