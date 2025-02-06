import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc, classification_report
from sklearn.preprocessing import label_binarize
import torch


class ModelEvaluator:
    def __init__(self, vae_model, regressor, feature_extractor, device='cuda'):
        """
        Initialize model evaluator
        Args:
            vae_model: Trained Beta-VAE model
            regressor: Trained PotassiumRegressor model
            feature_extractor: HandcraftFeatureExtractor instance
            device: Computing device (cuda/cpu)
        """
        self.vae_model = vae_model
        self.regressor = regressor
        self.feature_extractor = feature_extractor
        self.device = device

        # Set models to evaluation mode
        self.vae_model.eval()
        self.regressor.eval()

    def evaluate_reconstruction(self, data_loader, num_examples=5):
        """
        Evaluate and visualize VAE reconstruction performance
        Args:
            data_loader: DataLoader containing test data
            num_examples: Number of examples to visualize
        """
        with torch.no_grad():
            for i, (data, _) in enumerate(data_loader):
                if i >= num_examples:
                    break

                data = data.to(self.device)
                recon_batch, mu, logvar = self.vae_model(data)

                # Plot original and reconstruction for both leads
                plt.figure(figsize=(15, 10))
                for lead_idx, lead_name in enumerate(['Lead II', 'Lead V5']):
                    # Original signal
                    plt.subplot(2, 2, 2 * lead_idx + 1)
                    plt.title(f'Original - {lead_name}')
                    plt.plot(data[0, lead_idx].cpu().numpy())
                    plt.grid(True)

                    # Reconstructed signal
                    plt.subplot(2, 2, 2 * lead_idx + 2)
                    plt.title(f'Reconstruction - {lead_name}')
                    plt.plot(recon_batch[0, lead_idx].cpu().numpy())
                    plt.grid(True)

                plt.tight_layout()
                plt.show()

                # Calculate and print reconstruction error
                mse = torch.nn.functional.mse_loss(recon_batch, data).item()
                mae = torch.nn.functional.l1_loss(recon_batch, data).item()
                print(f'Reconstruction Error - MSE: {mse:.6f}, MAE: {mae:.6f}')

    def evaluate_predictions(self, data_loader):
        """
        Evaluate regressor predictions
        Args:
            data_loader: DataLoader containing test data
        Returns:
            Dictionary containing true and predicted values
        """
        true_values = []
        pred_values = []

        with torch.no_grad():
            for data, labels in data_loader:
                data = data.to(self.device)
                labels = labels.to(self.device)

                # Get VAE features
                _, mu, _ = self.vae_model(data)

                # Get handcrafted features
                handcraft_features = self.feature_extractor.extract_features(data).to(self.device)

                # Make predictions
                predictions = self.regressor(mu, handcraft_features)

                true_values.extend(labels.cpu().numpy())
                pred_values.extend(predictions.cpu().numpy())

        return {
            'true_values': np.array(true_values),
            'pred_values': np.array(pred_values)
        }

    def plot_confusion_matrix(self, true_values, pred_values, thresholds=[4.0, 5.0, 6.0, 7.0, 8.0]):
        """
        Plot confusion matrix for potassium level ranges
        """

        # Convert continuous values to categories
        def categorize(values):
            categories = np.zeros_like(values, dtype=int)
            for i, threshold in enumerate(thresholds):
                categories[values >= threshold] = i + 1
            return categories

        y_true_cat = categorize(true_values)
        y_pred_cat = categorize(pred_values)

        # Create confusion matrix
        cm = confusion_matrix(y_true_cat, y_pred_cat)

        # Plot
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')

        # Create labels
        labels = [f'<{thresholds[0]}'] + \
                 [f'{thresholds[i]}-{thresholds[i + 1]}' for i in range(len(thresholds) - 1)] + \
                 [f'>{thresholds[-1]}']

        plt.title('Confusion Matrix - Potassium Level Ranges')
        plt.xlabel('Predicted Range')
        plt.ylabel('True Range')
        plt.xticks(np.arange(len(labels)) + 0.5, labels, rotation=45)
        plt.yticks(np.arange(len(labels)) + 0.5, labels, rotation=0)
        plt.tight_layout()
        plt.show()

        # Print classification report
        print("\nClassification Report:")
        print(classification_report(y_true_cat, y_pred_cat,
                                    target_names=labels,
                                    zero_division=0))

    def plot_prediction_analysis(self, true_values, pred_values):
        """
        Comprehensive prediction analysis plots
        """


        # 2. Bland-Altman plot
        difference = pred_values - true_values
        mean = (pred_values + true_values) / 2

        plt.figure(figsize=(10, 8))
        plt.scatter(mean, difference, alpha=0.5)
        plt.axhline(y=np.mean(difference), color='r', linestyle='--',
                    label=f'Mean difference: {np.mean(difference):.3f}')
        plt.axhline(y=np.mean(difference) + 1.96 * np.std(difference), color='g',
                    linestyle='--', label='+1.96 SD')
        plt.axhline(y=np.mean(difference) - 1.96 * np.std(difference), color='g',
                    linestyle='--', label='-1.96 SD')

        plt.xlabel('Mean of True and Predicted Values (mEq/L)')
        plt.ylabel('Difference (Predicted - True) (mEq/L)')
        plt.title('Bland-Altman Plot')
        plt.legend()
        plt.grid(True)
        plt.show()

    def evaluate_all(self, test_loader):
        """
        Run complete model evaluation
        Args:
            test_loader: DataLoader containing test data
        """
        print("1. Evaluating VAE Reconstruction...")
        self.evaluate_reconstruction(test_loader)

        print("\n2. Evaluating Regressor Predictions...")
        results = self.evaluate_predictions(test_loader)
        true_values = results['true_values']
        pred_values = results['pred_values']

        # Calculate regression metrics
        mse = np.mean((true_values - pred_values) ** 2)
        mae = np.mean(np.abs(true_values - pred_values))
        rmse = np.sqrt(mse)
        r2 = 1 - np.sum((true_values - pred_values) ** 2) / np.sum((true_values - np.mean(true_values)) ** 2)

        print("\nRegression Metrics:")
        print(f"Mean Squared Error (MSE): {mse:.4f}")
        print(f"Mean Absolute Error (MAE): {mae:.4f}")
        print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
        print(f"R² Score: {r2:.4f}")

        print("\n3. Plotting Confusion Matrix...")
        self.plot_confusion_matrix(true_values, pred_values)

        print("\n4. Plotting Prediction Analysis...")
        self.plot_prediction_analysis(true_values, pred_values)
