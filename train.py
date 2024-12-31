import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler


def train_model(model, train_loader, val_loader, epochs, learning_rate, class_weights):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Adam Optimizer와 Learning Rate Scheduler 설정
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2, verbose=True)

    # 가중 손실 함수 정의
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float32).to(device))

    best_val_loss = float('inf')

    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)

        train_loss /= len(train_loader.dataset)

        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * inputs.size(0)

        val_loss /= len(val_loader.dataset)

        # Learning rate scheduler update
        scheduler.step(val_loss)

        # Best model 저장
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_model.pth")
            print(f"Best model saved at epoch {epoch + 1} with validation loss: {val_loss:.4f}")

        print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

    print("Training complete. Best validation loss:", best_val_loss)


# Example training loop setup (requires DataLoader)
# Initialize DataLoader with WeightedRandomSampler
def create_dataloaders(dataset, labels, batch_size, class_weights):
    # Weighted sampler
    sample_weights = torch.tensor([class_weights[label] for label in labels])
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

    # Train DataLoader
    train_loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)
    return train_loader
