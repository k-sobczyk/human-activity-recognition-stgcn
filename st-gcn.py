import torch
import torch.nn as nn
import wandb
import pandas as pd
import numpy as np
from typing import Dict
from datetime import datetime
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

from data_loader import prepare_dataloaders


class PoseAugmenter:
    def __init__(self, rotation_range: float = 0.1):
        self.rotation_range = rotation_range

    def rotate_sequence(self, sequence: torch.Tensor) -> torch.Tensor:
        theta = np.random.uniform(-self.rotation_range, self.rotation_range)
        rotation_matrix = torch.tensor([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)]
        ], dtype=torch.float32)

        xy_coords = sequence[..., :2]  # Get x,y coordinates
        original_shape = xy_coords.shape

        # Reshape to 2D for rotation
        xy_coords = xy_coords.reshape(-1, 2)

        # Apply rotation
        rotated_coords = torch.matmul(xy_coords, rotation_matrix.T)
        rotated_coords = rotated_coords.reshape(original_shape)

        augmented_sequence = sequence.clone()
        augmented_sequence[..., :2] = rotated_coords

        return augmented_sequence


class SpatialGraphConv(nn.Module):
    """Spatial Graph Convolution layer."""

    def __init__(self, in_channels, out_channels, graph_nodes):
        super(SpatialGraphConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.2)

        # Learnable adjacency matrix
        self.A = nn.Parameter(torch.FloatTensor(graph_nodes, graph_nodes))
        nn.init.xavier_uniform_(self.A)

    def forward(self, x):
        # Apply spatial attention
        N, C, T, V = x.size()
        A = torch.softmax(self.A, dim=1)
        x_space = x.permute(0, 2, 3, 1).contiguous().view(N * T, V, C)
        x_space = torch.matmul(A, x_space)
        x_space = x_space.view(N, T, V, C).permute(0, 3, 1, 2)

        x = self.conv(x_space)
        x = self.bn(x)
        x = self.relu(x)
        x = self.dropout(x)

        return x


class STGCN(nn.Module):
    def __init__(
            self,
            in_channels: int = 3,
            num_class: int = 17,
            hidden_channels: int = 64,
            graph_nodes: int = 8,
            use_augmentation: bool = True
    ):
        super(STGCN, self).__init__()

        self.use_augmentation = use_augmentation
        self.augmenter = PoseAugmenter() if use_augmentation else None

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(),
            nn.Dropout(p=0.2)
        )

        # Spatial Graph Convolution
        self.spatial_conv = SpatialGraphConv(
            hidden_channels,
            hidden_channels * 2,
            graph_nodes
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(p=0.5),
            nn.Linear(hidden_channels * 2, num_class)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apply augmentation during training only
        if self.training and self.use_augmentation:
            x = self.augmenter.rotate_sequence(x)

        N, T, V, C = x.size()
        x = x.permute(0, 3, 1, 2)  # (N, C, T, V)

        x = self.conv1(x)
        x = self.spatial_conv(x)
        out = self.classifier(x)

        return out


def train_model(
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Dict,
        device: str = 'cpu'
) -> None:
    """Train the model with dynamic learning rate adjustment."""
    run = wandb.init(
        project="stgcn-exercise-classification",
        config=config,
        name=f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])

    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=0.5,
        patience=5,
        verbose=True,
        min_lr=1e-6
    )

    wandb.watch(model, criterion, log="all", log_freq=10)

    best_val_acc = 0.0
    early_stopping_counter = 0
    early_stopping_patience = 15

    for epoch in range(config['epochs']):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch_idx, (sequences, labels) in enumerate(train_loader):
            sequences, labels = sequences.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(sequences)

            loss = criterion(outputs, labels)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

        train_accuracy = train_correct / train_total

        # Validation phase
        model.eval()
        val_correct = 0
        val_total = 0
        val_loss = 0.0

        with torch.no_grad():
            for sequences, labels in val_loader:
                sequences, labels = sequences.to(device), labels.to(device)
                outputs = model(sequences)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_accuracy = val_correct / val_total

        scheduler.step(val_accuracy)
        current_lr = optimizer.param_groups[0]['lr']

        wandb.log({
            'epoch': epoch,
            'train_loss': train_loss / len(train_loader),
            'train_accuracy': train_accuracy,
            'val_loss': val_loss / len(val_loader),
            'val_accuracy': val_accuracy,
            'learning_rate': current_lr
        })

        # Update early stopping
        if val_accuracy > best_val_acc:
            best_val_acc = val_accuracy
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1

        if early_stopping_counter >= early_stopping_patience:
            print(f"Early stopping triggered after {epoch + 1} epochs")
            break

        print(f'Epoch [{epoch + 1}/{config["epochs"]}] - LR: {current_lr:.6f}')
        print(f'Train Loss: {train_loss / len(train_loader):.4f} | Accuracy: {train_accuracy:.4f}')
        print(f'Val Loss: {val_loss / len(val_loader):.4f} | Accuracy: {val_accuracy:.4f}')
        print('-' * 40)

    wandb.finish()


if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)

    wandb.login()

    df = pd.read_csv('data/data.csv')

    # Remap labels to start from 0
    unique_labels = df['exercise'].unique()
    label_map = {label: idx for idx, label in enumerate(sorted(unique_labels))}
    df['exercise'] = df['exercise'].map(label_map)

    num_classes = len(unique_labels)
    print(f"Original exercise labels: {sorted(unique_labels)}")
    print(f"Number of classes: {num_classes}")
    print(f"Label mapping: {label_map}")

    train_loader, val_loader = prepare_dataloaders(
        df=df,
        window_size=64,
        stride=32,
        batch_size=32,
        test_size=0.2,
        num_workers=0
    )

    config = {
        'hidden_channels': 64,
        'learning_rate': 0.001,
        'batch_size': 32,
        'epochs': 100,
        'use_augmentation': True
    }

    model = STGCN(
        in_channels=3,
        num_class=num_classes,
        hidden_channels=config['hidden_channels'],
        graph_nodes=8,
        use_augmentation=config['use_augmentation']
    )

    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device='cpu'
    )
