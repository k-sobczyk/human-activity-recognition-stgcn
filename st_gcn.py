from datetime import datetime
from typing import Dict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import wandb
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from data_loader import prepare_dataloaders


#TODO I believe the model might misrepresent frames where the user is preparing for exercises (should investigate this and potentially remove these frames from the dataset)
class PoseAugmenter:
    def __init__(self, rotation_range: float = 0.1, noise_scale_factor: float = 0.005):
        self.rotation_range = rotation_range
        self.noise_scale_factor = noise_scale_factor

    def run(self, sequence, run_horizontal_flip: bool = False):
        if run_horizontal_flip:
            sequence = self.horizontal_flip_prob(sequence)
        sequence = self.rotate_sequence(sequence)
        sequence = self.distort_sequence(sequence)
        return sequence

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

    def distort_sequence(self, sequence: torch.Tensor):
        return sequence + torch.randn(sequence.size()) * self.noise_scale_factor

    def horizontal_flip(self, sequence: torch.Tensor):
        out = sequence.detach().clone()
        out[..., 0] = 1 - out[..., 0]
        return out

    def horizontal_flip_prob(self, sequence: torch.Tensor, prob: float = 0.1):
        return self.horizontal_flip(sequence) if np.random.uniform(0, 1) < prob else sequence


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
            num_stgcn_blocks: int = 2,
            hidden_channels: int = 32,
            graph_nodes: int = 8,
            use_augmentation: bool = True
    ):
        super(STGCN, self).__init__()

        self.use_augmentation = use_augmentation
        self.augmenter = PoseAugmenter() if use_augmentation else None

        conv_blocks = []
        spatial_conv_blocks = []

        curr_layer_in_channels = in_channels
        curr_layer_out_channels = hidden_channels
        for _ in range(num_stgcn_blocks):
            conv = nn.Sequential(
                nn.Conv2d(curr_layer_in_channels, curr_layer_out_channels, kernel_size=1),
                nn.BatchNorm2d(curr_layer_out_channels),
                nn.ReLU(),
                nn.Dropout(p=0.2)
            )
            conv_blocks.append(conv)

            # Spatial Graph Convolution
            spatial_conv = SpatialGraphConv(
                curr_layer_out_channels,
                curr_layer_out_channels * 2,
                graph_nodes
            )
            spatial_conv_blocks.append(spatial_conv)

            curr_layer_in_channels = curr_layer_out_channels * 2
            curr_layer_out_channels = curr_layer_in_channels * 2
        self.conv_blocks = nn.ModuleList(conv_blocks)
        self.spatial_conv_blocks = nn.ModuleList(spatial_conv_blocks)

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(p=0.5),
            nn.Linear(curr_layer_in_channels, num_class),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apply augmentation during training only
        if self.training and self.use_augmentation:
            x = self.augmenter.run(x)

        N, T, V, C = x.size()
        x = x.permute(0, 3, 1, 2)  # (N, C, T, V)

        for conv, spatial_conv in zip(self.conv_blocks, self.spatial_conv_blocks):
            x = conv(x)
            x = spatial_conv(x)
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


def run_training():
    torch.manual_seed(42)
    np.random.seed(42)

    wandb.login()

    df = pd.read_csv('data/data_for_model.csv')

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

    return model


if __name__ == "__main__":
    run_training()
