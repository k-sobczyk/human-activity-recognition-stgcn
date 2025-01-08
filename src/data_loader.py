from typing import Tuple
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split


class ExerciseDataset(Dataset):
    """Dataset class for exercise sequences and their corresponding labels."""
    def __init__(self, sequences: np.ndarray, labels: np.ndarray) -> None:
        self.sequences = torch.FloatTensor(sequences)
        self.labels = torch.LongTensor(labels)

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.sequences[idx], self.labels[idx]


def create_sequences(df: pd.DataFrame, window_size: int = 64, stride: int = 32) -> Tuple[np.ndarray, np.ndarray]:
    """Convert landmark data into sequences using sliding window technique."""
    sequences = []
    labels = []
    n_landmarks: int = 8  # shoulders, elbows, wrists, and hips
    n_features: int = 3  # x, y, z coordinates

    feature_cols = [col for col in df.columns if col not in ['frame_number', 'exercise']]

    for i in range(0, len(df) - window_size + 1, stride):
        sequence = df[feature_cols].iloc[i:i + window_size].values
        sequence = sequence.reshape(window_size, n_landmarks, n_features)
        label = df['exercise'].iloc[i:i + window_size].mode()[0]

        sequences.append(sequence)
        labels.append(label)

    return np.array(sequences), np.array(labels)


def prepare_dataloaders(df: pd.DataFrame,
                       window_size: int = 64,
                       stride: int = 32,
                       batch_size: int = 32,
                       test_size: float = 0.2,
                       num_workers: int = 2) -> Tuple[DataLoader, DataLoader]:
    """Create train and test DataLoaders with sequence data for model training."""
    print("Creating sequences...")
    sequences, labels = create_sequences(df, window_size, stride)
    print(f"Created {len(sequences)} sequences of shape {sequences.shape}")

    print("\nSplitting into train/test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        sequences, labels,
        test_size=test_size,
        random_state=42,
        stratify=labels
    )

    train_dataset = ExerciseDataset(X_train, y_train)
    test_dataset = ExerciseDataset(X_test, y_test)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    print(f"\nTraining samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")
    print(f"Number of classes: {len(np.unique(labels))}")
    print(f"Sequence shape: {sequences.shape}")

    return train_loader, test_loader
