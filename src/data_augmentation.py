import numpy as np
import pandas as pd
from data_viz_3d import create_3d_visualization


def augment_pose_data(df: pd.DataFrame, noise_std: float = 0.005, rotation_range: float = 0.1, scale_range: float = 0.1) -> pd.DataFrame:
    """TESTS: Augment pose data by adding noise, rotation, and scaling to the original coordinates."""
    augmented_df = df.copy()

    x_cols = [col for col in df.columns if col.endswith('_x')]
    y_cols = [col for col in df.columns if col.endswith('_y')]
    z_cols = [col for col in df.columns if col.endswith('_z')]

    for cols in [x_cols, y_cols, z_cols]:
        for col in cols:
            noise = np.random.normal(0, noise_std, size=len(df))
            augmented_df[col] += noise

    theta = np.random.uniform(-rotation_range, rotation_range)
    rotation_matrix = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)]
    ])

    for i in range(len(df)):
        xy_points = np.column_stack([
            augmented_df.loc[i, x_cols],
            augmented_df.loc[i, y_cols]
        ])
        rotated_points = xy_points @ rotation_matrix.T
        augmented_df.loc[i, x_cols] = rotated_points[:, 0]
        augmented_df.loc[i, y_cols] = rotated_points[:, 1]

    scale_factor = np.random.uniform(1 - scale_range, 1 + scale_range)
    for cols in [x_cols, y_cols, z_cols]:
        augmented_df[cols] *= scale_factor

    return augmented_df


def generate_augmented_dataset(input_csv: str, output_csv: str, n_augmentations: int = 5):
    """Generate and save augmented versions of the original pose dataset."""
    df = pd.read_csv(input_csv)
    augmented_dfs = [df]

    for _ in range(n_augmentations):
        augmented_df = augment_pose_data(df)
        augmented_dfs.append(augmented_df)

    final_df = pd.concat(augmented_dfs, ignore_index=True)
    final_df.to_csv(output_csv, index=False)
    print(f"Saved augmented dataset with {len(final_df)} samples to {output_csv}")

    return final_df


if __name__ == "__main__":
    input_csv = "../data/data.csv"
    output_csv = "../data/augmented_data.csv"

    augmented_df = generate_augmented_dataset(
        input_csv=input_csv,
        output_csv=output_csv,
        n_augmentations=5
    )

    print("\nVisualizing original data...")
    create_3d_visualization(input_csv, frames_to_use=8000)

    print("\nVisualizing augmented data...")
    create_3d_visualization(output_csv, frames_to_use=8000)
