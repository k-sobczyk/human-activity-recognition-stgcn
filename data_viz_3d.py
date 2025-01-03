from math import ceil

import matplotlib
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import torch
from matplotlib.animation import FuncAnimation

matplotlib.use('TkAgg')


def prepare_sequence(df: pd.DataFrame) -> np.ndarray:
    """Prepare the sequence data from DataFrame."""
    landmarks = list(get_landmark_indices().keys())

    feature_cols = []
    for landmark in landmarks:
        feature_cols.extend([f"{landmark}_x", f"{landmark}_y", f"{landmark}_z"])

    sequence = df[feature_cols].values
    n_frames = len(df)
    n_landmarks = len(landmarks)  # 8 landmarks
    n_coordinates = 3  # x, y, z
    return sequence.reshape(n_frames, n_landmarks, n_coordinates)


def get_landmark_indices():
    """Get the indices of landmarks in the data."""
    return {
        'LEFT_SHOULDER': 0,
        'RIGHT_SHOULDER': 1,
        'LEFT_ELBOW': 2,
        'RIGHT_ELBOW': 3,
        'LEFT_WRIST': 4,
        'RIGHT_WRIST': 5,
        'LEFT_HIP': 6,
        'RIGHT_HIP': 7
    }


def get_connected_joints() -> list:
    """Define connections between joints to draw the skeleton."""
    idx = get_landmark_indices()
    return [
        # Torso
        (idx['LEFT_HIP'], idx['RIGHT_HIP']),
        (idx['LEFT_SHOULDER'], idx['RIGHT_SHOULDER']),
        (idx['LEFT_HIP'], idx['LEFT_SHOULDER']),
        (idx['RIGHT_HIP'], idx['RIGHT_SHOULDER']),

        # Left arm
        (idx['LEFT_SHOULDER'], idx['LEFT_ELBOW']),
        (idx['LEFT_ELBOW'], idx['LEFT_WRIST']),

        # Right arm
        (idx['RIGHT_SHOULDER'], idx['RIGHT_ELBOW']),
        (idx['RIGHT_ELBOW'], idx['RIGHT_WRIST'])
    ]


def create_3d_visualization(csv_path: str, frames_to_use: int = 100):
    """Create interactive 3D visualization of the pose motion."""
    print("Reading data...")
    df = pd.read_csv(csv_path)
    df = df.head(frames_to_use)
    sequence = prepare_sequence(df)
    connections = get_connected_joints()
    landmark_indices = get_landmark_indices()

    # Create figure and 3D axes
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111, projection='3d')

    # Set initial view angle
    ax.view_init(elev=-80, azim=-90)

    def update(frame):
        """Update animation frame."""
        ax.cla()

        frame_data = sequence[frame]
        x = frame_data[:, 0]
        y = frame_data[:, 1]
        z = frame_data[:, 2]

        # Store current view angles
        current_elev = ax.elev
        current_azim = ax.azim

        # Plot skeleton connections
        for start, end in connections:
            ax.plot([x[start], x[end]],
                    [y[start], y[end]],
                    [z[start], z[end]],
                    'gray', alpha=0.3, linewidth=1)

        # Add point labels
        for name, idx in landmark_indices.items():
            offset = 0.005
            label_text = name.replace('LEFT_', 'L_').replace('RIGHT_', 'R_')
            ax.text(x[idx] + offset, y[idx] + offset, z[idx] + offset,
                    label_text,
                    fontsize=6,
                    alpha=0.7)

        ax.text2D(0.05, 0.95, f'Frame: {frame}/{len(sequence)}',
                  transform=ax.transAxes)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f'3D Pose Animation - Frame {frame}')

        # Set view limits
        margin = 0.05
        ax.set_xlim(np.min(sequence[:, :, 0]) - margin,
                    np.max(sequence[:, :, 0]) + margin)
        ax.set_ylim(np.min(sequence[:, :, 1]) - margin,
                    np.max(sequence[:, :, 1]) + margin)
        ax.set_zlim(np.min(sequence[:, :, 2]) - margin,
                    np.max(sequence[:, :, 2]) + margin)

        ax.grid(True, alpha=0.2, linestyle=':')
        ax.view_init(elev=current_elev, azim=current_azim)

    anim = FuncAnimation(
        fig,
        update,
        frames=len(sequence),
        interval=50,
        blit=False,
        repeat=True
    )

    def rotate(event):
        if event.key == 'left':
            ax.view_init(azim=ax.azim + 5)
        elif event.key == 'right':
            ax.view_init(azim=ax.azim - 5)
        elif event.key == 'up':
            ax.view_init(elev=ax.elev + 5)
        elif event.key == 'down':
            ax.view_init(elev=ax.elev - 5)
        elif event.key == 'r':
            ax.view_init(elev=-80, azim=-90)

    fig.canvas.mpl_connect('key_press_event', rotate)

    print("\nControls:")
    print("- Use arrow keys to rotate the view")
    print("- Press 'r' to reset view")
    print("- Close the window to stop the animation")

    plt.show()


def create_graph(pose: torch.Tensor):
    g = nx.Graph()
    for i in range(len(pose)):
        g.add_node(i)
    pos = {i: (1 - xy[0], 1 - xy[1]) for i, xy in enumerate(pose)}

    for e1, e2 in get_connected_joints():
        g.add_edge(e1, e2)

    return g, pos


def plot_sequence(sequence: torch.Tensor, title):
    fig, axes = plt.subplots(ceil(sequence.shape[0] / 4), 4, figsize=(15, 15))
    for i, frame in enumerate(sequence):
        ax = axes[i // 4, i % 4]
        graph, pos = create_graph(frame)
        nx.draw(graph, pos, ax=ax, node_size=10, node_color='skyblue', font_size=2)
        ax.set_title(f"Frame {i + 1}")
    fig.tight_layout()
    fig.subplots_adjust(top=0.94)
    fig.suptitle(title, fontsize=12)
    fig.show()


if __name__ == "__main__":
    csv_path = "data/data.csv"
    create_3d_visualization(
        csv_path=csv_path,
        frames_to_use=8000
    )
