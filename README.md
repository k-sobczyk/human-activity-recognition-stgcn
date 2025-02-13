# Human Activity Recognition using STGCN

This project implements a Spatial Temporal Graph Convolutional Network (ST-GCN) for automated exercise recognition and classification. The model leverages deep learning techniques to understand and classify human movements by analyzing the spatial and temporal relationships between key body points during exercise execution. Built with PyTorch, the system processes motion data captured through MediaPipe's pose estimation landmarks, transforming raw spatial coordinates into meaningful movement patterns that enable accurate exercise classification.

The project demonstrates the effectiveness of graph-based deep learning approaches in understanding human motion, where body movements are represented as dynamic graphs evolving over time. This approach allows the model to capture both the spatial configuration of body parts and their temporal dynamics throughout exercise movements, providing a robust foundation for accurate activity recognition in fitness applications.

## Exercise Demonstration
Below is a demonstration of one of the exercises used for training the model. The video shows how the model analyzes continuous movement patterns to recognize specific exercises:

![Exercise Preview](docs/exercise_preview.gif)

## Data Processing and Loading

The model processes exercise motion data captured through MediaPipe landmarks (key body points) using a sophisticated data preparation pipeline. Raw landmark coordinates are transformed into meaningful sequences using a sliding window approach, where each window contains 64 frames with a stride of 32 frames, creating overlapping segments of motion data. This technique ensures that continuous motion patterns are captured effectively while maintaining temporal relationships between frames.

The data is handled through PyTorch's DataLoader, which provides efficient batch processing rather than loading all data at once. This approach offers several advantages: memory efficiency through batch processing (32 sequences at a time), automatic data shuffling for better training performance, parallel data loading with multiple workers, and optimized GPU memory handling through pin_memory. Each sequence is structured as a three-dimensional array [window_size, n_landmarks, n_features], where n_landmarks represents 8 key body points (shoulders, elbows, wrists, and hips) and n_features contains their x, y, z coordinates. This format is crucial for STGCN as it enables the model to analyze both spatial relationships between body points and their temporal evolution throughout the exercise movement.

### Practical Example
To better understand the sequence creation process, consider a 10-second exercise video recorded at 30 FPS (frames per second), resulting in 300 frames total. With our sliding window configuration (window_size=64 and stride=32), each sequence captures approximately 2.1 seconds of motion (64/30 ≈ 2.1s). The stride of 32 means each new sequence starts halfway through the previous one, creating overlapping windows that ensure smooth motion capture and no missing transitions. This approach transforms our 10-second video into roughly 8 distinct but overlapping sequences, each providing the model with a comprehensive view of the exercise motion while maintaining temporal continuity.

## Results Summary

              precision    recall  f1-score   support

    Class -1       0.00      0.00      0.00         0
     Class 0       0.82      0.69      0.75        13
     Class 1       0.74      1.00      0.85        14
     Class 2       1.00      1.00      1.00         1
     Class 3       1.00      0.50      0.67         2
     Class 4       1.00      1.00      1.00         2
     Class 5       1.00      1.00      1.00         2
     Class 6       1.00      1.00      1.00         1
     Class 7       1.00      1.00      1.00         2
     Class 8       1.00      1.00      1.00         2
     Class 9       0.00      0.00      0.00         2
    Class 10       1.00      1.00      1.00         2
    Class 11       0.67      1.00      0.80         2
    Class 12       1.00      1.00      1.00         2
    Class 13       0.67      1.00      0.80         2
    Class 14       0.00      0.00      0.00         2
    Class 15       1.00      1.00      1.00         2

    accuracy                           0.83        53
    macro avg      0.76      0.78      0.76        53
    weighted avg   0.79      0.83      0.80        53

The model achieved an **accuracy** of **83%**. \
Precision and recall are high for most classes, with F1-scores of 1.00 for many classes. In a few classes (e.g., -1, 9, 14), precision, recall, and F1-score are all 0.00, indicating that the model did not predict any examples for these classes. It is important to note that these classes only have 1 or 2 samples in the test set, which may explain this result. These classes could be difficult to predict due to their limited representation. \
The average **precision** (0.76), **recall** (0.78), and **F1-score** (0.76) indicate a good balance in the model's performance across all classes. The weighted averages are higher, at 0.79, 0.83, and 0.80 respectively, which naturally suggests that the model performs better on classes that appear more frequently in the dataset.

The model demonstrates high efficiency, especially for classes that appear more frequently in the dataset. To improve the results, increasing the representation of classes with fewer samples should be considered. Additionally, class weighting during training could be explored to make the model pay more attention to underrepresented classes.

## Running Instructions
To run the project, follow these steps:
1. Install the required libraries:
pip install -r requirements.txt
2. Run the data-exploration.ipynb notebook to explore and save data
for the model.
3. Run the model training by executing the st_gcn.py script.
By default, training logs are tracked using Weights & Biases (wandb):
```console
python src/model/st_gcn.py
```

To disable wandb logging, run:
```console
python src/model/st_gcn.py --use_wandb=False
```
