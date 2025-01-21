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

Model osiągnął **dokładność** (accuracy) na poziomie **83%**.\
Precyzja (precision) oraz czułość (recall) w większości klas są wysokie, a F1-score dla wielu klas ma wartość 1.00.\
W kilku klasach (np. -1, 9, 14) wartości precyzji, recall i F1-score są równe 0.00, co oznacza, że model nie przewidział żadnych przykładów dla tych klas. Warto zwrócić uwagę, że te klasy mają tylko 1 lub 2 próbki w zbiorze testowym, co może tłumaczyć ten wynik. Takie klasy mogą być trudne do przewidzenia z powodu niewielkiej reprezentacji.\
Średnie wartości precision (0.76), recall (0.78) i F1-score (0.76) wskazują na dobry balans wydajności modelu we wszystkich klasach.\
Średnie ważone są wyższe, odpowiednio 0.79, 0.83, 0.80, co naturalnie wskazuje, że model lepiej przewiduje klasy, które występują częściej w zbiorze danych.

Model osiąga wysoką efektywność, zwłaszcza w przypadku klas, które występują częściej w zbiorze danych.\
W celu poprawy wyników należałoby zwiększyć reprezentację klas o niskiej liczbie próbek. Można również rozważyć wagowanie klas w procesie treningu, aby model przywiązywał większą wagę do klas rzadziej reprezentowanych.