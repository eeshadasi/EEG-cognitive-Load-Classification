# EEG-cognitive-Load-Classification
## Project Overview
This project focuses on classifying cognitive load levels using EEG data. It employs a self-supervised transformer-based model, where a masked autoencoder is pre-trained on tokenized EEG feature segments and fine-tuned for downstream cognitive load classification.

## Dataset
The dataset consists of EEG recordings with the following feature columns:
- `TP9_psd`, `TP9_de`, `AF7_psd`, `AF7_de`, `AF8_psd`, `AF8_de`, `TP10_psd`, `TP10_de`
- `Cognitive Load` (Target variable with three levels: 0, 1, 2)
- The dataset contains **1823 rows**.

## Preprocessing
1. **Filtering**: A 2nd order Butterworth bandpass filter (1-75 Hz) and a notch filter (60 Hz) are applied to remove noise.
2. **Feature Extraction**: Power Spectral Density (PSD) and Differential Entropy (DE) are computed for Delta, Theta, Alpha, Beta, and Gamma bands.
3. **Normalization**: PSD and DE features are concatenated and z-score normalized.
4. **Segmentation**: Data is tokenized into **10-second segments** for transformer model processing.

## Model Architecture
### Pre-training (Self-Supervised Learning)
- **Masked Autoencoder (MAE)**: A transformer-based encoder learns meaningful EEG representations by reconstructing masked input features.
- **Transformer Encoder**: Utilizes multi-head self-attention layers and feed-forward networks to capture temporal dependencies.

### Fine-tuning (Classification Task)
- The pre-trained encoder is either fine-tuned or frozen.
- A classification head is added to predict the cognitive load level.

## Implementation
### Dependencies
Ensure you have the required Python libraries installed:
```bash
pip install tensorflow keras numpy pandas scipy scikit-learn matplotlib ydata-profiling
```

### Training the Model
```python
from model import train_model
train_model()
```

### Evaluating the Model
```python
from model import evaluate_model
evaluate_model()
```

### Inference
```python
from inference import predict_cognitive_load
prediction = predict_cognitive_load(sample_input)
print(prediction)
```

## Results
- Achieved **high classification accuracy** on test data.
- Model learns meaningful EEG representations for cognitive load estimation.

## Future Improvements
- Optimize hyperparameters using Keras Tuner.
- Extend to real-time EEG cognitive load assessment.

## Authors
- **D.EESHA**

## References
- Relevant research papers and methodologies used for EEG-based cognitive load classification.
