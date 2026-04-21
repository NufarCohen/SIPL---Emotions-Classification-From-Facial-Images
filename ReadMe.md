# Facial Emotion Recognition (FER)
The code represents research project conducted at the Signal and Image Processing Lab at the Technion - Israel Institute of Technology.

## 🎯 Project Goals & Objectives

The main goal is to address the challenge of emotion classification from facial images. The research focuses on the trade-off between small high-quality datasets and large but complex datasets.

* **Implement an SAE (Stack Autoencoder)** model and evaluate its performance.
* **Utilize Data Augmentation** to balance and increase data diversity to improve generalization.
* **Compare leading network architectures** and improve performance.

**Stacking Ensemble**
## 📁 Pipeline Overview

The project is structured into four main scripts that must be run in sequence:

1.  **`train_base_models_heterogeneous.py`**: Trains the individual base architectures (e.g., ConvNext-XL, EfficientNet-B7) on the raw image dataset.
2.  **`create_features_hetro.py`**: Loads the trained base models, removes their classification heads, and extracts global features to create a new "Meta-Dataset."
3.  **`train_meta_hetero.py`**: Trains an MLP Meta-Learner on the extracted features to produce the final ensemble prediction.
4.  **`run_all_stacked_hetro.py`**: A master script that automates the entire 3-step pipeline.

## ⚙️ Configuration & Setup

### 1. Dataset Path
You must update the `DATA_ROOT` variable to point to your local dataset directory in the following files:
* `train_base_models_heterogeneous.py`
* `create_features_hetro.py`

```python
DATA_ROOT = r"/path/to/your/dataset"
```

### 2. Model Configuration & Customization (Important)
By default, the pipeline is configured to use `convnext_xlarge` and `tf_efficientnet_b7` (based on the `timm` library). However, you have the **option to change these architectures** to experiment with different base models, such as ResNets or Vision Transformers.

If you choose to customize the models, you **must ensure** the model names match exactly across both scripts:
1. **Base Trainer (`train_base_models_heterogeneous.py`)**: Update the `models_to_train` list with your chosen architectures.
2. **Feature Extractor (`create_features_hetro.py`)**: Update the `model_configs` list to perfectly match the choices you made in step 1.
