# Facial Emotion Recognition (FER)
The code represents research project conducted at the Signal and Image Processing Lab at the Technion - Israel Institute of Technology.

## 🎯 Project Goals & Objectives

The main goal is to address the challenge of emotion classification from facial images. The research focuses on the trade-off between small high-quality datasets and large but complex datasets.

* **Implement an SAE (Stack Autoencoder)** model and evaluate its performance.
* **Utilize Data Augmentation** to balance and increase data diversity to improve generalization.
* **Compare leading network architectures** and improve performance.

## Part 1: Stacked Autoencoder (SAE)

This model is specifically designed for the first phase of the project.

### How to Run the SAE Model
Run the script from your terminal using the following format:

```bash
python AutoEncoder.py --dataset_type small --data_dir "/path/to/your/dataset" --output_dir "./sae_results" --epochs 300
```

*Command-Line Arguments:*
* dataset_type (Required): Specify 'big' or 'small' to indicate the type of dataset you are loading.

* data_dir (Required): The absolute or relative path to the root folder of your dataset.

* output_dir (Optional): The folder where the generated graphs and confusion matrices will be saved. Defaults to a folder named results in your current directory.

## Part 2:

## Independent Base Model Training

Standalone scripts that independently train and evaluate the core architectures used in Part 2 of the research (on the FER2013 Dataset):

* **`convNeXt_model.py`**: Trains a ConvNeXt architecture (default: `convnext_xlarge`).
* **`EfficientNet_model.py`**: Trains an EfficientNet architecture (default: `tf_efficientnet_b7`).

Both scripts handles data splitting, augmentation and automatically generate and save training/validation loss graphs, accuracy plots, and confusion matrices upon completion.

#### To train the ConvNeXt architecture
```python
python convNeXt_model.py
```

#### To train the EfficientNet architecture
```python
python EfficientNet_model.py
```

### Required Configuration Before Running

### 1. Dataset Path
You must update the `data_root` variable to point to your local dataset directory in the main function for both files:

```python
data_root = r"/path/to/your/dataset"
```

### 2. Output Directory for Results (Graphs & Matrices)
Located near the end of the `train_convnext()` and `train_efficientfer()` functions. You must update the destination folder for the saved `.png` files (training graphs and confusion matrices):

```python
# Update the string path inside plt.savefig() and save_graphs()
plt.savefig(os.path.join("/your/local/output/folder", "confusion_matrix.png"))
save_graphs(train_loss_history, val_loss_history, train_acc_history, val_acc_history, "/your/local/output/folder")
```
### 3. Changing the Model Architecture (Variant)
By default, the standalone scripts are set to train specific model sizes (e.g., `convnext_xlarge` and `tf_efficientnet_b7`). If you want to experiment with a smaller or larger version of the architecture , you can easily change the `variant` variable.

## Smart Hard Voting
As part of the ensemble evaluation phase, this repository includes an implementation of a **Smart Hard Voting** mechanism. This approach leverages the collective confidence of multiple independent model runs to improve overall classification accuracy and robustness.

### Required Configuration Before Running

1. **The Dataset Path:** The `data_root` path is currently empty. You must update it with the absolute path to your local dataset.
2. **Number of Voting Models:** You can customize the `runs` parameter to change how many independent models are trained to participate in the ensemble vote (default is `5`).

## Stacking Ensemble
### Pipeline Overview

The project is structured into four main scripts that must be run in sequence:

1.  **`train_base_models_heterogeneous.py`**: Trains the individual base architectures (e.g., ConvNext-XL, EfficientNet-B7) on the raw image dataset.
2.  **`create_features_hetro.py`**: Loads the trained base models, removes their classification heads, and extracts global features to create a new "Meta-Dataset."
3.  **`train_meta_hetero.py`**: Trains an MLP Meta-Learner on the extracted features to produce the final ensemble prediction.
4.  **`run_all_stacked_hetro.py`**: A master script that automates the entire 3-step pipeline.

#### To train the ConvNeXt architecture
```python
python run_all_stacked_hetro.py
```

### Configuration & Setup

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
