# HiCE: Hierarchical Cognitive Evolution for Lightweight CNN OOD Generalization

This is the official implementation of the paper:  
**"Feature Attention Bias in Lightweight CNNs: Unveiling Mechanisms for Out-of-Distribution Generalization"** (submitted to *The Visual Computer*).

## 📋 Dependencies & Requirements
- Python 3.11+
- PyTorch 2.5.1+cu121
- torchvision 0.20.1+cu121
- numpy, matplotlib, scikit-learn, opencv-python

## Code File Description
### Core Training Code
- Jupyter Notebook files starting with `train` (e.g., `train_cnn.ipynb`, `train_hice_model.ipynb`):
  These are the core training codes for the model, including complete training logic such as data loading, model initialization, training process, loss function definition, and training log saving. The dataset path and hyperparameters need to be configured before running.

### Performance Comparison Experiment Code
- Files with `compare` in the file name (e.g., `compare_model_performance.ipynb`, `compare_cnn_baseline.py`):
  These are the experimental codes for model performance comparison, covering the comparative calculation of accuracy metrics under different models/hyperparameters, as well as codes for saving experimental result data and visualization.

### Shape Consistency Score + Information Entropy Calculation Code
- The file named `Calculation of Shape Consistency Score plus Information Entropy`:
  This is the execution file for Grad-CAM-based Shape Consistency IoU and Information Entropy calculation, including the computation of Shape Consistency Score, quantification of IoU (Intersection over Union) metrics, calculation of Information Entropy values, and the linkage analysis logic between Grad-CAM heatmaps and Shape Consistency results.

### Runtime Environment Instructions
- Jupyter Notebook files are recommended to run in a Python 3.8+ environment. After installing dependencies, execute the command `jupyter notebook` to start the service and open the files for execution.
- All experimental results are reproducible, and specific dependencies are listed in `requirements.txt` below.

## Quick Reproduction Steps
1. Train the model: Run the `train_*.ipynb` files to train the model from scratch (configure the dataset path and hyperparameters first).
2. Compare model performance: Execute the code files with `compare` in the name to compare the performance metrics of different trained models.
3. Calculate IoU and Information Entropy: Run the file named `Calculation of Shape Consistency Score plus Information Entropy` to obtain the IoU (Intersection over Union) and Information Entropy scores of the model's shallow, middle, and deep layers.

# HiCE: Hierarchical Cognitive Evolution for Lightweight CNN OOD Generalization
[![DOI](https://zenodo.org/badge/1177524614.svg)](https://doi.org/10.5281/zenodo.18933628)

This is the official implementation of the paper:
"Feature Attention Bias in Lightweight CNNs: Unveiling Mechanisms for Out-of-Distribution Generalization" (submitted to The Visual Computer).
