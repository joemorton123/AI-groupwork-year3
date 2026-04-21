# Fruit and Vegetable Quality Classification System

## Project Overview

This project implements an intelligent fruit and vegetable quality classification system using deep learning. The system automatically classifies produce into quality grades (A, B, C) based on visual characteristics. The pipeline includes automated dataset curation, model training, iterative refinement through self-labeling, and comprehensive model evaluation with explainability analysis.

## Project Structure

```
├── analysis/              # Dataset analysis and visualization tools
├── classifier/            # Model training and classification modules
├── evaluation/            # Model evaluation and error analysis
├── xai/                   # Explainability analysis (Grad-CAM, LIME, SHAP)
├── requirements.txt       # Python dependencies
└── pipeline.txt           # Execution pipeline documentation
```

## System Components

### 1. **Analysis Module** (`analysis/`)
- **`analyse_dataset.py`**: Analyzes dataset distribution and generates statistics
- Produces summary reports and file listings for data validation

### 2. **Classifier Module** (`classifier/`)
Core training and classification components:
- **`cluster_healthy_to_ABC.py`**: Clusters healthy produce into quality grades (A, B, C)
- **`cluster_rotten_to_C.py`**: Classifies rotten produce as grade C
- **`split_dataset.py`**: Splits dataset into training, validation, and test sets
- **`train_classifier.py`**: Trains the neural network classifier
- **`relabel_with_model.py`**: Uses trained model to re-label uncertain samples
- **`relabel_A_to_C.py`**: Refines labels for uncertain produce classified as grade C
- **`cluster_utils.py`**: Utility functions for clustering operations

### 3. **Evaluation Module** (`evaluation/`)
Comprehensive model evaluation tools:
- **`evaluate_classifier.py`**: Assesses model performance on test set
- **`evaluate_errors.py`**: Analyzes classification errors and generates error reports
- Generates confusion matrices, accuracy metrics, and error summaries

### 4. **XAI Module** (`xai/`)
Explainability and interpretability analysis:
- **`explain_model.py`**: Generates model explanations using multiple techniques
- **`condense_images.py`**: Processes and organizes explanation visualizations
- Techniques: Grad-CAM, Integrated Gradients, LIME, SHAP, and Robustness testing

## Execution Pipeline

The system operates through a 13-step pipeline that progressively refines the dataset and model:

### Stage 1: Initial Clustering (Steps 1-3)
1. **Cluster healthy produce** → Groups into quality grades A, B, C
2. **Cluster rotten produce** → Marks as grade C
3. **Clean dataset** → Removes temporary classification folders

### Stage 2: First Model Training (Steps 4-6)
4. **Reset split** → Remove previous data splits
5. **Split dataset** → Create train/validation/test sets
6. **Train initial model** → First-pass classifier training

### Stage 3: Iterative Refinement (Steps 7-9)
7. **Clean intermediate data** → Remove refined labels and uncertain samples
8. **Auto-label uncertain samples** → Use trained model to classify unclear images
9. **Refine grade C samples** → Upgrade uncertain C samples to appropriate grades

### Stage 4: Final Model Training (Steps 10-13)
10. **Reset split** → Remove previous splits
11. **Split refined dataset** → Create new train/validation/test sets with corrected labels
12. **Train final model** → Train classifier on refined dataset
13. **Evaluate performance** → Comprehensive model validation and error analysis

## Installation

### Prerequisites
- Python 3.8+
- GPU recommended for faster training (CUDA support)

### Setup
```bash
# Clone the repository
git clone <repository-url>
cd AI-groupwork-year3

# Create virtual environment (recommended)
python -m venv venv
source venv/Scripts/activate  # Windows
# or
source venv/bin/activate      # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

## Dependencies

- **Deep Learning**: PyTorch, TorchVision
- **Data Processing**: NumPy, Pandas, Pillow, OpenCV
- **Machine Learning**: Scikit-learn
- **Visualization**: Matplotlib, Seaborn
- **Explainability**: SHAP, LIME, Captum (Grad-CAM, Integrated Gradients)

## Usage

### Running the Complete Pipeline

Execute each step sequentially as documented in `pipeline.txt`:

```bash
# Prepare and cluster data
python classifier/cluster_healthy_to_ABC.py
python classifier/cluster_rotten_to_C.py

# Initial training
python classifier/split_dataset.py
python classifier/train_classifier.py

# Refine labels using model predictions
python classifier/relabel_with_model.py
python classifier/relabel_A_to_C.py

# Final training on refined data
python classifier/split_dataset.py      # Re-split with refined labels
python classifier/train_classifier.py   # Train final model

# Evaluation
python evaluation/evaluate_classifier.py
python evaluation/evaluate_errors.py
```

### Individual Components

**Dataset Analysis**
```bash
python analysis/analyse_dataset.py
```

**Model Evaluation**
```bash
python evaluation/evaluate_classifier.py
python evaluation/evaluate_errors.py
```

**Model Explainability**
```bash
python xai/explain_model.py
```

## Model Architecture

The system uses a convolutional neural network (CNN) for image classification. Pre-trained models are saved as:
- `classifier/model_first_pass.pth` - Initial model from first training
- `classifier/model_AtoC.pth` - Final model trained on refined dataset

## Output and Results

### Evaluation Results (`evaluation/results/`)
- **`metrics.txt`**: Overall accuracy, precision, recall, F1-scores
- **`per_fruit_accuracy.csv`**: Performance breakdown by fruit/vegetable type
- **`all_errors.csv`**: Complete list of misclassified samples
- **`top_errors.csv`**: Most common classification errors
- **`error_summary.txt`**: Human-readable error analysis

### Explainability Outputs (`xai/xai/`)
- **gradcam/**: Grad-CAM attention maps showing critical image regions
- **lime/**: LIME explanations for individual predictions
- **shap/**: SHAP value analysis
- **integrated_gradients/**: Feature importance maps
- **robustness/**: Robustness testing results

## Key Features

**Automated Quality Grading** - Three-tier classification system (A, B, C)
**Iterative Refinement** - Self-labeling to improve dataset quality
**Comprehensive Evaluation** - Per-fruit accuracy and error analysis
**Model Explainability** - Multiple XAI techniques for interpretability
**Error Analysis** - Identification and visualization of problem cases
**Robustness Testing** - Validation of model reliability

## Notes

- The pipeline includes explicit cleanup steps (deleting intermediate folders) to manage disk space
- The iterative refinement process significantly improves model accuracy
- Error analysis helps identify systematic misclassifications and dataset issues
- XAI tools provide insights into model decision-making for debugging and validation
