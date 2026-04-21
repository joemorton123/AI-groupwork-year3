# Task 1:  Hybrid Recommender System

A comprehensive hybrid recommender system that combines collaborative filtering (ALS) with content-based filtering (TF-IDF) to generate personalized recommendations.

## Project Overview

This project implements an end-to-end recommender system using implicit feedback data. The system combines matrix factorization techniques with content similarity to provide robust recommendations with built-in explainability and monitoring capabilities.

## Workflow

The project is organized into three sequential Jupyter notebooks:

### 1. Data Preparation (`01_data_preparation.ipynb`)

**Purpose**: Load, clean, and prepare data for model training.

**Key Steps**:
- Load raw data from CSV files (`events.csv`, `item_properties_part1.csv`, `item_properties_part2.csv`, `category_tree.csv`)
- Filter events to valid types: "view", "addtocart", "transaction"
- Apply implicit weights to events:
  - View: 1.0
  - Add to cart: 3.0
  - Transaction: 5.0
- Build user and item indices for mapping between IDs and matrix indices
- Construct a sparse user-item interaction matrix using implicit feedback

**Outputs**:
- `interactions.npz`: Sparse user-item interaction matrix
- `items_aligned.csv`: Aligned item information
- `user_id_to_idx.pkl`: User ID to matrix index mapping
- `item_id_to_idx.pkl`: Item ID to matrix index mapping

### 2. Training Hybrid Recommender (`02_train_hybrid_recommender.ipynb`)

**Purpose**: Train both collaborative and content-based models and implement a hybrid recommender.

**Key Steps**:
- Load preprocessed data and artifacts from notebook 1
- **Collaborative Filtering**: Train an ALS (Alternating Least Squares) model with:
  - Factors: 64
  - Regularization: 0.01
  - Iterations: 20
- **Content-Based Filtering**: Build a TF-IDF model for item content:
  - Max features: 10,000
  - Stop words: English
- **Hybrid Recommender**: Combine both models using weighted scoring
- Evaluate recommendations using precision@k metrics

**Outputs**:
- `als_model.npz`: Trained ALS model
- `als_item_factors.npy`: ALS item latent factors
- `als_user_factors.npy`: ALS user latent factors
- `tfidf_vectorizer.pkl`: Fitted TF-IDF vectorizer
- `tfidf_matrix.npz`: Sparse TF-IDF matrix

### 3. Explainability & Monitoring (`03_explainability_and_monitoring.ipynb`)

**Purpose**: Provide interpretable explanations for recommendations and monitor system performance.

**Key Features**:
- **ALS Explainability**: Identify which latent factors contributed most to a recommendation
- **Content Explainability**: Show similarity to user's previously interacted items
- **Top Terms Analysis**: Display the most important TF-IDF terms for each item
- **Monitoring Signals**:
  - Precision drift detection
  - Cold-start problem identification
  - Recommendation override tracking
- **Diagnostic Utilities**: Tools for understanding model behavior and identifying issues

**Outputs**:
- Visualization of recommendation explanations
- Monitoring dashboards and reports in the `output/figures/` directory

## File Structure

```
TASK_1/
├── README.md                           # This file
├── 01_data_preparation.ipynb           # Data loading and preprocessing
├── 02_train_hybrid_recommender.ipynb   # Model training and hybrid implementation
├── 03_explainability_and_monitoring.ipynb  # Explainability and monitoring
└── output/
    └── figures/                        # Generated visualizations and reports
```

## Prerequisites

### Required Libraries
- pandas
- numpy
- scipy
- scikit-learn
- implicit (for ALS algorithm)
- matplotlib
- seaborn

### Installation

```bash
pip install -r requirements.txt
```

## Usage

Run the notebooks in order:

1. **Start with data preparation**:
   - Open `01_data_preparation.ipynb`
   - Ensure input CSV files are in the correct path
   - Run all cells to generate artifacts

2. **Train models**:
   - Open `02_train_hybrid_recommender.ipynb`
   - Load the artifacts generated from notebook 1
   - Run all cells to train ALS and TF-IDF models
   - Results will be saved as numpy arrays and pickle files

3. **Analyze and monitor**:
   - Open `03_explainability_and_monitoring.ipynb`
   - Load trained models and data
   - Run analysis cells for explainability and monitoring insights
   - Visualizations will be saved to `output/figures/`

## Data Format

The system expects the following input CSV files:

| File | Columns | Purpose |
|------|---------|---------|
| `events.csv` | visitorid, itemid, event, timestamp | User-item interactions |
| `item_properties_part1.csv` | itemid, properties... | Item metadata (part 1) |
| `item_properties_part2.csv` | itemid, properties... | Item metadata (part 2) |
| `category_tree.csv` | categoryid, parent_id... | Category hierarchy |

## Model Details

### Collaborative Filtering (ALS)

The Alternating Least Squares (ALS) algorithm factorizes the user-item interaction matrix into latent factors:
- Captures user preferences and item characteristics
- Works well with implicit feedback
- Scalable to large datasets

### Content-Based Filtering (TF-IDF)

The TF-IDF model analyzes item content:
- Identifies important terms for each item
- Computes similarity between items based on content
- Complements collaborative filtering for cold-start scenarios

### Hybrid Approach

Combines both models through weighted scoring:
- Leverages strengths of both approaches
- Improves robustness and coverage
- Enables contextual recommendations

## Explainability

The system provides interpretable recommendations through:

1. **Latent Factor Analysis**: Shows which model dimensions contributed to recommendations
2. **Content Similarity**: Identifies similar items the user has already interacted with
3. **Top Terms**: Displays key TF-IDF terms for recommended items
4. **Category-Based Reasoning**: Links recommendations to item categories

## Monitoring

Tracks key metrics to ensure system health:

- **Precision Drift**: Detects degradation in recommendation quality over time
- **Cold-Start Coverage**: Monitors ability to recommend to new users/items
- **Override Tracking**: Records user overrides of recommendations
- **Diversity Metrics**: Ensures recommendations aren't too homogeneous

## Output

Generated figures and reports are saved to `output/figures/` including:
- Recommendation explanation visualizations
- Monitoring dashboard screenshots
- Performance metrics plots

## Notes

- Ensure all input files are properly formatted and located before running notebook 1
- The interaction matrix is built only from events data for consistency
- ALS factors are saved both as the model object and as individual numpy arrays for cross-compatibility
- TF-IDF vectorizer and matrix are persisted for reproducibility in notebook 3