# ItemKNN comparison pipeline

This folder contains the RetailRocket **ItemKNN** workflow used for **comparison only** within Task 1.

## Purpose

The ItemKNN notebooks were prepared to provide a collaborative-filtering baseline / comparison model for the recommender study. This workflow is **not intended to be the main deployed recommender** for the project. Its role is to:

- prepare RetailRocket interaction data for ItemKNN experiments,
- train and test an ItemKNN model with chronological splits,
- generate example recommendations and comparison outputs,
- support evaluation against other recommender approaches used in the project.

## Expected location

Place this README and the notebooks inside:

```text
Task 1 - recommender-model/itemknn/
```

## Input files

Create the following folder if not already present:

```text
Task 1 - recommender-model/itemknn/input/
```

Place the RetailRocket source files inside that folder:

```text
itemknn/
├── input/
│   ├── category_tree.csv
│   ├── events.csv
│   ├── item_properties_part1.csv
│   └── item_properties_part2.csv
```

These file names should remain unchanged, since the notebooks expect those names.

## Main notebooks

Suggested execution order:

1. **dataset_audit_itemknn_eda.ipynb**  
   Initial dataset audit, descriptive statistics, and figures.

2. **retailrocket_chronological.ipynb**  
   Chronological split preparation and export of RecBole-compatible train/validation/test files.

3. **retailrocket_itemknn_colab.ipynb**  
   ItemKNN model training and evaluation. This notebook was prepared for Colab-style execution but can be adapted for local use.

4. **use_trained_model.ipynb**  
   Reloading saved artefacts and generating recommendation examples / analysis outputs.

## Generated folders

The workflow writes generated artefacts to folders such as:

```text
itemknn/output/
itemknn/log/
```

These contain generated tables, figures, prepared splits, logs, and saved model artefacts. They are normally treated as reproducible outputs rather than source files.

## Installation

Create and activate a virtual environment, then install the required packages.

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Requirements file

The following `requirements.txt` is used for this ItemKNN workflow:

```text
numpy
pandas
matplotlib
scipy
torch
recbole
ray
```

## requirements.txt

Save the following as `requirements.txt` in the same `itemknn` folder:

```text
numpy
pandas
matplotlib
scipy
torch
recbole
ray
```

## Notes

- Python standard-library modules such as `os`, `sys`, `pathlib`, `shutil`, and `zipfile` do not need separate installation.
- `google.colab` is only needed when running in Google Colab and should not be added to `requirements.txt` for local installation.
- If Git should ignore generated artefacts, add `itemknn/output/` and optionally `itemknn/log/` to `.gitignore`.
- If RecBole or PyTorch installation issues occur, install PyTorch first and then re-run `pip install -r requirements.txt`.

## Minimal folder layout

```text
Task 1 - recommender-model/
├── itemknn/
│   ├── README.md
│   ├── requirements.txt
│   ├── dataset_audit_itemknn_eda.ipynb
│   ├── retailrocket_chronological.ipynb
│   ├── retailrocket_itemknn_colab.ipynb
│   ├── use_trained_model.ipynb
│   ├── input/
│   │   ├── category_tree.csv
│   │   ├── events.csv
│   │   ├── item_properties_part1.csv
│   │   └── item_properties_part2.csv
│   ├── log/
│   └── output/
```
