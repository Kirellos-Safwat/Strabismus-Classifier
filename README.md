# Strabismus Screening (Normal vs Abnormal)

A production-ready binary screening pipeline for eye images using an ensemble of four specialist CNNs and a RandomForest meta-classifier. The system flags images as NORMAL or ABNORMAL (any strabismus type).

## Highlights
- Four specialist models: Normal vs Esotropia, Exotropia, Hypertropia, Hypotropia
- Meta-ensemble (RandomForest) combining the four model scores
- Simple CLI for training, assembling from pre-trained models, and running inference
- Clean repository setup with data/models ignored from Git

## Repository structure
- `ensemble_classifier.py`: unified CLI (train from scratch or assemble from existing models)
- `predict_image.py`: CLI for single-image or folder inference
- `ensemble_explanation.py`: optional explanation and visualization (uses real metrics if artifacts exist)
- `requirements.txt`: minimal dependencies
- `.gitignore`: excludes datasets, weights, and generated outputs

## Installation
```
pip install -r requirements.txt
```
Python 3.10+ and TensorFlow 2.14+ recommended.

## Dataset
- Source: [Kaggle — Strabismus Dataset](https://www.kaggle.com/datasets/ananthamoorthya/strabismus)
- After download, extract to `STRABISMUS/` with the following subfolders:
  - `NORMAL/`, `ESOTROPIA/`, `EXOTROPIA/`, `HYPERTROPIA/`, `HYPOTROPIA/`
- This repository does not include data. Please review and comply with the dataset license/terms of use.

## Usage

### 1) Assemble and evaluate from existing models (recommended)
Place the trained weights and meta-classifier at repo root:
- `best_esotropia_classifier.h5`
- `best_exotropia_classifier.h5`
- `best_hypertropia_classifier.h5`
- `best_hypotropia_classifier.h5`
- `meta_classifier.pkl`

Optionally, provide a dataset for evaluation at `STRABISMUS/` with subfolders:
`NORMAL/`, `ESOTROPIA/`, `EXOTROPIA/`, `HYPERTROPIA/`, `HYPOTROPIA/`.

Build and (optionally) evaluate:
```
python ensemble_classifier.py assemble
```
The script prints two numbers:
- Keras-head ensemble accuracy (expected low; head is untrained)
- Meta-ensemble accuracy (RandomForest) — the primary performance metric

### 2) Train everything from scratch (optional)
```
python ensemble_classifier.py train
```
This will:
- Create four binary datasets
- Train the four specialist CNNs
- Train the RandomForest meta-classifier
- Build a Keras ensemble head (for completeness)
- Evaluate and save artifacts

### 3) Inference (single image or folder)
```
# Single image
python predict_image.py --image path/to/image.jpg

# Folder of images (jpg/jpeg/png)
python predict_image.py --folder path/to/images
```
The tool prints the four per-model scores, final NORMAL/ABNORMAL prediction, and a confidence value.

## Visual explanation (optional)
If models and dataset are available, generate plots with real metrics:
```
python ensemble_explanation.py
```
Creates `ensemble_architecture_explanation.png` showing per-model and ensemble accuracies.

## Results (example)
- Best individual model: ~0.88 accuracy
- Meta-ensemble (RandomForest): ~0.92–0.95 accuracy (dataset dependent)

## Data and model management
This repository ignores local data and weights by default (`.gitignore`):
- `STRABISMUS/`, `BINARY_*/`
- `*.h5`, `*.keras`, `meta_classifier.pkl`
- Generated plots (`*.png`, `*.jpg`)
If you wish to publish trained weights, use Git LFS or attach them to a GitHub Release.

## Reproducibility tips
- Fix seeds (TensorFlow, NumPy, scikit-learn) for stable runs
- Record package versions (`pip freeze > requirements-lock.txt`)
- Keep consistent preprocessing (`rescale=1./255`)

## License
Add your chosen license here (e.g., MIT).

## Citation
If you use this project, please cite it or link back to the repository.
