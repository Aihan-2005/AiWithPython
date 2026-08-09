# FashionMLP Lab

**End-to-end neural network training, optimization, and experimentation with Keras 3 + TensorFlow.**

FashionMLP Lab is a portfolio-ready deep learning project built around Fashion-MNIST. It goes beyond a single notebook: the repository demonstrates model design, three Keras model-building APIs, backpropagation-driven training, learning-rate search, callbacks, checkpointing, TensorBoard, hyperparameter tuning, evaluation, and reproducible experiments.

## Why this project?

This repository turns core multilayer perceptron (MLP) theory into an engineering workflow:

- **Multiclass classification** with Softmax + sparse categorical cross-entropy
- **Sequential API** for a clean baseline MLP
- **Functional API** for a non-sequential Wide & Deep-style skip connection
- **Subclassing API** for a dynamic Python-defined MLP
- **Auxiliary output head** as a multi-output regularization experiment
- **Backpropagation + SGD/Adam** through Keras' built-in training loop
- **Learning-rate range test** (LR Finder)
- **Early stopping, model checkpointing, CSV logging, and TensorBoard**
- **Keras Tuner** with Random Search, Hyperband, or Bayesian Optimization
- **Confusion matrix, classification report, learning curves, and top-k predictions**
- **Reproducible seeds and CLI-driven experiments**

## Dataset

Fashion-MNIST contains 70,000 grayscale 28×28 images in 10 classes. The official split contains 60,000 training images and 10,000 test images. This project creates a stratified validation split from the original training data.

Classes:

| ID | Class |
|---:|---|
| 0 | T-shirt/top |
| 1 | Trouser |
| 2 | Pullover |
| 3 | Dress |
| 4 | Coat |
| 5 | Sandal |
| 6 | Shirt |
| 7 | Sneaker |
| 8 | Bag |
| 9 | Ankle boot |

## Architecture options

### 1. Sequential baseline

```text
28×28 image
   ↓
Flatten (784)
   ↓
Dense + ReLU
   ↓
Dense + ReLU
   ↓
Dropout (optional)
   ↓
Dense(10) + Softmax
```

### 2. Functional Wide & Deep classifier

```text
                         ┌──────────── wide/raw flattened features ────────────┐
28×28 → Flatten ─────────┤                                                     ├→ Concatenate → Softmax(10)
                         └→ Dense → Dense → ... → deep representation ────────┘
```

This demonstrates a non-sequential computation graph: the classifier can use both direct low-level features and deeper learned representations.

### 3. Functional model with auxiliary output

The main classifier uses the wide + deep representation, while an auxiliary classifier predicts directly from the deep representation. The total loss is a weighted sum:

```text
L_total = 0.85 * L_main + 0.15 * L_aux
```

### 4. Subclassed dynamic MLP

A custom `keras.Model` stores layers in `__init__()` and executes them in a Python loop inside `call()`. This demonstrates the imperative Subclassing API.

## Project structure

```text
fashion-mlp-lab/
├── README.md
├── README_FA.md
├── PROJECT_EXPLANATION_FA.md
├── RESUME_BULLETS.md
├── requirements.txt
├── pyproject.toml
├── Makefile
├── LICENSE
├── .gitignore
├── src/fashion_mlp/
│   ├── __init__.py
│   ├── constants.py
│   ├── config.py
│   ├── data.py
│   ├── models.py
│   ├── training.py
│   ├── callbacks.py
│   ├── plots.py
│   ├── utils.py
│   ├── train.py
│   ├── lr_find.py
│   ├── tune.py
│   ├── evaluate.py
│   └── predict.py
├── examples/
│   └── api_showcase.py
└── tests/
    └── test_models.py
```

## Installation

Python 3.10–3.13 is recommended with a current TensorFlow release.

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd fashion-mlp-lab

python -m venv .venv
source .venv/bin/activate       # Linux/macOS
# .venv\\Scripts\\activate      # Windows PowerShell

python -m pip install --upgrade pip
pip install -e .
```

For an NVIDIA GPU on supported systems, follow TensorFlow's official GPU installation instructions.

## Quick start

### Train the Sequential baseline

```bash
fashion-mlp-train \
  --model sequential \
  --hidden-units 300 100 \
  --optimizer adam \
  --learning-rate 0.001 \
  --batch-size 128 \
  --epochs 50
```

### Train the Functional Wide & Deep model

```bash
fashion-mlp-train --model functional --hidden-units 256 128 --dropout 0.2
```

### Train the Functional model with an auxiliary output

```bash
fashion-mlp-train --model functional_aux --hidden-units 256 128 --dropout 0.2
```

### Train the Subclassed model

```bash
fashion-mlp-train --model subclassed --hidden-units 256 128 64 --dropout 0.2
```

Every training run creates a timestamped directory under `runs/` containing:

- `best_model.keras`
- `final_model.keras`
- `history.csv`
- `training_log.csv`
- `metrics.json`
- `config.json`
- `learning_curves_loss.png` and `learning_curves_metrics.png`
- TensorBoard event files

## Learning-rate finder

Run a range test that exponentially increases the learning rate and records the loss:

```bash
fashion-mlp-lr-find \
  --start-lr 1e-5 \
  --end-lr 1 \
  --steps 500 \
  --batch-size 128
```

Outputs:

- `lr_find.csv`
- `lr_finder.png`
- a suggested learning-rate region printed to the terminal

Use the plot to identify the region where loss falls quickly but before instability/divergence.

## Hyperparameter tuning

### Random Search

```bash
fashion-mlp-tune --strategy random --max-trials 12 --epochs 30
```

### Hyperband

```bash
fashion-mlp-tune --strategy hyperband --epochs 30
```

### Bayesian Optimization

```bash
fashion-mlp-tune --strategy bayesian --max-trials 15 --epochs 30
```

The search space includes:

- number of hidden layers
- neurons per hidden layer
- dropout rate
- optimizer (`adam` or `sgd`)
- learning rate on a logarithmic scale
- batch size

The best hyperparameters are saved to JSON. The best configuration is then retrained with checkpointing, early stopping, and TensorBoard logging before final test evaluation.

## Evaluate a saved model

```bash
fashion-mlp-evaluate runs/<RUN_NAME>/best_model.keras
```

Generated evaluation artifacts include:

- test metrics JSON
- classification report JSON
- confusion matrix image
- prediction examples image

## Predict one test image

```bash
fashion-mlp-predict runs/<RUN_NAME>/best_model.keras --index 42 --top-k 3
```

Example output:

```text
True class: Sneaker
Top predictions:
  Sneaker      0.9431
  Ankle boot   0.0412
  Sandal       0.0127
```

## TensorBoard

```bash
tensorboard --logdir runs
```

Then inspect training/validation loss, accuracy, and comparisons across runs.

## Model-selection workflow

A disciplined workflow used in this project:

1. Build a simple baseline.
2. Find a sensible learning-rate range.
3. Train with validation monitoring.
4. Use early stopping instead of guessing the exact number of epochs.
5. Compare optimizer and batch-size choices.
6. Tune architecture/hyperparameters using validation performance only.
7. Evaluate the selected model **once** on the test set.
8. Report final metrics without tuning on test data.

## Concepts demonstrated

| Chapter concept | Where it appears |
|---|---|
| MLP / hidden layers | `models.py` |
| ReLU / Softmax | all classifiers |
| Sparse categorical cross-entropy | `training.py` |
| Backpropagation | Keras `fit()` training loop |
| SGD / Adam | `training.py`, `tune.py` |
| Learning rate | CLI + LR Finder |
| Batch size | CLI + tuner `HyperModel.fit()` |
| Sequential API | `build_sequential_classifier()` |
| Functional API | `build_functional_classifier()` |
| Multiple outputs | `build_functional_aux_classifier()` |
| Subclassing API | `DynamicMLP` |
| Checkpointing | `callbacks.py` |
| Early stopping | `callbacks.py` |
| Custom callback | `ValTrainRatioCallback` |
| TensorBoard | `callbacks.py` / tuner |
| Model save/restore | train/evaluate/predict CLIs |
| Random/Hyperband/Bayesian tuning | `tune.py` |
| Overfitting analysis | learning curves + validation gap |

## Tests

```bash
pytest -q
```

The tests validate forward-pass shapes for all model APIs and the multi-output model.

## Reproducibility

The project sets Python, NumPy, and Keras/TensorFlow random seeds. Exact GPU results can still vary slightly due to nondeterministic hardware kernels.

## Suggested GitHub screenshots

After running experiments, add these to your README or a `docs/images/` folder:

1. learning curves
2. confusion matrix
3. TensorBoard comparison of two learning rates
4. Keras Tuner best-trial summary
5. sample predictions

## Resume-ready summary

> Built an end-to-end Fashion-MNIST neural-network experimentation framework using Keras 3 and TensorFlow, implementing Sequential, Functional, and Subclassing APIs; automated learning-rate search, early stopping, checkpointing, TensorBoard experiment tracking, and Random Search/Hyperband/Bayesian hyperparameter optimization; added reproducible evaluation with confusion matrices and top-k predictions.

See [`RESUME_BULLETS.md`](RESUME_BULLETS.md) for shorter alternatives.

## References

- Keras: https://keras.io/
- Keras Tuner: https://keras.io/keras_tuner/
- TensorFlow Fashion-MNIST: https://www.tensorflow.org/api_docs/python/tf/keras/datasets/fashion_mnist/load_data

## License

MIT
