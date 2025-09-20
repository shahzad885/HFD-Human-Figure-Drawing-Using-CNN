# Human Figure Drawing (HFD) — Mania vs Schizophrenia Classifier

A compact transfer-learning classifier that distinguishes human figure drawings labeled **Mania** vs **Schizophrenia** using a MobileNetV2 backbone and Keras/TensorFlow. This repo contains a Colab notebook (`HFD.ipynb`) with data download, preprocessing, training, evaluation, and visualization steps. Key ideas: aggressive augmentation, class-weighting, strong regularization and careful callbacks to avoid overfitting.&#x20;

---



# Table of contents
* [Frontend with flutter](#Frontend)
* [Overview](#overview)
* [Folder / File structure](#folder--file-structure)
* [Dataset](#dataset)
* [Requirements](#requirements)
* [Quickstart (Colab)](#quickstart-colab)
* [Training details & hyperparameters](#training-details--hyperparameters)
* [Model architecture](#model-architecture)
* [Evaluation & visualizations](#evaluation--visualizations)
* [Tips & notes](#tips--notes)
* [License & Contact](#license--contact)

---
# Frontend:



https://github.com/user-attachments/assets/9c1e7ba1-d53d-4f3a-81f5-84f832b944e5




# Overview

This project trains a binary classifier on the **Human-Figure-Drawing** dataset (two folders: `Mania` and `Schizophrenia`) using transfer learning (MobileNetV2). The notebook includes multiple training variants (small head vs stronger regularization, different augmentation intensities) and utilities to plot accuracy/loss/AUC and build a confusion matrix.

---

# Folder & file structure (suggested)

```
.
├── HFD.ipynb                # main Colab notebook (data download → train → evaluate)              # place dataset here if not using Kaggle download
└── README.md
```

---

# Dataset

The notebook downloads the dataset programmatically from Kaggle (example uses `kagglehub` in Colab) and moves it into `/content/human-figure-drawing`. The dataset structure used by the code is:

```
/content/human-figure-drawing/1/hfdt/Mania
/content/human-figure-drawing/1/hfdt/Schizophrenia
```

Total images used in examples: \~230 (108 Mania + 122 Schizophrenia) in one run; other runs split differently (examples below).

---

# Requirements:

```
tensorflow>=2.10
numpy
scikit-learn
matplotlib
seaborn
kaggle    # or use kagglehub in Colab as shown
```

(If you use the notebook as-is in Colab, most packages are preinstalled. The notebook uses `kagglehub` for a direct dataset download in the provided Colab workflow.)&#x20;

---

# Quickstart (Colab)

1. Open `HFD.ipynb` in Google Colab.
2. Run the dataset download cell (it uses `kagglehub.dataset_download("manuldas/human-figure-drawing")` in the notebook).&#x20;
3. Run preprocessing → model build → training cells. The notebook includes cells for multiple training configurations (lighter head vs heavier regularization).&#x20;

If you prefer to run locally:

* Download the dataset from Kaggle manually and place it under `data/` with the same folder structure.
* Run a `train.py` script that mirrors the notebook steps.

---

# Training details & hyperparameters

The notebook shows several training configurations. below are the primary values used in examples:

* Image size: `IMG_SIZE = (224, 224)`.&#x20;
* Batch sizes used in examples: `BATCH_SIZE = 8` (one run) and `BATCH_SIZE = 16` (another enhanced run).
* Train/validation splits used: examples show `test_size=0.10` and `test_size=0.15` (stratified). Example training sizes reported: `Training set: 207, Validation set: 23` and another run `Training set: 195, Validation set: 35`.
* Optimizer & LR:

  * Example 1: `Adam(learning_rate=1e-4)` (simple run).&#x20;
  * Example 2: schedule example using `ExponentialDecay(initial_lr=2e-4, decay_rate=0.9)` + `Adam(learning_rate=lr_schedule, clipnorm=1.0)`.&#x20;
* Loss: `binary_crossentropy`
* Metrics: `["accuracy"]` or `["accuracy", AUC]` for the stronger run.
* Callbacks:

  * `EarlyStopping` (monitor `val_accuracy` or `val_loss`) with patience (15–20) and `restore_best_weights=True`.
  * `ReduceLROnPlateau` (factor 0.5, patience 5–7, min\_lr=1e-5).
  * `ModelCheckpoint('best_model.h5' or 'best_model.keras')` saving best by `val_loss`.&#x20;
* Class weights are computed to compensate for class imbalance (example uses numpy-based formula).&#x20;

---

# Data preprocessing & augmentation

Training augmentation (examples included):

* random flip left/right and up/down
* random brightness/contrast/saturation/hue (varied intensity: `max_delta=0.2` → `0.3` depending on experiment)
* random rotation (small angles) and random cropping / resize (random zoom)
* optional random Gaussian noise injection (probabilistic)
* `tf.keras.applications.mobilenet_v2.preprocess_input` used as final preprocessing step.

Example preprocessing pipeline uses `tf.data.Dataset.map(...).cache().batch(BATCH_SIZE).shuffle(...).prefetch(tf.data.AUTOTUNE)` for efficiency.&#x20;

---

# Model architecture

Two common heads are shown in the notebook:

1. **Medium head (example A)** — regularized MLP:

```py
base_model = MobileNetV2(include_top=False, weights='imagenet', input_shape=(224,224,3))
base_model.trainable = False
x = base_model(inputs, training=False)
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu', kernel_regularizer=l2(0.001))(x)
x = Dropout(0.5)(x)
x = Dense(64, activation='relu', kernel_regularizer=l2(0.001))(x)
x = Dropout(0.3)(x)
outputs = Dense(1, activation='sigmoid')(x)
```

(Compiled with `Adam(1e-4)` and `binary_crossentropy`).&#x20;

2. **Smaller, heavily-regularized head (example B)** — stronger regularization/dropout:

```py
x = GlobalAveragePooling2D()(base_model(inputs))
x = Dense(64, activation='relu', kernel_regularizer=l2(0.005),
          activity_regularizer=l1(0.002))(x)
x = Dropout(0.6)(x)
outputs = Dense(1, activation='sigmoid', kernel_regularizer=l2(0.005))(x)
```

(Uses `Adam` with learning rate schedule and tracks AUC).

Notes:

* In one configuration the notebook unfreezes only the *last 5* layers of MobileNetV2 for light fine-tuning; in other configs the base is fully frozen to reduce overfitting.

---

# Evaluation & visualizations

The notebook includes:

* training/validation accuracy and loss plots (matplotlib).&#x20;
* AUC curves when AUC is tracked.&#x20;
* Confusion matrix (sklearn + seaborn heatmap) and classification report (precision/recall/f1).&#x20;

---

# How to reproduce (concise)

1. Open `HFD.ipynb` on Colab.
2. Run dataset download cell (kagglehub or Kaggle CLI).&#x20;
3. Run the preprocessing cell(s) to build `train_ds` and `val_ds`. Adjust `IMG_SIZE` / `BATCH_SIZE` if needed.&#x20;
4. Inspect and choose the model-head cell you prefer (regularized head or simpler head).
5. Run training cell (the notebook uses callbacks and often saves `'best_model.h5'` or `'best_model.keras'`).&#x20;
6. Run evaluation cells (plots + confusion matrix).&#x20;

---

# Practical tips & troubleshooting

* **Small dataset caution**: dataset is small (≈200 images). Strong augmentation, heavy regularization, freezing the backbone, and class weights are important to avoid overfitting.
* **If val accuracy fluctuates**: try switching from monitoring `val_accuracy` → `val_loss` for EarlyStopping, or increase patience.&#x20;
* **Model saving**: the notebook uses `ModelCheckpoint` to persist best weights to `'best_model.h5'` or `'best_model.keras'`. Load that file after training for evaluation.&#x20;
* **Batch size / memory**: examples use `BATCH_SIZE=8` or `16` — choose based on GPU/Colab memory.

---

# Results (example runs)

* Example run A (lighter run): `Training set: 207 images`, `Validation set: 23 images`, training run shown for up to 100 epochs with EarlyStopping.&#x20;
* Example run B (stronger/reg schedule): `Training set: 195 images`, `Validation set: 35 images`, tracks accuracy + AUC and saved best model to `best_model.h5`.&#x20;

(Exact metrics vary by random seed, augmentation, and whether last layers were unfrozen. See the notebook training plot cells.)

---

# License & contact

This example code is provided for educational/research purposes. If you adapt the code for publication or distribution, please add an appropriate license to this repo.


