# Simpsons character classifier using CNNs & Transfer Learning

**Lidia Velicia Ruiz · MSc in Artificial Intelligence**

---

## Overview

Multi-class image classification problem for 29 characters from *The Simpsons*. This project compares six model architectures, from a fully connected baseline to fine-tuned deep transfer learning models, achieving **97.08% test accuracy** with the best configuration.

---

## Dataset

- **Source**: [The Simpsons Characters Dataset](https://www.kaggle.com/datasets/alexattia/the-simpsons-characters-dataset) by Alexandre Attia (Kaggle)
- **Original classes**: 41 characters
- **Classes used**: 29 (after filtering out classes with fewer than 50 training samples)
- **Final split**:

| Set | Images | Classes |
|---|---|---|
| Train | 14,362 | 29 |
| Validation | 3,077 | 29 |
| Test | 3,078 | 29 |

### Key challenges

**Class imbalance**: Some main characters have thousands of images while minor characters have only a handful. Classes below a minimum sample threshold were excluded, and a stratified train/val/test split was used to ensure class proportions are preserved across all three sets.

**Visual variability**: Images are extracted directly from episode frames, resulting in varied poses, backgrounds, lighting, and multi-character scenes. Data augmentation was applied during training (rotation, shifts, zoom, horizontal flip) to improve generalization.

**Image size heterogeneity**: Resolution varied significantly across samples. All images were resized to **160×160 px**, the best trade-off between representation quality and computational cost, and compatible with the input requirements of pretrained models used.

---

## Models

Six architectures were trained and compared. Models were selected for final test evaluation based on validation performance only, to prevent data leakage.

### Architecture overview

| Model | Type | Val Accuracy | Params |
|---|---|---|---|
| ResNet50 (fine-tuned) | Transfer Learning | **97.69%** | 24.6M |
| MobileNetV2 (fine-tuned) | Transfer Learning | **96.39%** | 2.6M |
| CNN (from scratch) | Custom CNN | 89.60% | 6.7M |
| ResNet50 (frozen) | Transfer Learning | 81.51% | 24.7M |
| MobileNetV2 (frozen) | Transfer Learning | 68.87% | 2.6M |
| Fully Connected | MLP baseline | 10.95% | 39.5M |

### Design decisions

**Fully Connected (baseline)**: A 3-layer MLP (512 → 256 → 128) with Dropout regularization. Included purely as a baseline to empirically demonstrate that fully connected networks are unsuitable for image classification: the model cannot learn spatial structure and converges to near-random performance (~11%).

**Custom CNN**: Three convolutional blocks (32 → 64 → 128 filters) with BatchNormalization, MaxPooling, and Dropout, followed by a dense classification head. Trained from scratch with standard rescaling. Reaches ~90% accuracy, confirming that convolutional inductive biases are essential for this task.

**Frozen transfer learning (ResNet50 & MobileNetV2)**: Both pretrained models used as fixed feature extractors (base layers frozen). Performance was moderate: ResNet50 reached ~81%, MobileNetV2 ~69%. The gap from fine-tuned variants confirms that ImageNet representations do not transfer directly to cartoon images — the visual domain is sufficiently different that high-level features need relearning.

**Fine-tuned transfer learning**: The key architectural modification: upper convolutional blocks unfrozen (conv4 + conv5 for ResNet50; layers from index 100 onwards for MobileNetV2), while keeping BatchNormalization layers frozen to preserve ImageNet statistics in the frozen portion. This approach produced the two best models, with a dramatic performance jump over the frozen equivalents.

> **Note on BatchNormalization**: frozen BN layers are a deliberate choice. When fine-tuning with a small learning rate on a visually different domain, allowing BN statistics to update in frozen blocks can corrupt the learned representations and destabilize training.

---

## Results

### Test set evaluation (best 2 models)

| Model | Accuracy | Precision (macro) | Recall (macro) | F1 (macro) |
|---|---|---|---|---|
| ResNet50 (fine-tuned) | **97.08%** | **95.03%** | **95.18%** | **95.06%** |
| MobileNetV2 (fine-tuned) | 95.81% | 93.30% | 92.68% | 92.83% |

Both models generalize consistently from validation to test, indicating no significant overfitting.

### Classes with lowest F1 (ResNet50, test)

| Class | F1 | Support |
|---|---|---|
| martin_prince | 0.76 | 10 |
| barney_gumble | 0.85 | 16 |
| ralph_wiggum | 0.86 | 14 |

Low-F1 classes share a common trait: small test support (10–16 samples). Performance is limited by sample size rather than model capacity.

### Error analysis

Misclassifications were qualitatively inspected. Observed error patterns:

- **Costume/disguise scenes**: Characters wearing unusual clothing confuse the model (e.g. Mr. Burns in golf attire → moe_szyslak).
- **Multi-character frames**: The "dominant character" label is ambiguous when two characters share similar visual weight in the scene (e.g. Ned Flanders + Homer → classified as Homer).
- **Labelling noise**: Some misclassified frames appear to be incorrectly labelled in the original dataset.

These are edge cases without a systematic pattern, suggesting the models have learned robust character representations rather than superficial shortcuts.

---

## Technical stack

```
Python · TensorFlow / Keras · NumPy · Pandas · Scikit-learn · Matplotlib · Seaborn · PIL
```

**Models**: Sequential API with Keras, MobileNetV2 and ResNet50 from `keras.applications`  
**Training**: Adam optimizer, ReduceLROnPlateau, EarlyStopping, ModelCheckpoint callbacks  
**Data**: ImageDataGenerator with `flow_from_dataframe` for stratified class filtering  
