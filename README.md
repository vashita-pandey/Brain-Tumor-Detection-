# Brain-Tumor-Detection-
Deep learning pipeline for MRI brain tumor classification (CNN, ResNet50, EfficientNetB0) with GradCAM explainability, tumor localization, and Monte Carlo Dropout uncertainty estimation.

# 🧠 Brain Tumor Detection & Classification Using Deep Learning

> End-to-end MRI brain tumor classification into 4 categories — Glioma, Meningioma,
> Pituitary Tumor, and No Tumor — using a custom CNN and fine-tuned ResNet50,
> with GradCAM explainability, contour-based tumor localization, and
> real-image batch inference on 15 MRI scans.

---

## 📌 Project Overview

Brain tumor diagnosis requires trained neuro-radiologists manually examining MRI
scans — a slow, expensive process unavailable in many regions. This project builds
a full deep learning pipeline that:

- Classifies MRI scans into 4 tumor classes
- Achieves **95% test accuracy** with fine-tuned ResNet50
- Visually explains model decisions using **GradCAM heatmaps**
- Localizes tumor regions using **OpenCV contour detection + bounding boxes**
- Runs **batch inference on 15 real MRI images** with per-image prediction,
  confidence score, heatmap, and bounding box output

---

## 🗂️ Dataset

| Split    | Images | Per Class |
|----------|--------|-----------|
| Training | 5,600  | 1,400     |
| Testing  | 1,600  | 400       |

**Classes:** Glioma · Meningioma · No Tumor · Pituitary  
**Perfectly balanced** — no class imbalance issues.  
Source: [Brain Tumor MRI Dataset – Kaggle](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)

---

## 🏗️ Models & Results

### Model 1 — Custom CNN (Baseline)

**Architecture:**
- 3 × Conv2D blocks (32 → 64 → 128 filters) + BatchNormalization + MaxPooling + Dropout(0.25)
- Flatten → Dense(256) → BatchNorm → Dropout(0.5) → Softmax(4)
- Callbacks: EarlyStopping (patience=5), ReduceLROnPlateau

**Separate preprocessing pipeline:** manual rescale (÷255) + augmentation  
(rotation ±20°, zoom 15%, width/height shift 10%, horizontal flip, brightness [0.8–1.2])

**Results (test set):**

| Class      | Precision | Recall | F1   |
|------------|-----------|--------|------|
| Glioma     | 0.92      | 0.67   | 0.77 |
| Meningioma | 0.80      | 0.46   | 0.58 |
| No Tumor   | 0.88      | 0.90   | 0.89 |
| Pituitary  | 0.59      | 1.00   | 0.74 |
| **Overall**|           |        |**76%**|

---

### Model 2 — ResNet50 Transfer Learning (Best Model ⭐)

**Architecture:**
- ResNet50 base (ImageNet weights) + GlobalAveragePooling2D + Dense(256)
  + BatchNorm + Dropout(0.5) + Softmax(4)
- **Model-specific preprocessing:** `resnet50.preprocess_input` (not manual rescale)
- **Phase 1:** All base layers frozen — head only trained (10 epochs, Adam)
- **Phase 2:** Last 30 ResNet layers unfrozen — fine-tuned at `lr=1e-4`
  with EarlyStopping + ReduceLROnPlateau

**Fine-tuning training progression:**
| Epoch | Train Acc | Val Acc |
|-------|-----------|---------|
| 1     | 91.1%     | 90.9%   |
| 2     | 95.4%     | 93.1%   |
| 3     | 96.7%     | 93.4%   |

**Results (test set):**

| Class      | Precision | Recall | F1   |
|------------|-----------|--------|------|
| Glioma     | 0.99      | 0.83   | 0.90 |
| Meningioma | 0.90      | 0.97   | 0.94 |
| No Tumor   | 0.95      | 1.00   | 0.97 |
| Pituitary  | 0.97      | 1.00   | 0.98 |
| **Overall**|           |        |**95%**|

---

### Model 3 — EfficientNetB0 (Analysis)

Trained for analysis and comparison. Exhibited **BatchNorm collapse** in the
frozen-layer configuration (25% accuracy = random chance on 4 classes).
Fine-tuning (last 40 layers, lr=1e-4) recovered to ~43% — still underperforming
due to frozen BatchNorm statistics mismatch. Documented as a learning exercise
in transfer learning failure modes.

---

## 🔬 Key Features & Techniques

### Separate Preprocessing Pipelines
A key improvement over naive implementations: the CNN and ResNet use **entirely
separate data generators**:
- CNN: manual `rescale=1./255` with standard augmentation
- ResNet: `resnet50.preprocess_input` (which applies ImageNet mean subtraction
  and channel scaling) — using the wrong preprocessor here would silently
  degrade performance

### GradCAM – Explainability
- Built using `tf.GradientTape` targeting ResNet layer `conv5_block3_out`
- Gradients of the predicted class score w.r.t. the last conv layer activations
- Pooled to generate importance weights per feature map
- Heatmap normalized with `tf.maximum(heatmap, 0) / (tf.reduce_max + 1e-8)`
  (epsilon added to prevent division-by-zero)
- Overlaid on original MRI at 40% opacity using `cv2.COLORMAP_JET`

### Tumor Localization with Bounding Boxes
- GradCAM heatmap thresholded at activation value 120 (tuned from initial 180)
- `cv2.findContours` with `RETR_EXTERNAL` + `CHAIN_APPROX_SIMPLE`
- Largest contour (by area, > 50px) selected as tumor region
- Bounding rectangle drawn and coordinates logged per image

### Batch Inference on 15 Real MRI Images
Full pipeline applied to 15 real MRI images (`img1.jpg`–`img15.jpg`):
- Per-image: preprocessing → ResNet prediction → confidence score →
  GradCAM heatmap → bounding box
- Summary grid visualization (original + localized, 5 columns)

