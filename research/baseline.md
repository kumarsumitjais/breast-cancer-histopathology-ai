# Experiment 06.3A — Baseline record

## Objective

Evaluate candidate-group-balanced sampling with DenseNet121 while keeping the underlying train/validation/test partitions and model configuration consistent with Experiment 06.1.

## Model and input pipeline

- Backbone: DenseNet121, ImageNet initialization, `include_top=False`.
- Input: RGB 224×224.
- Resize: TensorFlow resize with antialiasing.
- Preprocessing: `tf.keras.applications.densenet.preprocess_input`.
- Augmentation during training: horizontal flip, rotation 0.08, zoom 0.10, contrast 0.10.
- Head: GlobalAveragePooling2D → BatchNormalization → Dropout(0.35) → Dense(128, ReLU) → Dropout(0.25) → Dense(1, sigmoid).
- Stage A: frozen backbone, Adam 1e-4, maximum 8 epochs.
- Stage B: final `conv5_block*` and `norm5*` layers unfrozen; BatchNormalization layers frozen; Adam 1e-5, maximum 12 epochs.
- Batch size: 8; FP32.
- Class weights: benign 1.5574, malignant 0.7364.
- Sampling: class chosen proportional to original train image counts; candidate group uniformly within class; image uniformly within group; replacement.

## Latest rerun results

Validation-selected threshold: 0.88.

| Metric | Development test |
|---|---:|
| Accuracy | 87.91% |
| ROC-AUC | 91.29% |
| Malignant precision | 87.38% |
| Malignant sensitivity | 96.00% |
| Benign specificity | 71.02% |
| Malignant F1 | 91.48% |
| Balanced accuracy | 83.51% |

Confusion matrix: TN=348, FP=142, FN=41, TP=983 (test n=1,514).

## Interpretation boundaries

The test partition has been used for repeated exploratory model comparisons and error analysis. Treat these values as development results, not an untouched final estimate. Candidate-group IDs are filename-derived and must be checked against official BreakHis metadata before asserting patient-level independence. The dataset split is not evidence of clinical validation.
