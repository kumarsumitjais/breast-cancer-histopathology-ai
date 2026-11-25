# Model card — Experiment 06.3A DenseNet121

## Intended use

Educational and research demonstration of binary benign/malignant classification on breast histopathology images similar to the training dataset.

## Out-of-scope use

- Clinical diagnosis or screening
- Treatment selection or patient triage
- Replacing pathology review
- Estimating individual patient risk
- Claims of clinical-grade accuracy

## Model details

DenseNet121 initialized with ImageNet weights and fine-tuned on the project's BreakHis-derived training split. The output is a sigmoid score interpreted as the model's malignant probability. The operating threshold in this build is 0.88, selected on validation balanced accuracy with malignant sensitivity as tie-breaker.

## Performance

Latest development-test metrics are documented in `baseline.md`. They are dataset-specific and should not be interpreted as expected performance on arbitrary uploads or external institutions.

## Limitations and risks

- Candidate-group IDs require verification against official patient metadata.
- Existing test partition has been used during development.
- BreakHis-like images may differ from real-world slides, scanners, stains, magnifications, and workflows.
- Class-wise and candidate-group-wise performance is uneven; fibroadenoma false positives and papillary carcinoma false negatives were prominent in exploratory error analysis.
- Grad-CAM is a visualization of model gradients, not a clinically validated explanation.
- No prospective, external, or clinical validation is established.

## Human oversight

Any research output must be reviewed by qualified pathology professionals. Do not use predictions for patient care.
