# Baseline model weights

Place the Experiment 06.3A final Keras checkpoint here as:

`densenet121_063A.keras`

Expected source filename:

`Experiment_06_3A_DenseNet121_GroupBalanced_best.keras`

The weights are not bundled because the training checkpoint must be copied from your training machine and its provenance verified. Run:

```bash
python backend/scripts/package_model.py --source "/absolute/path/to/Experiment_06_3A_DenseNet121_GroupBalanced_best.keras"
```

The script copies the checkpoint and writes `densenet121_063A.keras.sha256`.

Do not commit model weights until you have checked repository size, dataset/model licensing, and whether you want to use Git LFS or a release asset.
