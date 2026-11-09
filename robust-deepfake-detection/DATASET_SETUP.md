# Dataset and reproducibility notes

The original notebook expects an RVF10K directory at `/home/student/Downloads/rvf10k` and uses `/home/student/deepfake_env/deepfake_results` for several outputs. Search the notebook for these absolute paths and replace them with your own paths before execution.

Expected source organization (check against your dataset download):

```text
rvf10k/
  train/
    real/
    fake/
  valid/
    real/
    fake/
```

The manuscript records 7,000 training-source images and 3,000 held-out images, each directory balanced across real/fake classes. The notebook contains multiple experimental stages and saved outputs; results may depend on execution order and checkpoints. Do not interpret stored notebook outputs as evidence of a fresh reproducible run. Avoid committing datasets, credentials, large checkpoints, or personally identifying image files without permission.
