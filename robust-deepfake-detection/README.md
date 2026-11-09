# Robust Deepfake Detection Using Lightweight Deep Learning Models

Research-oriented comparison of **MobileNetV3-Small** and **ResNet18** for classifying real versus synthetic facial images, with transfer learning, corruption robustness evaluation, and Grad-CAM explanations.

> **Research status:** Experimental manuscript and original notebook. This repository does not claim a published/accepted paper, production-ready detector, or independently reproduced training run.

## Highlights

- Binary classification on **RVF10K** (10,000 balanced images): 7,000 development images and 3,000 held-out images.
- Earlier fine-tuned checkpoints: **MobileNetV3-Small 83.13%** accuracy / **0.9173** ROC-AUC; **ResNet18 83.87%** accuracy / **0.9198** ROC-AUC on 3,000 images.
- Extended MobileNetV3-Small: **85.63%** held-out accuracy and **0.9355** ROC-AUC. **No matching extended ResNet18 held-out evaluation was recorded**; its internal validation result must not be treated as a test score.
- Model sizes: **1,518,881** MobileNetV3-Small parameters vs **11,177,025** ResNet18 parameters.
- **38 model-condition evaluations** spanning JPEG compression, Gaussian blur, Gaussian noise, brightness changes, and resolution reduction, using the *earlier fine-tuned* checkpoints.
- Grad-CAM visualization and paired explanation-consistency analysis.

## Repository contents

| Path | Description |
| --- | --- |
| `notebooks/deepfake_detection_original.ipynb` | Original experimental notebook, retained as supplied, including saved outputs and historical cells. |
| `paper/research_manuscript.pdf` | Research manuscript describing methodology, results, and limitations. |
| `requirements.txt` | Python package names needed for notebook dependencies; versions should be validated in a fresh environment. |
| `DATASET_SETUP.md` | Dataset and execution notes. |

## Getting started

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter lab
```

Open `notebooks/deepfake_detection_original.ipynb`. **Before running**, obtain RVF10K according to its source terms, adjust `DATASET_PATH` and other `/home/student/...` paths, and review the notebook from top to bottom. It contains saved historical outputs, non-sequential experiment cells, and environment-specific assumptions; it is **not guaranteed to run end-to-end unchanged**. The dataset and trained model weights are **not included**.

## Experimental protocol

The manuscript describes 224 × 224 RGB inputs with ImageNet normalization, transfer learning, and a 5,950/1,050 stratified train/internal-validation split from the 7,000-image development directory. The 3,000-image directory was used for held-out evaluation. JPEG tests use all 3,000 images; other corruption families use a stratified 500-image subset. See the manuscript for checkpoint-stage differences, limitations, and detailed results.

## Important limitations

- Static real-vs-synthetic image classification, **not** a universal video/face-swap detector.
- No verified cross-dataset or unseen-generator generalization.
- The 500-image corruption subset lacks a recorded matched clean baseline.
- CPU evaluation-loop timing includes preprocessing and is not pure model inference latency.
- Grad-CAM heatmaps are diagnostic, not proof of faithful explanations.
- Some extended-checkpoint results and artifacts are unavailable; do not conflate checkpoint stages.

## Research team

Aditya Shenoy B, Rudransh Kapoor, Srijan Chauhan, and Sudhina Kumar G K — Manipal Institute of Technology, Manipal Academy of Higher Education, India. See manuscript for full attribution.

## License and data

No code license is granted by default. Confirm permission from all contributors and choose a license before public reuse. RVF10K data are not redistributed here; follow the dataset provider's licensing and access conditions.
