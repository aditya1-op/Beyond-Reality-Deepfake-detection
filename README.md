# Beyond Reality: Robust Deepfake Detection Using Lightweight Deep Learning Models

### A Comparative Study of MobileNetV3-Small and ResNet18 for Efficient, Robust, and Explainable Synthetic Face Detection

![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C?logo=pytorch)
![Computer Vision](https://img.shields.io/badge/Computer%20Vision-Image%20Forensics-purple)
![XAI](https://img.shields.io/badge/Explainable%20AI-Grad--CAM-green)
![Research](https://img.shields.io/badge/Project-Research-orange)

---

## Overview

The rapid advancement of generative artificial intelligence has made synthetic facial images increasingly realistic, raising concerns about digital authenticity, misinformation, identity manipulation, and information security.

**Beyond Reality** investigates whether lightweight deep learning architectures can reliably distinguish real facial images from AI-generated images while maintaining computational efficiency and robustness against common image degradations.

This research compares two pretrained convolutional neural networks:

- **MobileNetV3-Small:** A computationally efficient architecture designed for lightweight visual inference.
- **ResNet18:** A residual convolutional neural network used as a comparative baseline.

Beyond conventional classification accuracy, the project investigates how model predictions behave under JPEG compression, Gaussian blur, Gaussian noise, brightness changes, and resolution reduction.

Grad-CAM is incorporated to visualize image regions associated with model predictions and analyze the consistency of these explanations under compression.

## Research Objectives

1. Compare MobileNetV3-Small and ResNet18 for real-versus-synthetic facial image classification.
2. Evaluate the trade-off between classification accuracy and computational efficiency.
3. Investigate model robustness under five image corruption categories.
4. Examine the effect of transfer learning and extended fine-tuning.
5. Analyze prediction explanations using Grad-CAM and spatial heatmap similarity.

## Dataset

**Dataset:** RVF10K — Real vs. Fake Facial Images

| Property | Description |
|---|---|
| Total images | 10,000 |
| Real images | 5,000 |
| Synthetic images | 5,000 |
| Development images | 7,000 |
| Training split | 5,950 |
| Internal validation | 1,050 |
| Held-out evaluation | 3,000 |
| Input resolution | 224 × 224 |
| Classification | Binary: Real / Fake |

The dataset is balanced across the two classes.

Images are resized, converted into RGB format, and normalized using ImageNet statistics. Training augmentation includes horizontal flipping, limited rotation, and color jitter.

The original dataset is not included in this repository. Users must obtain it separately and configure the corresponding dataset paths.

## Model Architectures

### MobileNetV3-Small

MobileNetV3-Small uses efficient convolutional operations, inverted residual blocks, and channel attention mechanisms.

**Key characteristics:**
- Approximately 1.52 million parameters.
- ImageNet-pretrained feature extractor.
- Modified binary classification output.
- Partial backbone fine-tuning.
- Designed for computationally constrained environments.

### ResNet18

ResNet18 is a convolutional neural network that incorporates residual connections to support effective feature learning.

**Key characteristics:**
- Approximately 11.18 million parameters.
- ImageNet-pretrained backbone.
- Residual learning architecture.
- Modified binary classification head.
- Final residual stage fine-tuning.

Both models produce a single output logit, converted to a fake-class probability using the sigmoid function.

## Methodology

The experimental workflow consists of the following stages:

```text
RVF10K Dataset
       |
       v
Dataset Partitioning
       |
       v
Image Preprocessing
       |
       v
Pretrained CNN Models
       |
       +----------------------+
       |                      |
       v                      v
MobileNetV3-Small          ResNet18
       |                      |
       v                      v
Transfer Learning and Fine-Tuning
       |                      |
       +-----------+----------+
                   |
                   v
           Clean Evaluation
                   |
          +--------+--------+
          |                 |
          v                 v
   Robustness Testing   Grad-CAM Analysis
          |                 |
          +--------+--------+
                   |
                   v
         Comparative Analysis
```

### Training Configuration

The implementation uses:

- Framework: PyTorch and Torchvision
- Optimizer: Adam
- Loss function: Binary Cross-Entropy with Logits
- Initial learning rate: `0.001`
- Fine-tuning learning rate: `0.00001`
- Weight decay: `0.0001`
- Input size: `224 × 224`
- Model initialization: ImageNet pretrained weights

The study separates initial training, earlier fine-tuning, and extended fine-tuning because checkpoint selection and validation procedures differ between stages.

## Experimental Results

### Earlier Fine-Tuned Model Comparison

The following results were obtained using the same 3,000-image held-out evaluation directory.

| Metric | MobileNetV3-Small | ResNet18 |
|---|---:|---:|
| Accuracy | 83.13% | 83.87% |
| Precision | 80.45% | 83.07% |
| Recall | 87.53% | 85.07% |
| F1-score | 0.8384 | 0.8406 |
| ROC-AUC | 0.9173 | 0.9198 |
| Parameters | 1.52M | 11.18M |

ResNet18 achieves slightly higher accuracy and ROC-AUC at this stage, while MobileNetV3-Small achieves higher fake-image recall with substantially fewer parameters.

### Extended MobileNetV3-Small Results

Further fine-tuning improved the MobileNetV3-Small model.

| Metric | Result |
|---|---:|
| Accuracy | **85.63%** |
| Precision | 84.07% |
| Recall | 87.93% |
| F1-score | 0.8596 |
| ROC-AUC | **0.9355** |
| Correct classifications | 2,569 / 3,000 |

These results belong to the extended MobileNetV3-Small checkpoint.

**Important:** The extended ResNet18 model was evaluated on internal validation data, but a corresponding held-out test evaluation was not recorded. Its internal validation accuracy of 91.24% must not be directly compared with MobileNetV3-Small's 85.63% held-out accuracy.

## Robustness Evaluation

A key focus of this project is evaluating how image degradation affects deepfake detection performance.

Five corruption categories were investigated:

| Corruption | Evaluated Conditions |
|---|---|
| JPEG Compression | Quality 90, 70, 50, 30 |
| Gaussian Blur | Radius 1, 2, 3, 5 |
| Gaussian Noise | Standard deviation 5, 15, 25, 40 |
| Brightness Variation | Factors 0.6, 0.8, 1.2, 1.4 |
| Resolution Reduction | Scales 0.75, 0.50, 0.25 |

The robustness experiments comprise **38 model-condition evaluations**, with 19 conditions per architecture.

JPEG compression was evaluated on 3,000 held-out images, while the other corruption categories used a balanced 500-image subset.

### Key Findings

- MobileNetV3-Small demonstrated stronger threshold-based detection performance under severe JPEG compression.
- ResNet18 performed better under several Gaussian noise conditions.
- Both models experienced substantial degradation in fake-image recall under strong Gaussian blur.
- Brightness and resolution changes affected the architectures differently.
- Robustness depended on the type and severity of corruption rather than clean accuracy alone.

These robustness findings correspond to the earlier fine-tuned checkpoints, not the extended MobileNetV3-Small checkpoint.

## Explainable Artificial Intelligence — Grad-CAM

To improve interpretability, Gradient-weighted Class Activation Mapping (Grad-CAM) was used to visualize image regions associated with the models' classification decisions.

The analysis includes:

- Grad-CAM heatmap generation.
- Visualization of spatial attention patterns.
- Comparison of MobileNetV3-Small and ResNet18 explanations.
- Clean-versus-JPEG-compressed image comparisons.
- Cosine similarity analysis of paired explanation maps.

Twenty image indices were used per model for paired explanation-consistency analysis.

**Interpretation:** Grad-CAM provides diagnostic insight into model behavior. Heatmap similarity does not prove prediction correctness, causal feature importance, or explanation faithfulness.

## Computational Efficiency

| Property | MobileNetV3-Small | ResNet18 |
|---|---:|---:|
| Parameters | 1,518,881 | 11,177,025 |
| Relative model size | Smaller | Larger |
| Architecture | Lightweight CNN | Residual CNN |

MobileNetV3-Small uses approximately **86.4% fewer parameters** and achieved approximately **4.46× higher throughput** in the recorded CPU evaluation loop.

The CPU measurements include preprocessing, data loading, inference, and result collection. They should not be interpreted as pure forward-pass latency or deployment benchmarks.

## Technologies Used

**Programming and Deep Learning**
- Python
- PyTorch
- Torchvision
- NumPy
- Pandas

**Computer Vision and Evaluation**
- Pillow
- OpenCV
- Scikit-learn
- Matplotlib
- Seaborn

**Research Techniques**
- Transfer Learning
- CNN Fine-Tuning
- Binary Image Classification
- Corruption Robustness Evaluation
- ROC-AUC Analysis
- Confusion Matrix Analysis
- Grad-CAM Explainability

## Getting Started

### Clone the Repository

```bash
git clone https://github.com/aditya1-op/Beyond-Reality-Deepfake-detection.git
cd Beyond-Reality-Deepfake-detection
```

### Create a Virtual Environment

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.venv\Scripts\Activate.ps1
```

Or on Linux/macOS:

```bash
source .venv/bin/activate
```

### Install Dependencies

If the repository contains a `requirements.txt` file:

```bash
pip install -r requirements.txt
```

Otherwise, install the core dependencies:

```bash
pip install torch torchvision numpy pandas scikit-learn matplotlib seaborn pillow opencv-python jupyter
```

### Launch Jupyter Notebook

```bash
jupyter notebook
```

Open the project notebook and configure the dataset paths before executing the cells.

**Reproducibility note:** The original notebook contains environment-specific paths, saved experimental outputs, and checkpoint dependencies. A complete clean execution requires the RVF10K dataset, appropriate model files, and configuration of the local environment.

## Research Contributions

This project contributes an experimental comparison of lightweight and conventional CNN architectures for synthetic facial image detection.

The study combines:

- Classification performance evaluation.
- Model efficiency measurements.
- Controlled image corruption analysis.
- Class-specific error analysis.
- Extended transfer learning experiments.
- Grad-CAM explanation consistency assessment.

The contribution is an integrated experimental evaluation rather than a newly proposed neural network architecture.

## Limitations and Future Scope

The current experiments focus on real-versus-synthetic still images from RVF10K.

Important limitations include:
- Evaluation on a single dataset.
- No demonstrated cross-dataset or unseen-generator generalization.
- No video-based deepfake detection.
- No independent deployment benchmark.
- No matching held-out extended ResNet18 evaluation.
- Limited explanation-consistency sampling.

Potential future extensions include external dataset validation, additional lightweight architectures, improved robustness training, model calibration, quantization, and a web-based inference interface.

## Research Team

**Aditya Shenoy B**  
**Rudransh Kapoor**  
**Srijan Chauhan**

**Faculty Guide:** Sudhina Kumar G K

Manipal Institute of Technology  
Manipal Academy of Higher Education  
Manipal – 576104, India

## Research Manuscript

**Title:** *Robust Deepfake Detection Using Lightweight Deep Learning Models*

The associated manuscript documents the methodology, experimental design, results, robustness analysis, and explainability findings.

The manuscript should be treated as research work rather than a published IEEE Access article unless publication is independently confirmed.

## Disclaimer

This repository is intended for academic research and educational purposes. Reported results apply to the evaluated dataset, model checkpoints, and experimental conditions.

The system should not be treated as a universally reliable deepfake detector or used as the sole basis for determining the authenticity of digital media.

---

**Beyond Reality — Exploring the intersection of efficient deep learning, digital media forensics, and explainable artificial intelligence.**
