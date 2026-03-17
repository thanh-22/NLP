# Manufacturing Defect Classification from Industrial Quality Logs

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow 2.15](https://img.shields.io/badge/tensorflow-2.15-orange.svg)](https://tensorflow.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1-red.svg)](https://pytorch.org)

This repository contains the official implementation of the paper: **"Manufacturing Defect Classification from Industrial Quality Logs"**. We explore various Deep Learning architectures to automate the severity assessment of industrial defect reports.

---

## 📌 Abstract
Manufacturing systems produce vast volumes of free-form textual reports. Categorizing these by severity (CRITICAL, MAJOR, MINOR, UNCLEAR) is vital for maintenance prioritization. This project compares **CNN, BiLSTM+Attention, RCNN, and Transformers**. Our proposed **BERT-based classifier**, featuring **Attentive Pooling** and **Multi-sample Dropout**, achieves a state-of-the-art **97.3% accuracy** on domain-specific industrial data.



---

## 🏷️ Label Definitions & Severity Criteria

We define four severity levels based on industrial quality control guidelines. Understanding these labels is key to the model's logic:

| Label | Severity | Description | Example Keywords |
| :--- | :---: | :--- | :--- |
| **CRITICAL** | 🔴 | Structural failures or safety risks. Requires immediate action. | *leak, hole, fire hazard, missing part* |
| **MAJOR** | 🟠 | Significant quality issues affecting functionality or assembly. | *crack, bent, deep dent, malfunction* |
| **MINOR** | 🟡 | Cosmetic imperfections; does not affect product performance. | *scratch, stain, smudge, discoloration* |
| **UNCLEAR** | ⚪ | Ambiguous, underspecified, or administrative entries. | *check, see notes, incomplete, unclear* |

> **Note:** These labels were validated against [Tên ngành của bạn, ví dụ: Baijiu Packaging] quality standards.



---

## 🏗 Model Architecture: BERT + Custom Head
The core of our research is a modified BERT-base model designed for small, noisy industrial datasets.

- **Attentive Pooling:** Instead of using only the `[CLS]` token, we aggregate hidden states from all tokens using an attention mechanism to capture fine-grained defect cues.
- **Multi-sample Dropout:** We use multiple dropout masks (p=0.5) to create multiple subsamples, stabilizing the gradients during fine-tuning.
- **Staged Fine-tuning:** The BERT backbone is initially frozen and gradually unfrozen to prevent catastrophic forgetting.



---

## 📊 Performance Summary

### 1. Model Comparison (In-Domain)
| Model | Accuracy | Macro-F1 | Weighted-F1 |
| :--- | :---: | :---: | :---: |
| CNN (KimCNN) | 0.966 | 0.966 | 0.966 |
| BiLSTM + Attention | 0.939 | 0.939 | 0.939 |
| RCNN | 0.949 | 0.949 | 0.949 |
| **BERT + Custom Head** | **0.973** | **0.973** | **0.973** |

### 2. Ablation Study
| Variant | Accuracy | Δ |
| :--- | :---: | :---: |
| **Full Model** | **0.9761** | - |
| w/o Attentive Pooling | 0.9693 | -0.68% |
| w/o Multi-Sample Dropout | 0.9750 | -0.11% |

![Confusion Matrix](./results/confusion_matrix.png)

---

## 🛠 Installation & Usage

### Setup
```bash
git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
cd your-repo-name
pip install -r requirements.txt
