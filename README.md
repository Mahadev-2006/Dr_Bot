# 👁️ DRBot — Multimodal Diabetic Retinopathy Assistant

<p align="center">
  <img src="assets/drbot_banner.png" alt="DRBot Banner" width="800"/>
</p>

<p align="center">
  <a href="#architecture">Architecture</a> •
  <a href="#results">Results</a> •
  <a href="#setup">Setup</a> •
  <a href="#usage">Usage</a> •
  <a href="#project-structure">Structure</a>
</p>

---

## Abstract

Diabetic Retinopathy (DR) remains a leading cause of preventable blindness, yet a significant gap persists between automated diagnostic accuracy and clinical interpretability for patients. **DRBot** is an end-to-end multimodal framework that bridges high-precision computer vision with patient-centric communication.

The system utilises a fine-tuned **Swin Transformer** for five-stage DR classification, leveraging shifted-window self-attention to capture subtle retinal biomarkers. This vision module achieves an **accuracy of 83%** and a **Quadratic Weighted Kappa of 0.75**, outperforming ResNet and ViT baselines.

To address interpretability, a **decoupled dual-LLM architecture** is implemented, coordinated by a **DistilBART-MNLI** semantic router. The conversational engine employs a **Hybrid Retrieval-Augmented Generation (RAG)** pipeline — combining FAISS dense retrieval and BM25 sparse search — to ground responses in peer-reviewed medical literature. This framework dynamically routes queries between **BioGPT** for specialised medical reasoning and **LLaMA 3.2** for empathetic natural language synthesis.

System performance was validated using an **LLM-as-a-Judge** protocol (*N* = 30), achieving a clinical accuracy of **4.60 / 5.0** and an empathy score of **4.07 / 5.0**.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         Gradio 5 UI                             │
│  ┌──────────────────┐           ┌──────────────────────────┐    │
│  │  Fundus Image    │           │   Chat Interface         │    │
│  │  (upload)        │           │   (multi-turn)           │    │
│  └────────┬─────────┘           └────────────┬─────────────┘    │
└───────────┼──────────────────────────────────┼──────────────────┘
            │                                  │
   ┌────────▼──────────────────────────────────▼──────────┐
   │              handle_submission()                      │
   │                                                       │
   │  Image? ──► Swin Transformer (CLAHE + fine-tuned)    │
   │                       └──► DR Stage (0–4)            │
   │                                                       │
   │  Text?  ──► DistilBART-MNLI (intent router)          │
   │               │                                       │
   │    ┌──────────┴──────────────┐                       │
   │    ▼ medical question        ▼ general query          │
   │                                                       │
   │  Hybrid RAG                 LLaMA 3.2                 │
   │  ├── FAISS (BioLORD-2023)   (general mode)            │
   │  ├── BM25 sparse search                               │
   │  ├── RRF fusion                                       │
   │  └── CrossEncoder rerank                              │
   │              └──► LLaMA 3.2 (RAG-grounded)           │
   └───────────────────────────────────────────────────────┘
```

### Key Components

| Component | Model / Library | Role |
|---|---|---|
| **Retinal Classifier** | Swin Transformer (`swin_base_patch4_window7_224`) | 5-class DR grading |
| **Image Preprocessing** | CLAHE (OpenCV, LAB colour space) | Fundus contrast enhancement |
| **Dense Retrieval** | FAISS + BioLORD-2023 | Semantic chunk search |
| **Sparse Retrieval** | BM25 (rank-bm25) | Keyword chunk search |
| **Fusion** | Reciprocal Rank Fusion (RRF) | Combine dense & sparse |
| **Re-ranking** | CrossEncoder (ms-marco-MiniLM-L-6-v2) | Precision boosting |
| **Intent Router** | DistilBART-MNLI | Medical vs. general routing |
| **Conversational LLM** | LLaMA 3.2 3B Instruct (4-bit) | Response generation |
| **UI** | Gradio 5 | Web interface |

---

## Results

### Vision Module (Swin Transformer on APTOS 2019)

| Metric | DRBot (Swin) | ResNet-50 | ViT-B/16 |
|---|---|---|---|
| Accuracy | **83%** | 76% | 79% |
| Quadratic Weighted Kappa | **0.75** | 0.68 | 0.71 |

### Conversational Module (LLM-as-a-Judge, N=30)

| Dimension | Score (/ 5.0) |
|---|---|
| Clinical Accuracy | **4.60** |
| Empathy | **4.07** |
| Relevance | 4.43 |
| Safety | 4.80 |

<p align="center">
  <img src="assets/confusion_matrix.png" alt="Confusion Matrix" width="500"/>
</p>

---

## Setup

### Prerequisites

- Python 3.10+
- CUDA 12.1+ (recommended; CPU fallback available with reduced performance)
- HuggingFace account with access to `meta-llama/Llama-3.2-3B-Instruct`

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/Mahadev-2006/Dr_Bot.git
cd Dr_Bot

# 2. Create a virtual environment
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set your HuggingFace token
export HF_TOKEN="hf_your_token_here"
```

### Data Preparation (APTOS 2019)

Download the [APTOS 2019 dataset](https://www.kaggle.com/c/aptos2019-blindness-detection) and place it as:

```
data/
├── train_images/   *.png
├── val_images/     *.png
├── test_images/    *.png
├── train.csv
├── valid.csv
└── test.csv
```

### Build the RAG Knowledge Base

```bash
# Place your medical PDFs/texts in data/raw_docs/
python scripts/build_index.py   # generates data/dr_faiss.index & data/dr_chunks.json
```

---

## Usage

### Train the Swin Transformer

```bash
python src/train_swin.py
# Best model saved to models/swin_aptos_best.pth
```

### Launch DRBot

```bash
python app.py
# Opens at http://localhost:7860 (public share link printed to console)
```

### Quick retrieval test

```python
from src.retriever import hybrid_retrieve, rerank

candidates = hybrid_retrieve("What causes hard exudates in diabetic retinopathy?", top_k=15)
top3 = rerank("What causes hard exudates?", candidates, top_k=3)
for chunk, meta, score in top3:
    print(chunk[:300])
```

---

## Project Structure

```
Dr_Bot/
├── app.py                  # Gradio UI + inference pipeline
├── requirements.txt
├── README.md
│
├── src/
│   ├── retriever.py        # Hybrid RAG (FAISS + BM25 + RRF + CrossEncoder)
│   └── train_swin.py       # Swin Transformer training pipeline
│
├── data/                   # Dataset & index files (not tracked in Git)
│   ├── train_images/
│   ├── val_images/
│   ├── test_images/
│   ├── dr_faiss.index
│   └── dr_chunks.json
│
├── models/                 # Saved model weights (not tracked in Git)
│   └── swin_aptos_best.pth
│
├── notebooks/              # Exploratory notebooks
│
└── assets/                 # Figures & plots
    └── confusion_matrix.png
```

---

## Citation

If you use DRBot in your research, please cite:

```bibtex
@misc{drbot2025,
  author    = {Mahadev},
  title     = {DRBot: A Multimodal Framework for Diabetic Retinopathy Diagnosis and Patient Communication},
  year      = {2025},
  publisher = {GitHub},
  url       = {https://github.com/Mahadev-2006/Dr_Bot}
}
```

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

> ⚠️ **Medical Disclaimer**: DRBot is a research prototype and is **not** a certified medical device. It should not be used as a substitute for professional ophthalmological examination and diagnosis.
