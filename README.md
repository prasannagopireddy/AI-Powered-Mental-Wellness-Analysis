# 🧠 AI-Powered-Mental-Wellness-Analysis

> Automatically classifying mental health conditions from Reddit posts using a fine-tuned DistilBERT transformer model.

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0-orange?logo=pytorch)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-yellow?logo=huggingface)
![Accuracy](https://img.shields.io/badge/Test%20Accuracy-82.89%25-brightgreen)
![License](https://img.shields.io/badge/License-MIT-lightgrey)
---

## 📌 Project Overview

Mental health disorders affect hundreds of millions of people globally, yet many go undetected due to stigma around seeking help. Social media platforms like Reddit have become spaces where people openly express their struggles — often using language patterns that are characteristic of specific mental health conditions.

This project fine-tunes **DistilBERT** (`distilbert-base-uncased`) on Reddit posts to automatically classify text into one of **five mental health categories**. The goal is to demonstrate that transformer-based NLP can support early detection and awareness of mental health conditions from user-generated text.

---

## 🎯 Conditions Classified

| Label | Condition |
|---|---|
| 0 | Stress |
| 1 | Depression |
| 2 | Bipolar Disorder |
| 3 | Personality Disorder |
| 4 | Anxiety |

---

## 📊 Results

Evaluated on a held-out 20% test set (1,122 samples):

| Metric | Score |
|---|---|
| **Accuracy** | **82.89%** |
| Weighted Precision | 83% |
| Weighted Recall | 83% |
| Weighted F1 Score | 83% |

### Per-Class Performance

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Stress | 0.89 | 0.91 | 0.90 | 220 |
| Depression | 0.73 | 0.78 | 0.75 | 241 |
| Bipolar Disorder | 0.90 | 0.83 | 0.87 | 217 |
| Personality Disorder | 0.80 | 0.81 | 0.80 | 215 |
| Anxiety | 0.85 | 0.82 | 0.83 | 229 |

> **Note:** Depression has the lowest F1 (0.75), which is expected — depressive language overlaps significantly with anxiety and stress. This is a known challenge in mental health NLP.

---

## 🗂️ Repository Structure

```
AI-Powered-Mental-Wellness-Analysis/
│
├── data/
│   └── README.md                    # How to download dataset from Kaggle
│
├── mental_health_code.ipynb         # Full project notebook (EDA + Training + Results)
│
├── preprocess.py                    # Data cleaning, label assignment, train/test split
├── train.py                         # DistilBERT fine-tuning with early stopping
├── evaluate.py                      # Evaluation metrics + confusion matrix
├── predict.py                       # Single text prediction with confidence score
│
├── requirements.txt                 # All Python dependencies
├── .gitignore                       # Excludes dataset, model weights, cache
└── README.md                        # Project description with badges and results                    # You are here
```

---

## ⚙️ Pipeline

```
Raw Reddit CSV
      │
      ▼
 preprocess.py        → cleans data, encodes labels, saves train/test/label_map
      │
      ▼
   train.py           → fine-tunes DistilBERT, saves best model to final_model/
      │
      ▼
  evaluate.py         → loads final_model/, reports accuracy / F1 / confusion matrix
      │
      ▼
  predict.py          → loads final_model/, classifies any custom text input
```

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/prasannagopireddy/AI-Powered-Mental-Wellness-Analysis.git
cd mental-health-disorder-detection
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Add the dataset

Download the **Reddit Mental Health Dataset** from Kaggle and place it as:

```
Reddit Mental Health Data/Reddit Mental Health Dataset.csv
```

### 4. Run the pipeline

```bash
# Step 1 — Preprocess
python preprocess.py

# Step 2 — Train
python train.py

# Step 3 — Evaluate
python evaluate.py

# Step 4 — Predict on custom text
python predict.py
```

> **GPU recommended.** Training takes ~30–35 minutes on a Google Colab T4 GPU.  
> To run on Colab, open `mental_health_codex.ipynb` and follow the cells top to bottom.

---

## 🏗️ Model Architecture & Training Details

| Parameter | Value |
|---|---|
| Base model | `distilbert-base-uncased` |
| Task | Sequence classification (5 classes) |
| Max token length | 512 |
| Learning rate | 2e-5 |
| Batch size | 16 |
| Epochs | 4 (early stopping, patience = 2) |
| Weight decay | 0.01 |
| Mixed precision | fp16 (auto on GPU) |
| Best model selection | Highest validation accuracy |

DistilBERT is 40% smaller and 60% faster than BERT while retaining 97% of its language understanding capability — making it ideal for fine-tuning on limited GPU resources.

---

## 📁 Dataset

- **Source:** Reddit Mental Health Dataset (Kaggle)
- **Total samples:** 5,607 posts
- **Split:** 80% train (4,485) / 20% test (1,122)
- **Split strategy:** Stratified — class balance preserved in both splits
- **Input:** Post title + post body combined into a single text field

### Class Distribution

| Condition | Samples |
|---|---|
| Depression | 1,202 |
| Anxiety | 1,144 |
| Stress | 1,099 |
| Bipolar Disorder | 1,085 |
| Personality Disorder | 1,077 |

The dataset is well-balanced across all five classes, which avoids bias toward any single condition during training.

---

## 🔍 Sample Prediction Output

```
Enter the text to analyse: I feel overwhelmed, anxious, and constantly worried about everything.

Prediction Result
-----------------
Predicted class number : 4
Predicted class label  : Anxiety
Confidence             : 87.43%

All class probabilities:
  Stress                    : 4.21%
  Depression                : 5.18%
  Bipolar Disorder          : 1.63%
  Personality Disorder      : 1.55%
  Anxiety                   : 87.43%
```

---

## ⚠️ Limitations

- **Dataset size:** 5,607 samples is relatively small for transformer fine-tuning. Larger datasets would improve generalisation.
- **Text truncation:** Posts longer than 512 tokens are truncated — very long posts may lose some context.
- **Domain specificity:** Trained only on Reddit data; may not generalise to clinical notes or other platforms.
- **Depression overlap:** Depressive language shares features with anxiety and stress, leading to the lowest per-class F1 (0.75).

---

## 🔭 Future Work

- Fine-tune on larger, multi-source datasets (e.g., Twitter, clinical transcripts)
- Experiment with **Mental-BERT** — a BERT model pre-trained on mental health corpora
- Add explainability using **LIME** or **SHAP** to highlight which words drove each prediction
- Build a web interface using **Flask** or **Streamlit** for a live demo

---

## 🛠️ Tech Stack

- **Language:** Python 3.10
- **Deep Learning:** PyTorch
- **NLP / Transformers:** Hugging Face Transformers, DistilBERT
- **Data Processing:** Pandas, NumPy, scikit-learn
- **Training Platform:** Google Colab (T4 GPU)

---

## 👤 Author

**Prasanna Gopireddy**  
B.Tech — Artificial Intelligence & Data Science  
SRKR Engineering College  

[![GitHub](https://img.shields.io/badge/GitHub-prasannagopireddy-black?logo=github)](https://github.com/prasannagopireddy/)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-prasannagopireddy-blue?logo=linkedin)](https://linkedin.com/in/prasannagopireddy/)


---

 
## 📄 License
 
This project is licensed under the MIT License.
 
---

## 📚 References

- Sanh, V. et al. (2019). *DistilBERT, a distilled version of BERT.* [arXiv:1910.01108](https://arxiv.org/abs/1910.01108)
- Devlin, J. et al. (2018). *BERT: Pre-training of Deep Bidirectional Transformers.* [arXiv:1810.04805](https://arxiv.org/abs/1810.04805)
- Hugging Face Transformers: https://huggingface.co/docs/transformers
- Reddit Mental Health Dataset: Kaggle
