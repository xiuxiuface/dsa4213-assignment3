# DSA4213 Assignment 3 — Fine-Tuning Pretrained Transformers

This repository contains the implementation and results for **Assignment 3: Fine-Tuning Pretrained Transformers**.

**Student:** Tan Yun Xiu  
**Model:** DistilBERT  
**Dataset:** IMDb Sentiment Classification  
**Institution:** National University of Singapore  

---

## 📘 Overview
This project explores how large pretrained Transformer models can be adapted to a downstream NLP task — sentiment analysis — using two fine-tuning strategies:

1. **Full Fine-tuning** — update all model parameters.  
2. **LoRA Fine-tuning** — parameter-efficient fine-tuning using the PEFT library.

The task compares both approaches in terms of performance (Accuracy, F1-score) and parameter efficiency.

---

## ⚙️ Environment Setup

### Option 1: Kaggle / Google Colab
You can open the `.ipynb` file directly in Kaggle or Colab and run all cells.

### Option 2: Local / Command Line
If you prefer running locally:
```bash
pip install torch transformers datasets peft scikit-learn matplotlib accelerate
