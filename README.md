# DSA4213 Assignment 3: Fine-Tuning Pretrained Transformers

This repository contains the implementation and results for **Assignment 3: Fine-Tuning Pretrained Transformers**.

**Student:** Tan Yun Xiu  
**Model:** DistilBERT  
**Dataset:** IMDb Sentiment Classification  
**Institution:** National University of Singapore  

---

## Overview
This project explores how large pretrained Transformer models can be adapted to a downstream NLP task — sentiment analysis — using two fine-tuning strategies:

1. **Full Fine-tuning** — update all model parameters.  
2. **LoRA Fine-tuning** — parameter-efficient fine-tuning using the PEFT library.

The task compares both approaches in terms of performance (Accuracy, F1-score) and parameter efficiency.

---

## Environment Setup

### Option 1: Kaggle / Google Colab
You can open the `.ipynb` file directly in Kaggle or Colab and run all cells (I use Kaggle).

### Option 2: Local / Command Line
## 🔧 Setup
Clone the repository and install the dependencies:

```bash
git clone https://github.com/<your-username>/dsa4213-a3.git
cd dsa4213-a3
pip install -r requirements.txt
```

## Run
Run one of the following:
```bash
python main.py --strategy full
python main.py --strategy lora
```


