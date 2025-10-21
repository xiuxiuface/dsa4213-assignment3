"""
Before running:
    pip install -r requirements.txt
or
    pip install numpy==1.26.4 torch==2.3.0 transformers==4.40.2 datasets==2.20.0 peft==0.10.0 scikit-learn matplotlib accelerate
    """
import torch
import numpy as np
import os
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

os.environ["WANDB_DISABLED"] = "true"

from datasets import load_dataset
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, DataCollatorWithPadding
)
# ==== Prompted by GPT to bypass chat template bug ====
import transformers.utils.hub as hub_utils
hub_utils.list_repo_templates = lambda *args, **kwargs: []

from peft import get_peft_model, LoraConfig, TaskType
from sklearn.metrics import accuracy_score, f1_score
import matplotlib.pyplot as plt

CONFIG = {
    "model_name": "distilbert-base-uncased",
    "dataset_name": "imdb",
    "max_length": 512,
    "batch_size": 16,
    "learning_rate": 2e-5,
    "num_epochs": 3,
    "seed": 42,
    "output_dir": "/kaggle/working/results",
    "lora_r": 8,
    "lora_alpha": 32,
    "lora_dropout": 0.1,
}

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def load_and_prepare_data(config):
    print("Loading IMDb dataset...")
    dataset = load_dataset(config["dataset_name"])
    

    tokenizer = AutoTokenizer.from_pretrained(
        config["model_name"],
        use_fast=True,
        trust_remote_code=False
    )
    
    def tokenize_fn(examples):
        return tokenizer(
            examples["text"],
            max_length=config["max_length"],
            truncation=True,
            padding=True
        )
    
    print("Tokenizing dataset...")
    tokenized = dataset.map(tokenize_fn, batched=True, remove_columns=["text"])
    tokenized = tokenized.rename_column("label", "labels")
    
    return tokenized, tokenizer

def setup_lora_model(config):
    model = AutoModelForSequenceClassification.from_pretrained(
        config["model_name"],
        num_labels=2
    )
    
    # Print all module names to debug
    print("\nFinding Linear layers in model...")
    linear_layers = []
    for name, module in model.named_modules():
        if 'Linear' in str(type(module)):
            linear_layers.append(name)
            if len(linear_layers) <= 10: 
                print(f"  {name}")

    target_modules = ["q_lin", "k_lin", "v_lin"]
    
 
    
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=config["lora_r"],
        lora_alpha=config["lora_alpha"],
        lora_dropout=config["lora_dropout"],
        target_modules=target_modules,
        bias="none",
    )
    
    print(f"\nApplying LoRA with target_modules: {target_modules}")
    try:
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
    except ValueError as e:
        print(f"\n Error: {e}")
        print("\nAvailable linear layers found:")
        for layer in linear_layers[:20]:
            print(f"  - {layer}")
        raise
    
    return model

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    accuracy = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions)
    return {"accuracy": accuracy, "f1": f1}

def train_model(model, tokenized_data, config):
    training_args = TrainingArguments(
        output_dir=f"{config['output_dir']}/lora_finetuning",
        overwrite_output_dir=True,
        num_train_epochs=config["num_epochs"],
        per_device_train_batch_size=config["batch_size"],
        per_device_eval_batch_size=config["batch_size"],
        learning_rate=config["learning_rate"],
        logging_steps=100,
        evaluation_strategy="epoch",  

        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        seed=config["seed"],
        report_to="none",  
    )
    
    tokenizer = AutoTokenizer.from_pretrained(
        config["model_name"],
        use_fast=True,
        trust_remote_code=False
    )
    data_collator = DataCollatorWithPadding(tokenizer)
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_data["train"],
        eval_dataset=tokenized_data["test"],
        compute_metrics=compute_metrics,
        data_collator=data_collator,
    )
    
    print("\nTraining LoRA Fine-tuning...")
    trainer.train()
    return trainer

def evaluate_and_save(trainer, tokenized_data, config):
    print("\nEvaluating LoRA Fine-tuning...")
    eval_results = trainer.evaluate()
    
    predictions = trainer.predict(tokenized_data["test"])
    pred_labels = np.argmax(predictions.predictions, axis=1)
    true_labels = tokenized_data["test"]["labels"]
    
    results = {
        "strategy": "LoRA Fine-tuning",
        "accuracy": eval_results.get("eval_accuracy", 0),
        "f1": eval_results.get("eval_f1", 0),
        "model_size": sum(p.numel() for p in trainer.model.parameters()),
        "trainable_params": sum(p.numel() for p in trainer.model.parameters() if p.requires_grad),
        "lora_config": {
            "r": config["lora_r"],
            "alpha": config["lora_alpha"],
            "dropout": config["lora_dropout"],
        }
    }
    
    print(f"\nLoRA Fine-tuning Results:")
    print(f"  Accuracy: {results['accuracy']:.4f}")
    print(f"  F1-score: {results['f1']:.4f}")
    print(f"  Total params: {results['model_size']:,}")
    print(f"  Trainable params: {results['trainable_params']:,}")
    print(f"  LoRA rank: {config['lora_r']}")
    
    os.makedirs(config["output_dir"], exist_ok=True)
    results_summary = {
        "timestamp": datetime.now().isoformat(),
        "config": config,
        "results": results,
    }
    
    with open(f"{config['output_dir']}/lora_results.json", "w") as f:
        json.dump(results_summary, f, indent=2)
    print(f"\n✓ Results saved to: {config['output_dir']}/lora_results.json")
    
    return results, pred_labels, true_labels

def main():
    set_seed(CONFIG["seed"])
    os.makedirs(CONFIG["output_dir"], exist_ok=True)
    
    print("="*60)
    print("STRATEGY 2: LORA FINE-TUNING")
    print("="*60)

    # Prepare and train the model

    tokenized_data, tokenizer = load_and_prepare_data(CONFIG)
    
    model = setup_lora_model(CONFIG)
    trainer = train_model(model, tokenized_data, CONFIG)
    results, pred_labels, true_labels = evaluate_and_save(trainer, tokenized_data, CONFIG)
    
    print("\n" + "="*60)
    print("LORA FINE-TUNING COMPLETE!")
    print("="*60)

if __name__ == "__main__":
    main()