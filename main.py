# main.py
import argparse
from train_full import main as run_full
from train_lora import main as run_lora

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", type=str, required=True,
                        choices=["full", "lora"],
                        help="Choose fine-tuning strategy: full or lora")
    args = parser.parse_args()

    if args.strategy == "full":
        print("Running FULL Fine-tuning...")
        run_full()
    elif args.strategy == "lora":
        print("Running LORA Fine-tuning...")
        run_lora()
