import os
from pathlib import Path
from datasets import load_dataset

def prepare_wikitext2():
    """Prepare WikiText-2 dataset for training."""
    print("Loading WikiText-2 dataset...")
    
    # Load the dataset
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
    
    # Create data directory
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    # Save the splits
    print("Saving dataset splits...")
    with open(data_dir / "train.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(dataset["train"]["text"]))
    
    with open(data_dir / "val.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(dataset["validation"]["text"]))
    
    with open(data_dir / "test.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(dataset["test"]["text"]))
    
    print("Dataset prepared successfully!")
    print(f"Train samples: {len(dataset['train'])}")
    print(f"Validation samples: {len(dataset['validation'])}")
    print(f"Test samples: {len(dataset['test'])}")

if __name__ == "__main__":
    prepare_wikitext2() 