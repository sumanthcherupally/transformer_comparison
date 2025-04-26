import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Optional
import numpy as np

class TextDataset(Dataset):
    def __init__(
        self,
        texts: List[str],
        tokenizer,
        max_length: int = 512,
        stride: int = 256
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.stride = stride
        self.encoded = self.tokenizer(
            texts,
            max_length=max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        
    def __len__(self) -> int:
        return len(self.encoded["input_ids"])
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        input_ids = self.encoded["input_ids"][idx]
        attention_mask = self.encoded["attention_mask"][idx]
        
        # Shift right for language modeling (predict next token)
        target = input_ids[1:].clone()
        input_ids = input_ids[:-1]
        attention_mask = attention_mask[:-1]
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": target
        }

def create_dataloaders(
    train_texts: List[str],
    val_texts: List[str],
    tokenizer,
    batch_size: int = 32,
    max_length: int = 512,
    stride: int = 256,
    num_workers: int = 4
) -> Tuple[DataLoader, DataLoader]:
    
    train_dataset = TextDataset(
        train_texts,
        tokenizer,
        max_length=max_length,
        stride=stride
    )
    
    val_dataset = TextDataset(
        val_texts,
        tokenizer,
        max_length=max_length,
        stride=stride
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    
    return train_loader, val_loader 