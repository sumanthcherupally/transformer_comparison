import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Any
import numpy as np
from transformers import PreTrainedTokenizer
from transformers import AutoTokenizer

class TextDataset(Dataset):
    def __init__(
        self,
        file_path: str,
        tokenizer: PreTrainedTokenizer,
        max_seq_len: int
    ):
        self.file_path = file_path
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        
        # Load and tokenize data
        with open(file_path, 'r', encoding='utf-8') as f:
            self.text = f.read()
        
        # Tokenize the entire text
        self.encoded = tokenizer(
            self.text,
            truncation=False,
            padding=False,
            return_tensors=None
        )['input_ids']
        
        # Create chunks of max_seq_len
        self.chunks = []
        for i in range(0, len(self.encoded) - max_seq_len, max_seq_len):
            chunk = self.encoded[i:i + max_seq_len]
            if len(chunk) == max_seq_len:
                self.chunks.append(chunk)
    
    def __len__(self) -> int:
        return len(self.chunks)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        chunk = self.chunks[idx]
        
        # Create input and target sequences
        input_ids = chunk[:-1]
        labels = chunk[1:]
        
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'labels': torch.tensor(labels, dtype=torch.long)
        }

def get_dataloader(
    file_path: str,
    batch_size: int,
    max_seq_len: int,
    tokenizer_name: str = "gpt2"
) -> DataLoader:
    """
    Create a DataLoader for the text dataset.
    
    Args:
        file_path: Path to the text file
        batch_size: Batch size for the DataLoader
        max_seq_len: Maximum sequence length
        tokenizer_name: Name of the tokenizer to use
        
    Returns:
        DataLoader instance
    """
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    
    # Create dataset
    dataset = TextDataset(file_path, tokenizer, max_seq_len)
    
    # Create DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    return dataloader 