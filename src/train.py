import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import yaml
import argparse
from tqdm import tqdm
import wandb
from typing import Dict, Any

from models.flexible_transformer import FlexibleTransformer
from data_loader import get_dataloader
from metrics import compute_metrics

def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    position_embedding_type: str
) -> Dict[str, float]:
    model.train()
    total_loss = 0
    total_tokens = 0
    
    for batch in tqdm(dataloader, desc="Training"):
        input_ids = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)
        
        # Create attention mask
        mask = (input_ids != 0).unsqueeze(1).unsqueeze(2)
        
        # Forward pass
        outputs = model(input_ids, mask)
        
        # Compute loss
        loss = criterion(outputs.view(-1, outputs.size(-1)), labels.view(-1))
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Update metrics
        total_loss += loss.item() * labels.numel()
        total_tokens += (labels != 0).sum().item()
    
    return {
        'train_loss': total_loss / total_tokens,
        'train_perplexity': torch.exp(torch.tensor(total_loss / total_tokens)).item()
    }

def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    position_embedding_type: str
) -> Dict[str, float]:
    model.eval()
    total_loss = 0
    total_tokens = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            
            # Create attention mask
            mask = (input_ids != 0).unsqueeze(1).unsqueeze(2)
            
            # Forward pass
            outputs = model(input_ids, mask)
            
            # Compute loss
            loss = criterion(outputs.view(-1, outputs.size(-1)), labels.view(-1))
            
            # Update metrics
            total_loss += loss.item() * labels.numel()
            total_tokens += (labels != 0).sum().item()
    
    return {
        'val_loss': total_loss / total_tokens,
        'val_perplexity': torch.exp(torch.tensor(total_loss / total_tokens)).item()
    }

def main():
    parser = argparse.ArgumentParser(description='Train transformer with different attention mechanisms')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--attention_type', type=str, required=True, 
                      choices=['dot_product', 'linear', 'sparse', 'local', 'gqa', 'flash'],
                      help='Type of attention mechanism to use')
    parser.add_argument('--position_embedding', type=str, required=True,
                      choices=['sinusoidal', 'rope', 'alibi'],
                      help='Type of position embedding to use')
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Initialize wandb
    wandb.init(
        project="transformer-comparison",
        config={
            "attention_type": args.attention_type,
            "position_embedding": args.position_embedding,
            **config
        }
    )
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create dataloaders
    train_dataloader = get_dataloader(
        config['data']['train_path'],
        config['training']['batch_size'],
        config['model']['max_seq_len']
    )
    val_dataloader = get_dataloader(
        config['data']['val_path'],
        config['training']['batch_size'],
        config['model']['max_seq_len']
    )
    
    # Create model
    model = FlexibleTransformer(
        vocab_size=config['model']['vocab_size'],
        d_model=config['model']['d_model'],
        num_heads=config['model']['num_heads'],
        num_layers=config['model']['num_layers'],
        d_ff=config['model']['d_ff'],
        max_seq_len=config['model']['max_seq_len'],
        attention_type=args.attention_type,
        position_embedding_type=args.position_embedding,
        dropout=config['model']['dropout'],
        **config['attention_config'].get(args.attention_type, {})
    ).to(device)
    
    # Create optimizer and criterion
    optimizer = optim.Adam(
        model.parameters(),
        lr=config['training']['learning_rate'],
        betas=(0.9, 0.999),
        eps=1e-8
    )
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # 0 is padding token
    
    # Training loop
    best_val_loss = float('inf')
    for epoch in range(config['training']['num_epochs']):
        # Train
        train_metrics = train_epoch(
            model, train_dataloader, optimizer, criterion, device, args.position_embedding
        )
        
        # Evaluate
        val_metrics = evaluate(
            model, val_dataloader, criterion, device, args.position_embedding
        )
        
        # Log metrics
        metrics = {**train_metrics, **val_metrics}
        wandb.log(metrics)
        
        # Save best model
        if val_metrics['val_loss'] < best_val_loss:
            best_val_loss = val_metrics['val_loss']
            torch.save(model.state_dict(), f'checkpoints/{args.attention_type}_{args.position_embedding}_best.pt')
        
        print(f"Epoch {epoch + 1}/{config['training']['num_epochs']}")
        print(f"Train Loss: {train_metrics['train_loss']:.4f}")
        print(f"Train Perplexity: {train_metrics['train_perplexity']:.4f}")
        print(f"Val Loss: {val_metrics['val_loss']:.4f}")
        print(f"Val Perplexity: {val_metrics['val_perplexity']:.4f}")
        print("-" * 50)

if __name__ == '__main__':
    main() 