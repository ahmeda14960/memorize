#!/usr/bin/env python3
"""
GPU-based memorization evaluation using HuggingFace Transformers.
Adapted from levanter's eval_careless_lm.py for GPU inference.
"""

import argparse
import logging
import math
import pathlib
from dataclasses import dataclass
from typing import List, Optional, Tuple
import yaml

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import matplotlib.pyplot as plt

from utils import (
    chunk_text_to_sliding_window_token_chunks,
    compute_max_extraction_rates, 
    create_pz_histogram
)

logger = logging.getLogger(__name__)


@dataclass
class GPUEvalConfig:
    """Configuration for GPU-based memorization evaluation."""
    
    # Input text
    book_title: str = "gatsby"
    txt_path: str = "src/levanter/data/books/gatsby.txt"
    prompt_tokens: int = 50
    cursor_inc_chars: int = 10
    slice_length: int = 2000
    chunk_size: int = 100
    
    # Model and tokenizer
    model_name: str = "meta-llama/Llama-3.1-70B"
    tokenizer_name: str = "meta-llama/Llama-3.1-8B"
    
    # Inference settings
    eval_batch_size: int = 8
    torch_dtype: str = "bfloat16"  # or "float16", "float32"
    device_map: str = "auto"  # or "balanced", "sequential"
    max_memory: Optional[dict] = None
    
    # Output
    plot_path: str = "bar_plot_char_max_pz_70b_gpu.png"
    histogram_path: str = "pz_distribution_histogram_gpu.png"
    pz_threshold: float = 0.0001
    
    # Debug
    max_examples: Optional[int] = None


def load_config(config_path: str) -> GPUEvalConfig:
    """Load config from YAML file and convert to GPUEvalConfig."""
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    # Extract relevant fields, ignoring levanter-specific ones
    gpu_config = GPUEvalConfig()
    
    # Text processing params
    if 'book_title' in config_dict:
        gpu_config.book_title = config_dict['book_title']
    if 'txt_path' in config_dict:
        gpu_config.txt_path = config_dict['txt_path']
    if 'prompt_tokens' in config_dict:
        gpu_config.prompt_tokens = config_dict['prompt_tokens']
    if 'cursor_inc_chars' in config_dict:
        gpu_config.cursor_inc_chars = config_dict['cursor_inc_chars']
    if 'slice_length' in config_dict:
        gpu_config.slice_length = config_dict['slice_length']
    if 'chunk_size' in config_dict:
        gpu_config.chunk_size = config_dict['chunk_size']
    
    # Model/tokenizer
    if 'tokenizer_name' in config_dict:
        gpu_config.tokenizer_name = config_dict['tokenizer_name']
    if 'initialize_from_hf' in config_dict:
        gpu_config.model_name = config_dict['initialize_from_hf']
    
    # Batch size
    if 'eval_batch_size' in config_dict:
        gpu_config.eval_batch_size = config_dict['eval_batch_size']
    
    # Debug
    if 'max_examples' in config_dict:
        gpu_config.max_examples = config_dict['max_examples']
        
    return gpu_config


def compute_sequence_log_probs(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer, 
    input_ids: torch.Tensor,
    prompt_length: int,
    pad_token_id: int
) -> torch.Tensor:
    """
    Compute log probabilities for the suffix part of sequences.
    
    Args:
        model: HuggingFace causal LM
        tokenizer: HuggingFace tokenizer  
        input_ids: Tensor of shape (batch_size, seq_len)
        prompt_length: Number of prompt tokens (suffix starts after this)
        pad_token_id: Padding token ID to ignore in loss
        
    Returns:
        Tensor of shape (batch_size,) with sequence probabilities
    """
    batch_size, seq_len = input_ids.shape
    
    with torch.no_grad():
        # Get logits
        outputs = model(input_ids)
        logits = outputs.logits  # (batch_size, seq_len, vocab_size)
        
        # Compute log probabilities
        log_probs = F.log_softmax(logits, dim=-1)
        
        # Get target tokens (shifted by 1)
        targets = input_ids[:, 1:].contiguous()  # (batch_size, seq_len-1)
        log_probs = log_probs[:, :-1].contiguous()  # (batch_size, seq_len-1, vocab_size)
        
        # Gather log probs for target tokens
        target_log_probs = torch.gather(
            log_probs, 
            dim=-1, 
            index=targets.unsqueeze(-1)
        ).squeeze(-1)  # (batch_size, seq_len-1)
        
        # Create mask for suffix tokens (ignore prompt and padding)
        mask = torch.ones_like(targets, dtype=torch.float32)
        mask[:, :prompt_length-1] = 0  # Ignore prompt tokens
        mask[targets == pad_token_id] = 0  # Ignore padding tokens
        
        # Sum log probs for suffix only
        suffix_log_probs = torch.sum(target_log_probs * mask, dim=1)
        
        # Convert to probabilities
        suffix_probs = torch.exp(suffix_log_probs)
        
    return suffix_probs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--model_name", type=str, help="Override model name")
    parser.add_argument("--torch_dtype", type=str, default="bfloat16", 
                       choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--device_map", type=str, default="auto")
    parser.add_argument("--eval_batch_size", type=int, help="Override batch size")
    parser.add_argument("--max_examples", type=int, help="Limit number of examples")
    
    args = parser.parse_args()
    
    # Load config
    cfg = load_config(args.config_path)
    
    # Override with CLI args
    if args.model_name:
        cfg.model_name = args.model_name
    if args.torch_dtype:
        cfg.torch_dtype = args.torch_dtype
    if args.device_map:
        cfg.device_map = args.device_map
    if args.eval_batch_size:
        cfg.eval_batch_size = args.eval_batch_size
    if args.max_examples:
        cfg.max_examples = args.max_examples
    
    print(f"Loading model: {cfg.model_name}")
    print(f"Using dtype: {cfg.torch_dtype}")
    print(f"Device map: {cfg.device_map}")
    
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    pad_token_id = tokenizer.pad_token_id
    
    # Load model
    torch_dtype = getattr(torch, cfg.torch_dtype)
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name,
        torch_dtype=torch_dtype,
        device_map=cfg.device_map,
        trust_remote_code=True
    )
    model.eval()
    
    # Load and process text
    raw_text = pathlib.Path(cfg.txt_path).read_text()
    
    # Generate sliding window chunks using levanter's utility
    print("Generating sliding window chunks...")
    chunks = chunk_text_to_sliding_window_token_chunks(
        raw_text,
        tokenizer,
        chunk_size=cfg.chunk_size,
        slice_length=cfg.slice_length,
        cursor_inc=cfg.cursor_inc_chars,
    )
    
    total_chunks = len(chunks)
    print(f"Total sliding windows: {total_chunks}")
    
    if cfg.max_examples:
        chunks = chunks[:cfg.max_examples]
        total_chunks = len(chunks)
        print(f"Limited to {total_chunks} examples for debugging")
    
    # Process in batches
    pz_list: List[float] = []
    char_ranges: List[Tuple[int, int]] = []
    
    batch_size = cfg.eval_batch_size
    total_batches = math.ceil(total_chunks / batch_size)
    
    for batch_idx in tqdm(range(total_batches), desc="Processing batches"):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, total_chunks)
        batch_chunks = chunks[start_idx:end_idx]
        
        # Prepare batch
        batch_input_ids = []
        batch_char_ranges = []
        
        for chunk in batch_chunks:
            ids = chunk["input_ids"]
            # Pad to consistent length if needed
            max_len = cfg.chunk_size
            if len(ids) < max_len:
                ids = ids + [pad_token_id] * (max_len - len(ids))
            elif len(ids) > max_len:
                ids = ids[:max_len]
                
            batch_input_ids.append(ids)
            batch_char_ranges.append((chunk["start_idx"], chunk["end_idx"]))
        
        # Convert to tensor
        input_ids = torch.tensor(batch_input_ids, dtype=torch.long)
        
        # Move to appropriate device (model should handle this with device_map)
        if hasattr(model, 'device'):
            input_ids = input_ids.to(model.device)
        
        # Compute probabilities
        batch_probs = compute_sequence_log_probs(
            model, tokenizer, input_ids, cfg.prompt_tokens, pad_token_id
        )
        
        # Collect results
        pz_list.extend(batch_probs.cpu().numpy().tolist())
        char_ranges.extend(batch_char_ranges)
        
        # Progress logging
        done = min(end_idx, total_chunks)
        pct = 100 * done / total_chunks
        if batch_idx % 10 == 0:
            print(f"Batch {batch_idx+1}/{total_batches} – {done}/{total_chunks} windows ({pct:.1f}%)")
    
    print(f"\nProcessed {len(pz_list)} windows")
    
    # Compute extraction statistics using levanter's utility
    print("Computing extraction rate statistics...")
    stats = compute_max_extraction_rates(pz_list)
    print(f"First few (n,p) extraction entries: {stats[0][:5]}")
    
    # Create histogram using levanter's utility  
    print("Creating p_z histogram...")
    hist_stats = create_pz_histogram(
        pz_list=pz_list, 
        threshold=cfg.pz_threshold, 
        save_path=cfg.histogram_path, 
        book_title=cfg.book_title
    )
    if hist_stats:
        print(f"Histogram statistics: {hist_stats}")
    
    # Character-level max-P(z) analysis
    print("Computing character-level analysis...")
    text_len = len(raw_text)
    char_max = np.zeros(text_len, dtype=np.float32)
    
    for pz, (c0, c1) in zip(pz_list, char_ranges):
        char_max[c0:c1+1] = np.maximum(char_max[c0:c1+1], pz)
    
    # Print character analysis stats
    print(f"Character analysis:")
    print(f"  Mean max P(z): {np.mean(char_max):.6f}")
    print(f"  Median max P(z): {np.median(char_max):.6f}")  
    print(f"  Max P(z): {np.max(char_max):.6f}")
    print(f"  Chars above 0.5: {np.sum(char_max > 0.5)}")
    print(f"  Chars above 0.9: {np.sum(char_max > 0.9)}")
    print(f"  Total chars: {len(char_max)}")
    
    # Create character-level heatmap
    print("Creating character-level heatmap...")
    fig, ax = plt.subplots(figsize=(14, 2))
    im = ax.imshow(
        char_max[np.newaxis, :],  # shape (1, text_len)
        cmap="Blues",
        aspect="auto", 
        vmin=0.0,
        vmax=1.0,
        interpolation="nearest",
    )
    
    ax.set_title(f"{cfg.book_title}: Maximum per-character probability")
    ax.set_xlabel("Book position (character)")
    ax.set_yticks([])  # Hide y-axis (only one row)
    
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.04)
    cbar.set_label("Max. probability")
    
    plt.tight_layout()
    plt.savefig(cfg.plot_path, dpi=300)
    print(f"Character heatmap saved to: {cfg.plot_path}")
    
    # Save raw data
    npy_path = pathlib.Path(cfg.plot_path).with_suffix(".npy")
    np.save(npy_path, char_max)
    print(f"Raw character data saved to: {npy_path}")
    
    print("Evaluation complete!")


if __name__ == "__main__":
    main()