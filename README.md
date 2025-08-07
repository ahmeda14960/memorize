# GPU Memorization Evaluation

A GPU-compatible version of Levanter's memorization evaluation tool, adapted to use HuggingFace Transformers instead of JAX/Haliax for inference while preserving all the statistical analysis and visualization capabilities.

## Overview

This tool performs **memorization analysis** on large language models by:

1. **Sliding window evaluation**: Creates overlapping 100-token windows through a text (e.g., "The Great Gatsby")
2. **Suffix probability scoring**: For each window, uses first 50 tokens as prompt and evaluates the probability the model assigns to the next 50 tokens
3. **Statistical analysis**: Computes (n,p) extraction statistics showing how many attempts would be needed to extract text with various confidence levels
4. **Visualization**: Creates character-level heatmaps and probability distribution histograms

This is essential for understanding training data leakage and potential copyright implications.

## Features

- ✅ **Multi-GPU support** via HuggingFace's automatic device mapping
- ✅ **Mixed precision** inference (bfloat16, float16, float32)
- ✅ **Memory efficient** batched processing 
- ✅ **Compatible with any HuggingFace causal LM** (Llama, GPT, etc.)
- ✅ **Identical analysis** to original Levanter version
- ✅ **Progress tracking** and debugging options

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Optional: For better performance
pip install flash-attn>=2.0.0  # Requires compatible GPU
pip install bitsandbytes>=0.41.0  # For quantization
```

## Quick Start

### 1. Basic Usage

```bash
python eval_pz.py \
    --config_path config/gpu_eval_70b.yaml \
    --torch_dtype bfloat16
```

### 2. Custom Model/Settings

```bash
python eval_pz.py \
    --config_path config/gpu_eval_70b.yaml \
    --model_name "meta-llama/Llama-3.1-70B-Instruct" \
    --eval_batch_size 4 \
    --torch_dtype float16 \
    --max_examples 1000  # Debug with fewer examples
```

### 3. Multi-GPU Setup

The tool automatically uses all available GPUs with `device_map="auto"`. For custom GPU allocation:

```bash
# Use specific GPU distribution
CUDA_VISIBLE_DEVICES=0,1,2,3 python eval_pz.py \
    --config_path config/gpu_eval_70b.yaml \
    --device_map "balanced"
```

## Configuration

The config file (`gpu_eval_70b.yaml`) controls all evaluation parameters:

```yaml
# Text processing
book_title: "gatsby"
txt_path: "src/levanter/data/books/gatsby.txt"
prompt_tokens: 50          # Prompt length
cursor_inc_chars: 10       # Sliding window stride
chunk_size: 100           # Total tokens per window
slice_length: 2000        # Text slice size for tokenization

# Model
tokenizer_name: meta-llama/Llama-3.1-8B
initialize_from_hf: "meta-llama/Llama-3.1-70B"

# Evaluation
eval_batch_size: 8        # Reduce for larger models
pz_threshold: 0.0001      # Minimum probability for histogram

# Debug
debug: false              # Enable verbose token printing
# max_examples: 100       # Limit evaluation to N examples

# Output
plot_path: "char_heatmap.png"
histogram_path: "pz_histogram.png"
```

## Memory Requirements

**Recommended GPU memory for different model sizes:**

| Model Size | Precision | Min GPU Memory | Recommended |
|------------|-----------|----------------|-------------|
| 7B         | bfloat16  | 16GB          | 24GB        |
| 13B        | bfloat16  | 24GB          | 32GB        |
| 70B        | bfloat16  | 80GB          | 4x24GB      |
| 70B        | int8      | 40GB          | 2x24GB      |

**Tips for large models:**
- Use `--torch_dtype bfloat16` or `float16` for 2x memory savings
- Reduce `--eval_batch_size` if you get OOM errors
- Consider quantization with bitsandbytes for further memory reduction

## Output Files

The evaluation produces several outputs:

### 1. Character-level Heatmap (`char_heatmap.png`)
- Shows maximum suffix probability for each character in the text
- Darker blue = higher memorization
- Reveals which parts of the text the model has memorized most strongly

### 2. Probability Distribution (`pz_histogram.png`) 
- Histogram of all suffix probabilities above threshold
- Log-scale y-axis shows distribution shape
- Helps identify memorization patterns

### 3. Raw Data (`char_heatmap.npy`)
- NumPy array with per-character max probabilities
- Can be loaded for further analysis: `np.load('char_heatmap.npy')`

### 4. Console Output
- Extraction rate statistics: (n,p) values showing recoverability
- Character analysis: mean/median/max probabilities, high-confidence regions
- Progress tracking during evaluation

## Advanced Usage

### Custom Text Analysis

Replace the text file to analyze memorization of different content:

```yaml
txt_path: "path/to/your/text.txt"
book_title: "Custom Text"
```

### Quantized Inference

For memory-constrained setups, add quantization arguments:

```python
# In gpu_eval_careless.py, modify model loading:
from transformers import BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(
    load_in_8bit=True,  # or load_in_4bit=True
)

model = AutoModelForCausalLM.from_pretrained(
    cfg.model_name,
    quantization_config=quantization_config,
    device_map=cfg.device_map,
    trust_remote_code=True
)
```

### Debugging

For development and testing:

```bash
# Quick test with limited examples
python eval_pz.py \
    --config_path config/llama3.1_70b_gatsby.yaml \
    --eval_batch_size 1 \
    --max_examples 100

# Enable debug output (verbose token printing)
python eval_pz.py \
    --config_path config/llama3.1_70b_gatsby.yaml \
    --debug
```

## Comparison with Original

This GPU version maintains identical functionality to the original Levanter implementation:

| Feature | Original (JAX) | GPU Version (PyTorch) |
|---------|----------------|----------------------|
| Text processing | ✅ `levanter.books.util` | ✅ Same functions |
| Statistical analysis | ✅ NumPy-based | ✅ Identical |
| Visualizations | ✅ Matplotlib | ✅ Identical |
| Model inference | JAX/Haliax | HuggingFace Transformers |
| Parallelism | TPU sharding | Multi-GPU device mapping |
| Memory efficiency | JAX optimizations | Mixed precision + batching |

## Troubleshooting

### Common Issues

**OOM (Out of Memory)**
```bash
# Reduce batch size
--eval_batch_size 2

# Use lower precision
--torch_dtype float16

# Use quantization (modify code as shown above)
```

# NLPRUN
```
 nlprun --job-name llama3.1_70b --machine sphinx8 -w /nlp/u/ahmedah/code/memorize/ -a memorize -g 4 -c 16 -r 200G 'python eval_pz.py --config_path config/llama3.1_70b_gatsby.yaml  --eval_batch_size 1 > logs.txt 2>&1'
```


