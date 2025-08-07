# Tensor Parallelism for GPU Memorization Evaluation

## Current State: Pipeline Parallelism
- Using HuggingFace `device_map="auto"` 
- Sequential layer execution across GPUs (GPU0 → GPU1 → GPU2 → GPU3)
- Pipeline bubbles cause underutilization (~25-69% per GPU)
- Memory distribution: [70GB, 71GB, 71GB, 55GB]

## Tensor Parallelism Options

### 1. Accelerate + DeepSpeed ZeRO-3
**Easiest integration with existing code**
```python
from accelerate import Accelerator
from transformers import AutoModelForCausalLM

accelerator = Accelerator()
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map=None  # Let accelerate handle it
)
model = accelerator.prepare(model)
```

**Pros**: Minimal code changes, automatic sharding
**Cons**: Primarily for training, inference optimizations limited

### 2. FasterTransformer / TensorRT-LLM
**Best performance but complex**
```bash
# Convert HF model to TensorRT format
trtllm-build --hf_model_dir openai/gpt-oss-120b \
             --output_dir ./trt_engines/gpt-oss-120b \
             --tp_size 4
```

**Pros**: True tensor parallelism, optimal GPU utilization
**Cons**: Model conversion required, different API

### 3. vLLM 
**Good balance of performance and ease**
```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="openai/gpt-oss-120b",
    tensor_parallel_size=4,
    dtype="bfloat16"
)
```

**Pros**: Easy setup, good performance, HF model compatibility
**Cons**: Different API, may need code refactor

### 4. DeepSpeed Inference
**Microsoft's solution**
```python
import deepspeed

model = deepspeed.init_inference(
    model,
    mp_size=4,  # tensor parallel size
    dtype=torch.bfloat16,
    replace_method="auto"
)
```

**Pros**: Good performance, relatively simple
**Cons**: Another dependency

## Recommendation: vLLM

For memorization evaluation, **vLLM** offers the best trade-off:

1. **Simple integration**: Replace model loading with vLLM LLM
2. **True tensor parallelism**: All GPUs work simultaneously
3. **Batched inference**: Optimized for our use case
4. **Memory efficiency**: Better than pipeline parallelism

## Implementation Plan

### Phase 1: Add vLLM Support
```python
# Add to eval_pz.py
def load_vllm_model(cfg):
    from vllm import LLM, SamplingParams
    
    llm = LLM(
        model=cfg.model_name,
        tensor_parallel_size=torch.cuda.device_count(),
        dtype=cfg.torch_dtype,
        max_model_len=cfg.chunk_size,
        gpu_memory_utilization=0.9
    )
    return llm

def compute_vllm_log_probs(llm, input_ids, prompt_length):
    # Convert to string prompts
    # Use llm.generate() with logprobs=True
    # Extract suffix probabilities
    pass
```

### Phase 2: Benchmark
- Compare pipeline vs tensor parallelism throughput
- Measure GPU utilization across all devices
- Validate numerical accuracy

### Phase 3: Configuration
Add to config files:
```yaml
# Parallelism strategy
parallelism_strategy: "tensor"  # or "pipeline" 
tensor_parallel_size: 4
```

## Expected Benefits
- **GPU Utilization**: 80-95% across all GPUs vs 25-69% sequential
- **Throughput**: ~3-4x faster inference
- **Memory**: More balanced distribution across GPUs
- **Scalability**: Better scaling with more GPUs

## Risks
- **API Changes**: Different interface than HuggingFace transformers  
- **Accuracy**: Need to validate identical results
- **Dependencies**: Additional package requirements