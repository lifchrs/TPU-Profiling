# Final Models Analysis Guide

This guide walks you through testing the 3 final models, measuring TTFT/TPOT metrics, and generating comparative visualizations.

## Models to Test

1. **meta-llama/Llama-3.1-8B-Instruct** (8B parameters)
2. **Qwen/Qwen3-4B** (4B parameters)
3. **Qwen/Qwen3-32B** (32B parameters)

## Tensor Parallelism Configurations

We test each model with TP sizes: **1, 2, 4, 8** (using 8 TPU chips)

## Quick Start

### Option 1: Run Complete Workflow (Recommended)

Inside the vLLM Docker container:

```bash
cd /path/to/SysML\ Project
bash setup/run_final_models_analysis.sh
```

This will:
1. ✅ Test model compatibility with TPU
2. ✅ Run batch TTFT/TPOT measurements for all models and TP sizes
3. ✅ Generate analysis plots and summary

### Option 2: Run Steps Individually

#### Step 1: Test Compatibility

```bash
python3 setup/test_final_models_compatibility.py
```

This quickly tests if each model can load and generate text with different TP sizes. It will:
- Test each model with TP=1, 2, 4, 8
- Stop testing higher TP sizes if a model fails
- Print a summary of working configurations

**Expected output:**
```
Model 1/3: meta-llama/Llama-3.1-8B-Instruct
  Testing TP=1... ✓ SUCCESS
  Testing TP=2... ✓ SUCCESS
  Testing TP=4... ✓ SUCCESS
  Testing TP=8... ✓ SUCCESS
```

#### Step 2: Run Batch Measurements

```bash
python3 setup/batch_measure_ttft_tpot.py \
    --config configs/final_models_batch_config.json
```

This will:
- Load each model with each TP size
- Run 3 measurement runs (with 1 warmup run) for each configuration
- Generate 100 tokens per run
- Save individual JSON results to `results/` directory
- Create a summary JSON file

**Expected duration:** 
- ~30-60 minutes per model (depending on model size and TP configuration)
- Total: ~2-3 hours for all 3 models × 4 TP sizes = 12 configurations

**Output files:**
- `results/meta_llama_Llama_3_1_8B_Instruct_TP1_TIMESTAMP.json`
- `results/meta_llama_Llama_3_1_8B_Instruct_TP2_TIMESTAMP.json`
- ... (one file per configuration)
- `results/summary_TIMESTAMP.json` (combined summary)

#### Step 3: Analyze Results

```bash
# Using summary file (recommended)
python3 setup/analyze_final_models_results.py \
    --summary results/summary_*.json \
    --output-dir results/

# Or analyze all JSON files in results directory
python3 setup/analyze_final_models_results.py \
    --results-dir results/ \
    --output-dir results/
```

This will:
- Generate comparison tables
- Create visualization plots:
  - `ttft_vs_tp.png` - TTFT vs Tensor Parallelism
  - `tpot_vs_tp.png` - TPOT vs Tensor Parallelism
  - `throughput_vs_tp.png` - Throughput vs Tensor Parallelism
  - `combined_comparison.png` - All three metrics side-by-side
- Generate `analysis_summary.txt` with key findings

**Note:** Requires `matplotlib`. Install with:
```bash
pip install matplotlib
```

## Configuration

The batch measurement configuration is in `configs/final_models_batch_config.json`:

```json
{
  "models": [
    {
      "name": "meta-llama/Llama-3.1-8B-Instruct",
      "tp_sizes": [1, 2, 4, 8],
      "max_model_len": 2048
    },
    ...
  ],
  "test_config": {
    "prompt": "Explain the concept of machine learning in simple terms.",
    "num_tokens": 100,
    "temperature": 0.7,
    "top_p": 0.95,
    "num_runs": 3,
    "warmup_runs": 1
  },
  "output_dir": "results"
}
```

You can modify:
- `num_runs`: Number of measurement runs (more = better statistics, but slower)
- `warmup_runs`: Number of warmup runs before measurement
- `num_tokens`: Number of tokens to generate per run
- `prompt`: Test prompt text

## Understanding the Results

### Metrics Explained

1. **TTFT (Time-to-First-Token)**: Time from prompt submission to first token generation
   - Lower is better
   - Affects user-perceived latency
   - Typically improves with higher TP (more parallel computation)

2. **TPOT (Time-per-Output-Token)**: Average time to generate each subsequent token
   - Lower is better
   - Affects generation speed
   - Also improves with higher TP

3. **Throughput**: Tokens generated per second
   - Higher is better
   - Calculated as: `num_tokens / total_time`

### Expected Patterns

- **TTFT decreases** as TP increases (more chips = faster prefill)
- **TPOT decreases** as TP increases (more chips = faster decode)
- **Throughput increases** as TP increases
- **Larger models** (32B) may show better scaling than smaller models (4B)
- **Speedup may not be linear**: TP=8 might not be 8x faster than TP=1 due to communication overhead

### Interpreting Graphs

The generated plots will show:
- **Lines for each model**: Different models may have different scaling behavior
- **Error bars**: Standard deviation across multiple runs
- **Trends**: How metrics change with TP size

## Troubleshooting

### Model Fails to Load

1. **Check TPU connectivity:**
   ```bash
   python3 -c "import jax; print(jax.devices())"
   ```

2. **Try TP=1 first** (simplest configuration)

3. **Check model access:**
   - For gated models (Llama), ensure HF token is set:
     ```bash
     export HUGGING_FACE_HUB_TOKEN=your_token_here
     ```

4. **Check available memory:**
   - Larger models (32B) may need more TPU memory
   - Try higher TP sizes for larger models

### Measurements Fail

1. **Check if model loaded successfully** (Step 1 should pass)

2. **Reduce `num_tokens`** if running out of memory

3. **Check TPU status** - ensure TPU is healthy and not preempted

### Analysis Script Fails

1. **Install matplotlib:**
   ```bash
   pip install matplotlib
   ```

2. **Check JSON files** are valid:
   ```bash
   python3 -m json.tool results/summary_*.json
   ```

## Next Steps

After completing the analysis:

1. **Review the plots** in `results/` directory
2. **Read `analysis_summary.txt`** for key findings
3. **Compare models**:
   - Which model has best TTFT at TP=8?
   - Which model scales best with TP?
   - What's the speedup from TP=1 to TP=8?
4. **Create presentation slides** using the generated visualizations
5. **Document insights** about model scaling behavior

## Files Created

```
results/
├── meta_llama_Llama_3_1_8B_Instruct_TP1_*.json
├── meta_llama_Llama_3_1_8B_Instruct_TP2_*.json
├── meta_llama_Llama_3_1_8B_Instruct_TP4_*.json
├── meta_llama_Llama_3_1_8B_Instruct_TP8_*.json
├── Qwen_Qwen3_4B_TP1_*.json
├── Qwen_Qwen3_4B_TP2_*.json
├── Qwen_Qwen3_4B_TP4_*.json
├── Qwen_Qwen3_4B_TP8_*.json
├── Qwen_Qwen3_32B_TP1_*.json
├── Qwen_Qwen3_32B_TP2_*.json
├── Qwen_Qwen3_32B_TP4_*.json
├── Qwen_Qwen3_32B_TP8_*.json
├── summary_TIMESTAMP.json
├── ttft_vs_tp.png
├── tpot_vs_tp.png
├── throughput_vs_tp.png
├── combined_comparison.png
└── analysis_summary.txt
```

## Example Workflow Timeline

1. **Compatibility test**: ~5-10 minutes
2. **Batch measurements**: ~2-3 hours (12 configurations × ~10-15 min each)
3. **Analysis**: ~1-2 minutes

**Total time**: ~2.5-3.5 hours

