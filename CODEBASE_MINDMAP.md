# Picotron Codebase Mindmap - Learning Curriculum

```
PICOTRON CODEBASE
│
├── 🎯 PROJECT OVERVIEW
│   ├── Purpose: Minimalist distributed training framework for Llama-like models
│   ├── Key Feature: 4D Parallelism (Data, Tensor, Pipeline, Context)
│   ├── Design Philosophy: Educational, hackable, <300 lines per core module
│   └── Performance: ~38% MFU on LLaMA-2-7B (64 H100s), ~50% MFU on SmolLM-1.7B (8 H100s)
│
├── 📁 CORE TRAINING PIPELINE
│   │
│   ├── train.py (Main Entry Point)
│   │   ├── Configuration Loading
│   │   │   ├── JSON config parsing
│   │   │   └── Environment variable setup
│   │   ├── Distributed Setup
│   │   │   ├── Process group initialization (NCCL/Gloo)
│   │   │   ├── Device assignment (CUDA/CPU)
│   │   │   └── Data type selection (bfloat16/float32)
│   │   ├── Model Initialization Flow
│   │   │   ├── 1. Create model config (rank 0)
│   │   │   ├── 2. Broadcast config to all ranks
│   │   │   ├── 3. Initialize with dematerialized weights (meta device)
│   │   │   ├── 4. Apply Tensor Parallel (if tp_size > 1)
│   │   │   ├── 5. Apply Pipeline Parallel (if pp_size > 1)
│   │   │   ├── 6. Materialize weights from checkpoint
│   │   │   ├── 7. Apply Context Parallel (if cp_size > 1)
│   │   │   ├── 8. Move to device & dtype
│   │   │   └── 9. Apply Data Parallel (if dp_size > 1)
│   │   ├── Training Loop
│   │   │   ├── Pipeline Parallel Training
│   │   │   │   ├── AFAB (All-Forward-All-Backward)
│   │   │   │   └── 1F1B (One-Forward-One-Backward)
│   │   │   ├── Standard Training Step
│   │   │   │   ├── Gradient accumulation loop
│   │   │   │   ├── Forward pass
│   │   │   │   ├── Loss computation
│   │   │   │   └── Backward pass
│   │   │   ├── Loss averaging (across DP/CP ranks)
│   │   │   ├── Optimizer step
│   │   │   └── Metrics calculation (MFU, throughput)
│   │   └── Checkpointing & Logging
│   │       ├── Periodic checkpoint saves
│   │       ├── WandB integration
│   │       └── Training metrics logging
│   │
│   └── create_config.py (Configuration Generator)
│       ├── Parses command-line arguments
│       ├── Loads base config template
│       ├── Overrides with user parameters
│       ├── Downloads model if needed
│       └── Saves JSON config file
│
├── 🧠 MODEL ARCHITECTURE
│   │
│   └── model.py (Llama Implementation)
│       ├── Core Components
│       │   ├── Embedding Layer
│       │   │   └── Token to hidden state mapping
│       │   ├── DecoderLayer (Repeated N times)
│       │   │   ├── Input LayerNorm (RMSNorm)
│       │   │   ├── Attention Module
│       │   │   │   ├── Q/K/V projections
│       │   │   │   ├── RoPE (Rotary Position Embedding)
│       │   │   │   ├── Attention computation
│       │   │   │   │   ├── Flash Attention (default)
│       │   │   │   │   ├── Ring Attention (Context Parallel)
│       │   │   │   │   └── PyTorch SDPA (fallback)
│       │   │   │   └── Output projection
│       │   │   ├── Residual connection
│       │   │   ├── Post-Attention LayerNorm
│       │   │   ├── MLP Module
│       │   │   │   ├── Gate projection (SiLU activation)
│       │   │   │   ├── Up projection
│       │   │   │   └── Down projection
│       │   │   └── Residual connection
│       │   ├── Final LayerNorm
│       │   └── Final Projection (vocab_size)
│       ├── Normalization Layers
│       │   ├── TritonRMSNorm (with Flash Attention)
│       │   └── LlamaRMSNorm (standard)
│       └── Initialization
│           └── Uniform weight initialization
│
├── 📊 DATA HANDLING
│   │
│   └── data.py (MicroBatchDataLoader)
│       ├── Dataset Loading
│       │   ├── HuggingFace datasets integration
│       │   ├── Tokenization (via transformers)
│       │   └── Sequence chunking (seq_length + 1)
│       ├── Distributed Sampling
│       │   ├── DistributedSampler (for DP)
│       │   └── Context Parallel sequence splitting
│       ├── Batch Collation
│       │   ├── Input IDs (split by CP rank)
│       │   ├── Target IDs (shifted by 1)
│       │   └── Position IDs (for RoPE)
│       └── Iteration Management
│           └── Automatic epoch restart
│
├── 💾 CHECKPOINT MANAGEMENT
│   │
│   └── checkpoint.py
│       ├── Weight Initialization
│       │   ├── init_model_with_dematerialized_weights()
│       │   │   └── Meta device initialization (no memory)
│       │   └── init_model_with_materialized_weights()
│       │       ├── Loads from SafeTensors
│       │       ├── Handles sharded checkpoints
│       │       ├── Adjusts tensor sizes for TP/PP
│       │       └── Converts HF naming to Picotron naming
│       ├── InitializationManager
│       │   ├── get_layer_names_in_sft_format()
│       │   │   └── Determines which layers this rank owns (PP)
│       │   ├── adjust_tensor_size()
│       │   │   ├── Splits embeddings/final_proj for TP
│       │   │   ├── Splits attention heads for TP
│       │   │   └── Splits MLP dimensions for TP
│       │   └── convert_safetensors_to_hf_name()
│       │       └── Maps SafeTensors → HuggingFace → Picotron names
│       └── CheckpointManager
│           ├── save_checkpoint()
│           │   └── Saves per TP/PP rank (DP/CP rank 0 only)
│           └── load_checkpoint()
│               └── Restores model, optimizer, step, tokens
│
├── 🔧 PROCESS GROUP MANAGEMENT
│   │
│   └── process_group_manager.py
│       ├── ProcessGroupManager Class
│       │   ├── Grid Layout
│       │   │   └── 4D Grid: [DP, PP, CP, TP]
│       │   ├── Rank Calculation
│       │   │   ├── Global rank → (dp_rank, pp_rank, cp_rank, tp_rank)
│       │   │   └── Local rank (within node)
│       │   ├── Process Groups
│       │   │   ├── tp_group (Tensor Parallel)
│       │   │   ├── cp_group (Context Parallel)
│       │   │   ├── pp_group (Pipeline Parallel)
│       │   │   ├── dp_group (Data Parallel)
│       │   │   ├── cp_dp_group (CP + DP combined)
│       │   │   └── pp_dp_group (PP + DP combined)
│       │   └── Convenience Flags
│       │       ├── pp_is_first_stage / pp_is_last_stage
│       │       ├── pp_next_rank / pp_prev_rank
│       │       └── cp_send_rank / cp_recv_rank
│       └── setup_process_group_manager()
│           └── Global singleton initialization
│
├── ⚡ PARALLELISM STRATEGIES
│   │
│   ├── 🧩 TENSOR PARALLELISM (tensor_parallel/)
│   │   │
│   │   ├── tensor_parallel.py
│   │   │   ├── apply_tensor_parallel()
│   │   │   │   ├── Replaces Linear layers with parallel versions
│   │   │   │   ├── Attention: Q/K/V (column), Out (row)
│   │   │   │   ├── MLP: Up/Gate (column), Down (row)
│   │   │   │   ├── Embedding: Vocab parallel
│   │   │   │   └── Final projection: Column (with gather)
│   │   │   ├── ColumnParallelLinear
│   │   │   │   ├── Splits weight matrix along output dimension
│   │   │   │   ├── Each rank computes: Y_i = XW_i
│   │   │   │   └── Optional gather_output for final layer
│   │   │   ├── RowParallelLinear
│   │   │   │   ├── Splits weight matrix along input dimension
│   │   │   │   ├── Each rank computes: Y_i = X_iW_i
│   │   │   │   └── All-reduce to combine: Y = sum(Y_i)
│   │   │   └── VocabParallelEmbedding
│   │   │       ├── Splits vocabulary across ranks
│   │   │       ├── Masks out-of-range tokens
│   │   │       └── All-reduce to combine embeddings
│   │   └── tp_communications.py
│   │       ├── ReduceFromModelParallelRegion
│   │       │   └── All-reduce for row parallel layers
│   │       ├── GatherFromModelParallelRegion
│   │       │   └── All-gather for column parallel layers
│   │       └── Async all-reduce variants
│   │
│   ├── 🔄 PIPELINE PARALLELISM (pipeline_parallel/)
│   │   │
│   │   ├── pipeline_parallel.py
│   │   │   ├── PipelineParallel Class
│   │   │   │   ├── distribute_layers()
│   │   │   │   │   └── Evenly distributes layers across PP ranks
│   │   │   │   ├── Forward Pass
│   │   │   │   │   ├── First stage: embedding
│   │   │   │   │   ├── Middle stages: decoder layers only
│   │   │   │   │   └── Last stage: final norm + projection
│   │   │   │   └── Backward Pass
│   │   │   │       └── Receives gradient, computes local gradients
│   │   │   ├── train_step_pipeline_afab()
│   │   │   │   ├── Phase 1: All forward passes
│   │   │   │   │   └── Store activations for backward
│   │   │   │   └── Phase 2: All backward passes
│   │   │   │       └── Reconstruct computation graph
│   │   │   └── train_step_pipeline_1f1b()
│   │   │       ├── Warmup: Fill pipeline with forwards
│   │   │       ├── Steady State: Interleave forward/backward
│   │   │       └── Cooldown: Complete remaining backwards
│   │   └── pp_communications.py
│   │       ├── pipeline_communicate()
│   │       │   ├── send_forward: Send activation to next stage
│   │       │   ├── recv_forward: Receive activation from prev stage
│   │       │   ├── send_backward: Send gradient to prev stage
│   │       │   └── recv_backward: Receive gradient from next stage
│   │       └── bidirectional_pipeline_communicate()
│   │           ├── send_fwd_recv_bwd: Optimized for 1F1B
│   │           └── send_bwd_recv_fwd: Optimized for 1F1B
│   │
│   ├── 📦 DATA PARALLELISM (data_parallel/)
│   │   │
│   │   ├── data_parallel.py
│   │   │   ├── DataParallelNaive (Reference Implementation)
│   │   │   │   ├── Simple all-reduce on gradients
│   │   │   │   └── no_sync() context manager
│   │   │   └── DataParallelBucket (Production)
│   │   │       ├── BucketManager integration
│   │   │       ├── Gradient accumulation hooks
│   │   │       ├── Bucketed all-reduce (reduces communication)
│   │   │       ├── require_backward_grad_sync flag
│   │   │       └── reset() for gradient clearing
│   │   └── bucket.py
│   │       └── BucketManager
│   │           ├── Groups parameters into buckets
│   │           ├── Marks parameters as ready
│   │           ├── Triggers all-reduce when bucket full
│   │           └── Waits for completion
│   │
│   └── 🔗 CONTEXT PARALLELISM (context_parallel/)
│       │
│       ├── context_parallel.py
│       │   ├── apply_context_parallel()
│       │   │   └── Sets CONTEXT_PARALLEL env variable
│       │   ├── ring_attention()
│       │   │   └── RingAttentionFunc (autograd Function)
│       │   ├── Ring Attention Forward
│       │   │   ├── Ring communication of K/V
│       │   │   ├── Block-wise attention computation
│       │   │   ├── Online softmax (numerically stable)
│       │   │   └── Update output and log-sum-exp
│       │   ├── Ring Attention Backward
│       │   │   ├── Ring communication of dK/dV
│       │   │   ├── Reconstruct attention matrix
│       │   │   └── Compute gradients for Q/K/V
│       │   └── update_rope_for_context_parallel()
│       │       └── Splits RoPE cos/sin by CP rank
│       └── cp_communications.py
│           └── ContextCommunicate
│               ├── Ring send/recv operations
│               ├── Commit/wait for async communication
│               └── Multiple communication streams
│
└── 🛠️ UTILITIES
    │
    └── utils.py
        ├── Printing & Logging
        │   ├── print() - Thread-safe printing
        │   └── to_readable_format() - Human-readable numbers
        ├── Randomness
        │   └── set_all_seed() - Reproducibility
        ├── Model Utilities
        │   ├── get_num_params() - Count parameters (TP/PP aware)
        │   └── assert_no_meta_tensors() - Validation
        ├── Training Metrics
        │   ├── get_mfu() - Model FLOPs Utilization
        │   └── average_loss_across_dp_cp_ranks() - Loss reduction
        └── Model Download
            └── download_model() - HuggingFace model download
```

## 📚 Learning Path (Curriculum Order)

### Phase 1: Foundation (Start Here)
1. **README.md** - Understand project goals and quick start
2. **utils.py** - Simple utility functions
3. **process_group_manager.py** - Core distributed concepts
4. **model.py** - Model architecture basics

### Phase 2: Data & Configuration
5. **data.py** - How data flows through the system
6. **create_config.py** - Configuration management
7. **checkpoint.py** - Model initialization and checkpointing

### Phase 3: Parallelism (One at a time)
8. **data_parallel/** - Simplest parallelism strategy
9. **tensor_parallel/** - Model weight sharding
10. **pipeline_parallel/** - Layer distribution
11. **context_parallel/** - Sequence length splitting

### Phase 4: Integration
12. **train.py** - How everything comes together
13. **Advanced topics**: Combining multiple parallelism strategies

## 🔑 Key Concepts to Master

### 1. Process Group Hierarchy
- **4D Grid**: [DP, PP, CP, TP] determines rank layout
- **Process Groups**: Different groups for different communication patterns
- **Rank Mapping**: Global rank → (dp, pp, cp, tp) coordinates

### 2. Weight Sharding Strategies
- **Column Parallel**: Split output dimension (Q, K, V, MLP up/gate)
- **Row Parallel**: Split input dimension (Attention out, MLP down)
- **Vocab Parallel**: Split vocabulary (Embedding, Final projection)

### 3. Pipeline Scheduling
- **AFAB**: Simple but memory-intensive
- **1F1B**: Better GPU utilization, more complex

### 4. Communication Patterns
- **All-Reduce**: Sum gradients across ranks (DP)
- **Point-to-Point**: Send/recv between pipeline stages
- **Ring**: Circular communication for context parallel

### 5. Gradient Synchronization
- **Gradient Accumulation**: Accumulate across micro-batches
- **Bucket All-Reduce**: Group gradients to reduce communication
- **Selective Sync**: Only sync on last micro-batch

## 🎯 Critical Files by Function

| Function | Primary File | Key Classes/Functions |
|----------|-------------|----------------------|
| Training Loop | `train.py` | `train_step()`, main execution |
| Model Definition | `model.py` | `Llama`, `DecoderLayer`, `Attention` |
| Data Loading | `data.py` | `MicroBatchDataLoader` |
| Checkpointing | `checkpoint.py` | `CheckpointManager`, `InitializationManager` |
| Process Groups | `process_group_manager.py` | `ProcessGroupManager` |
| Tensor Parallel | `tensor_parallel/tensor_parallel.py` | `ColumnParallelLinear`, `RowParallelLinear` |
| Pipeline Parallel | `pipeline_parallel/pipeline_parallel.py` | `PipelineParallel`, `train_step_pipeline_*` |
| Data Parallel | `data_parallel/data_parallel.py` | `DataParallelBucket` |
| Context Parallel | `context_parallel/context_parallel.py` | `RingAttentionFunc` |

## 🧪 Testing & Validation

- **tests/test_dataloader.py** - Data loading tests
- **tests/test_tensor_parallel.py** - Tensor parallel correctness

## 📖 Additional Resources

- Tutorial videos: [YouTube Playlist](https://www.youtube.com/playlist?list=PL-_armZiJvAnhcRr6yTJ0__f3Oi-LLi9S)
- Tutorial codebase: [picotron_tutorial](https://github.com/huggingface/picotron_tutorial)
- Paper: [4D Parallelism](https://arxiv.org/abs/2407.21783)

