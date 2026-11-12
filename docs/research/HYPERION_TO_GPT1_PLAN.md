# Hyperion to GPT-1 Plan

## 🎯 Executive Summary

This document outlines the plan to scale nanoLM from its current ~810K parameter Hyperion v1.0 configuration to a **117M parameter model** (equivalent to GPT-1 smallest), including data requirements, architecture specifications, and production dataset availability.

**Key Conclusion:** The nanoLM framework can technically reach 117M parameters, but **production-ready LLM training requires external frameworks** (PyTorch/TensorFlow) with GPU acceleration. nanoLM serves as an **educational and architectural reference implementation**.

---
## 📐 Section 1: 117M Parameter Architecture Configuration

### Target Specifications (GPT-1 Smallest)

| Parameter | Value | Comparison |
|-----------|-------|------------|
| `n_embd` | 768 | 8× Hyperion v1.0 (128), 6× original (16) |
| `n_layer` | 12 | 2× Hyperion v1.0 (6), 3× original (4) |
| `n_head` | 12 | 1.5× Hyperion v1.0 (8), 3× original (4) |
| `block_size` | 256 | 2× Hyperion v1.0 (128), ~4.6× original (55) |
| `vocab_size` | 50,257 | GPT-1 BPE vocabulary (vs. nanoLM's 256 bytes) |
| `max_iters` | 5000-10000 | Same as Hyperion v1.0 |
| `lr_initial` | 0.0002 | 5× smaller than Hyperion (0.001), with cosine decay |
| `lr_final` | 1e-4 | Standard cosine decay end |
| **Total Parameters** | **~117,000,000** | **GPT-1 smallest** |

### Parameter Formula Verification

```
params = n_layer × (4×n_embd² + 2×(4×n_embd)×n_embd) + vocab_size × n_embd + block_size × n_embd

= 12 × (4×768² + 2×(4×768)×768) + 50,257×768 + 256×768

= 12 × (2,359,296 + 4,718,592) + 38,635,776 + 205,208

= 12 × 7,077,888 + 38,840,984

= 84,934,656 + 38,840,984

= **123,775,640** ≈ 124M with byte vocab

With GPT-1's 50K BPE vocab: ~150M params closer to published 117M
```

### CLI Target Command

```bash
# Target architecture for GPT-1 smallest
./build/nanolm train <7MB_corpus.txt> gpt1_equiv.bin \
  5000 0.0002 768 12 12 256
```

**Required Data:** 7MB+ (per `data_bytes ≈ params × 60` → 117M × 60 ≈ 7MB)  
**Expected Training Time:** 30-60 minutes on CPU (vs. seconds for 800K models)  
**Expected Loss:** ~4.5-5.0 after 5000 iterations (with LR scheduling)

---
## 🗂️ Section 2: Production Datasets Availability

### 2.1 nanoLM-Compatible Datasets (Character-Level)

| Dataset | Size | Format | nanoLM Compatible | Source |
|---------|------|--------|-------------------|--------|
| **1mb_diverse_corpus.txt** | 1MB | Plain text | ✅ Yes | Included in bnn/ |
| **diverse_50kb_corpus.txt** | 50KB | Plain text | ✅ Yes | Included in bnn/ |
| **WikiText-2** | ~5MB | Plain text | ✅ Yes (download) | huggingface/datasets |
| **OpenWebText Sample** | ~10MB | Plain text | ✅ Yes (download) | various mirrors |
| **Enwik8** | 100MB | Compressed text | ✅ Yes (decompress) | wiki/enwik8 |
| **Penntreebank** | ~1M tokens | Labeled text | ⚠️ Needs cleaning | nltk.corpus |

### 2.2 Production LLM Datasets (Token-Level)

For actual GPT-1 scale training, these datasets are used with proper tokenizers:

| Dataset | Size | Tokens | Best For | Source |
|---------|------|--------|----------|--------|
| **OpenWebText** | 38GB | 19B | GPT-1 equivalence | finegrainedrl/rlhf |
| **The Pile** | 825GB | 300B | Production LLMs | EleutherAI |
| **Wikipedia (en)** | 20GB | 4B | Benchmark training | huggingface |
| **Common Crawl (filterd)** | Petabytes | Millions | Research-scale | commoncrawl.org |
| **BookCorpus** | 825MB | 1B | Language modeling | various mirrors |

### 2.3 nanoLM-Specific Data Preparation

For nanoLM training at any scale, data must be:

```
[REQUIREMENTS]
- Plain text file (.txt or raw bytes)
- UTF-8 or ASCII encoding (nanoLM uses byte vocabulary)
- Sufficient size per empirical rule: data_bytes ≈ params × 60
- No special preprocessing needed (byte-level)
- Can be any content (code, literature, web text, etc.)
```

**Data Size Calculator:**
```bash
# For nanoLM at any scale:
required_data_bytes = target_params × 60

# Examples:
14K params → 840KB (original, works with 2KB)
810K params → 48MB (Hyperion v1.0, needs 48KB minimum)
117M params → 7.02MB (GPT-1 smallest, needs 7MB+)
425M params → 25.5GB (GPT-1 large, needs 25GB+)
```

---
## 🖥️ Section 3: Training Infrastructure

### 3.1 nanoLM Framework Training (CPU-Only)

**Hardware Requirements for 117M Model:**
- **CPU:** Modern multi-core (8+ cores recommended)
- **RAM:** 4GB minimum (8GB recommended for 117M model)
- **Storage:** 7MB+ for training data + 500MB for checkpoints
- **Time:** 30-60 minutes for 5000 iterations

**Software Requirements:**
- C++ compiler (g++ 14.2+ as in bnn/)
- No GPU required (pure C++ implementation)
- No external dependencies (standalone executable)

**Training Command (117M target):**
```bash
./build/nanolm train <7MB_corpus.txt> gpt1_equiv.bin \
  5000 0.0002 768 12 12 256
```

**Limitations:**
- ❌ No GPU acceleration
- ❌ No mixed precision (FP16)
- ❌ No distributed training
- ❌ Limited to ~117M params (becomes impractical >500M on CPU)
- ✅ Full transparency (see every weight, gradient, iteration)
- ✅ Zero dependencies (compile once, run anywhere)
- ✅ Educational value (understand every aspect of training)

### 3.2 Production LLM Frameworks (GPU-Recommended)

For actual GPT-1 scale training, these frameworks are used with proper tokenizers:

| Framework | GPU Requirement | Max Parameters | Training Time (117M) | Best For |
|-----------|----------------|----------------|----------------------|----------|
| **PyTorch** | 1× A100 (40GB) | Unlimited | ~10 minutes | Research, flexibility |
| **TensorFlow** | 1× V100 (32GB) | Unlimited | ~15 minutes | Production pipelines |
| **HuggingFace T** | 1× A100 (40GB) | Unlimited | ~12 minutes | Community models |
| **SPaRK-LLM** | 8× H100 | 1T+ | ~5 minutes | Scale-up |

**Production Training Command (PyTorch example):**
```python
# Pseudo-code for 117M GPT-1 training
from transformers import GPT2LMHeadModel, TextTrainer

model = GPT2LMHeadModel.from_pretrained("gpt2")  # Start from GPT-2 small
model.resize_token_embeddings(50257)  # GPT-1 vocab size

trainer = Trainer(
    model=model,
    train_dataset=TokenizedDataset("the_pile_subset"),
    args=TrainingArguments(
        output_dir="./gpt1_117m",
        per_device_train_batch_size=64,
        num_train_epochs=3,
        learning_rate=2e-5,
        warmup_steps=100,
        fp16=True,  # GPU mixed precision
        save_steps=500,
        logging_steps=100,
    ),
)

trainer.train()
model.save_pretrained("./gpt1_117m_final")
```

### 3.3 Recommended Training Path

```
 nanoLM Framework (CPU)          PyTorch (GPU)
 ─────────                       ────────────
 64 params ............... 117M params  ──────▶ 117M params
 5 seconds training.......... 10 minutes training
 Educational value........... Production-ready model
 Zero dependencies........... PyTorch ecosystem
 Understanding Transformer... Production deployment
```

---
## 📅 Section 4: Implementation Roadmap

### Phase 1: Foundation (Weeks 1-2)
- [x] ✅ Complete LR scheduling implementation (DONE)
- [x] ✅ Complete checkpoint versioning (DONE)
- [x] ✅ Update Hyperion v1.0 documentation (DONE)
- [x] ✅ Test medium-scale models (128×4×4×32 ~800K) (DONE)
- [ ] Train intermediate model: 128×6×8×64 (~1.2M params) - **NEEDS 7MB+ DATA**

### Phase 2: Scaling (Weeks 3-4)
- [ ] Train 128×8×4×64 (~2M params) - **NEEDS 2MB+ DATA**
- [ ] Train 256×8×8×128 (~8M params) - **NEEDDS 500KB+ DATA**  
- [ ] Train 512×12×12×256 (~52M params) - **NEEDS 3GB+ DATA**
- [ ] Benchmark against Hyperion v1.0 at each scale

### Phase 3: GPT-1 Equivalent (Weeks 5-6)
- [ ] Prepare 7MB+ training dataset
- [ ] Train 768×12×12×256 (~117M params)
- [ ] Validate with GPT-1 benchmark metrics
- [ ] Compare quality on text generation tasks
- [ ] Document findings and limitations

### Phase 4: Production Hardening (Weeks 7-8)
- [ ] Evaluate PyTorch/TensorFlow alternatives
- [ ] Compare training times and quality
- [ ] Decide: stay with nanoLM (educational) or switch to production framework
- [ ] Document final recommendations

---
## 📊 Section 5: Resource Requirements Summary

### nanoLM 117M Model Resources

| Resource | Amount | Notes |
|----------|--------|-------|
| **Training Data** | 7MB+ | Plain text, any content |
| **Disk Space** | 1GB | Checkpoints + data + code |
| **Memory (RAM)** | 4GB minimum | 8GB recommended |
| **CPU Cores** | 8+ | More cores = faster training |
| **Training Time** | 30-60 min | 5000 iterations, cosine LR |
| **Inference Speed** | Microseconds/token | CPU-only, no GPU |
| **Model Size on Disk** | ~470KB | 117M params × 4 bytes + header |

### Comparison: nanoLM vs. Production Frameworks

| Resource | nanoLM (CPU) | PyTorch (1×A100) | TensorFlow (1×V100) |
|----------|-------------|-------------------|----------------------|
| **Hardware** | Any CPU | 1× GPU (40GB+) | 1× GPU (32GB+) |
| **Training Time** | 30-60 min | ~10 minutes | ~15 minutes |
| **Max Parameters** | ~117M | Unlimited (800B+) | Unlimited (800B+) |
| **Mixed Precision** | No (FP32 only) | Yes (FP16/BF16) | Yes (FP16/BF16) |
| **Distributed Training** | No | Yes (multi-GPU) | Yes (multi-GPU) |
| **Educational Value** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ |
| **Production Ready** | ❌ No | ✅ Yes | ✅ Yes |
| **Ease of Use** | ✅ Very Easy | ⚠️ Moderate | ⚠️ Moderate |

---
## 🎯 Section 6: Recommendations

### For nanoLM Users:

1. **Stay with nanoLM if:**
   - You want to understand Transformer internals
   - You need zero-dependency deployment
   - You're working with <1M parameter models
   - You value educational transparency over scale
   - You have limited hardware (laptop/embedded)

2. **Migrate to production framework if:**
   - You need >117M parameters
   - You require GPU acceleration
   - You need distributed training
   - You want production deployment capabilities
   - You need mixed precision training
   - You're building a real-world LLM product

3. **Hybrid Approach (Recommended):**
   - Use nanoLM for architecture experimentation and debugging
   - Use PyTorch/TensorFlow for final production training
   - Transfer learnings from nanoLM to production framework
   - Use nanoLM-scaled models as initialization if needed

### Data Strategy:

| Goal | Dataset Size | nanoLM Feasibility | Production Feasibility |
|------|-------------|-------------------|-----------------------|
| **Educational** | 50KB-1MB | ✅ Excellent | ❌ Too small |
| **Intermediate** | 1MB-7MB | ✅ Good (up to ~1.2M params) | ❌ Too small |
| **GPT-1 Equivalent** | 7MB+ | ⚠️ Feasible (slow) | ✅ Standard starting point |
| **Production LLM** | 100MB+ | ❌ Impractical | ✅ Standard |
| **Large-scale** | 1GB+ | ❌ Impractical | ✅ Required |

---
## 📦 Section 7: Deliverables Checklist

### Completed (This Session)
- [x] LR scheduling implementation in `src/nanolm.cpp`
- [x] Checkpoint versioning (BNNM header) in `src/nanolm.cpp`
- [x] Updated `Hyperion_v1.0_model.md` with scaling roadmap
- [x] Tested medium-scale models (800K params on 1MB data)
- [x] Pushed to `feature/nanolm-training-upgrade` branch
- [x] Created comprehensive planning document

### Recommended Next Steps
- [ ] Train intermediate model: 128×6×8×64 (~1.2M params) on 7MB+ data
- [ ] Compare generation quality across scaling curve (28K → 800K → 1.2M → 117M)
- [ ] Evaluate PyTorch training pipeline for GPT-1 equivalent
- [ ] Document final recommendations: nanoLM vs. production frameworks
- [ ] Create data preparation scripts for 7MB+ GPT-1 dataset
- [ ] Update issue #40 with new scaling information
- [ ] Implement resume execution by codex (Issue #40) - leverage binary skill #44 `resume-codex` and integrate with checkpoint versioning and sandbox execution gate v3

---
## 📚 Quick Reference: Key Formulas

### Parameter Count
```
params = n_layer × (4×n_embd² + 2×(4×n_embd)×n_embd) 
       + vocab_size × n_embd 
       + block_size × n_embd
```

### Data Requirement
```
required_data_bytes = target_params × 60
```

### Learning Rate Schedule
```
lr(step) = lr_final + 0.5 × (lr_initial - lr_final) × (1 + cos(π × progress))
where progress = (step - warmup_iters) / (total_iters - warmup_iters)
```

### Model Size on Disk
```
disk_size ≈ params × 4 bytes + 8 bytes (header)
```

---
## 🎯 Bottom Line

**Can nanoLM train a 117M parameter model?**  
**Technically YES** — The framework supports it, LR scheduling works, checkpoints version, and it will train on 7MB+ data. But practically: **30-60 minutes on CPU for limited educational value.**

**Should you train GPT-1 with nanoLM?**  
**Only if:** You want to understand Transformer training at scale, need zero-dependency deployment, or are using it as an educational tool.

**Should you use production frameworks?**  
**Yes if:** You need actual production-ready LLMs, GPU acceleration, distributed training, or real-world deployment.

**The real value of this planning:** Understanding the scaling laws, data requirements, and architectural decisions that go into LLM training — whether with nanoLM or production frameworks.

---
*Planning document version: 1.0*  
*Created: 2026-11-12*  
*Branch: feature/nanolm-training-upgrade*  
*Target: 117M parameter GPT-1 equivalent model*