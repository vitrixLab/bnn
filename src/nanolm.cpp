// =============================================================================
// gpt.cpp — Tiny character-level GPT in pure C++
// Architecture from AI.png:
//   n_embd     = 16
//   n_head     = 4    (d_k = 4 per head)
//   n_layer    = 4    (Transformer blocks)
//   vocab_size = 65   (character vocabulary)
//   block_size = 55   (context window)
//   Total params: 14,208 | Trainable matrices: 27
//
// Forward pass (section 1 of AI.png):
//   Input token (id)
//     → Token Embedding wte (65×16)      + wpe (55×16)
//     → RMSNorm
//     → [Transformer Block × 4]
//         ├─ 1. Attention sub-layer
//         │    RMSNorm → Q/K/V (16×16 each) → MHSA (n_head=4, d_k=4)
//         │    → wo (16×16) → residual add
//         └─ 2. MLP sub-layer
//              RMSNorm → fc1 (16×64) → ReLU → fc2 (64×16) → residual add
//     → Linear lm_head (16×65, tied to wte^T)
//     → Softmax → Cross-Entropy Loss
//
// Backward pass (section 2 of AI.png):
//   Loss ← Softmax ← lm_head ← Residual Add ←
//     MLP (fc2→ReLU→fc1→RMSNorm) ← Residual Add ←
//       Attention (wo→Softmax→Q/K/V→RMSNorm) ← … ← RMSNorm (input)
//     ← Token Embedding + Position Embedding
//
// Build:  g++ -O2 -o gpt gpt.cpp
// Train:  ./gpt train data.txt
// Gen:    ./gpt gen   model.bin "seed text"
// =============================================================================

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <iostream>
#include <cstdint>
#include <limits>
#include <utility>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

// ─────────────────────────────────────────────────────────────────────────────
// 1.  Hyper-parameters (match AI.png exactly)
// ─────────────────────────────────────────────────────────────────────────────
struct Config {
    int vocab_size = 256;  // character vocabulary (full byte range)
    int block_size = 128;  // context window (token sequence length)
    int n_embd     = 128;  // embedding dimension
    int n_head     = 8;    // attention heads (d_k = 16 per head)
    int n_layer    = 6;    // transformer blocks
    // derived
    int d_k() const { return n_embd / n_head; } // 16 per head

    // Total params ≈ 1,228,800 (~1.2M)
    // = n_layer * (4*n_embd*n_embd + 2*(4*n_embd)*n_embd)  [blocks]
    // +  vocab_size * n_embd                                [wte]
    // +  block_size * n_embd                                [wpe]
    // lm_head is weight-tied to wte (same storage, counted once)
    // ─ Trainable matrices = 6*n_layer + 2 (+lm_head tied)
};

// ─────────────────────────────────────────────────────────────────────────────
// 2.  Flat tensor helpers
// ─────────────────────────────────────────────────────────────────────────────
// We represent every tensor as a contiguous float vector.
// Conventions: matrix M[rows][cols] → M[r*cols + c]

static inline float& at(std::vector<float>& v, int r, int c, int cols)
{ return v[r * cols + c]; }
static inline float  at(const std::vector<float>& v, int r, int c, int cols)
{ return v[r * cols + c]; }

// ─────────────────────────────────────────────────────────────────────────────
// 3.  Random initialisation (Xavier-like for matrices, zero for others)
// ─────────────────────────────────────────────────────────────────────────────
static std::mt19937 rng(42);

static void xavier_init(std::vector<float>& w, int fan_in, int fan_out) {
    float scale = std::sqrt(2.0f / (fan_in + fan_out));
    std::normal_distribution<float> dist(0.0f, scale);
    for (auto& x : w) x = dist(rng);
}

static void zero(std::vector<float>& v) { std::fill(v.begin(), v.end(), 0.0f); }

// ─────────────────────────────────────────────────────────────────────────────
// 4.  Model parameters (weights) and their gradients
// ─────────────────────────────────────────────────────────────────────────────
// Named exactly as in the diagram (AI.png).
// Matrices: stored [out_features × in_features] (row = output neuron).

struct AttentionWeights {
    // Q, K, V projections: (n_embd × n_embd) = (16×16)
    std::vector<float> wq, wk, wv;   // [n_embd × n_embd]
    // Output projection wo: (n_embd × n_embd) = (16×16)
    std::vector<float> wo;            // [n_embd × n_embd]

    // Gradients (same shape)
    std::vector<float> dwq, dwk, dwv, dwo;

    // Adam moment estimates (1st and 2nd)
    std::vector<float> mwq, vwq, mwk, vwk, mwv, vwv, mwo, vwo;

    void init(int n_embd) {
        int sz = n_embd * n_embd;
        wq.resize(sz); wk.resize(sz); wv.resize(sz); wo.resize(sz);
        xavier_init(wq, n_embd, n_embd);
        xavier_init(wk, n_embd, n_embd);
        xavier_init(wv, n_embd, n_embd);
        xavier_init(wo, n_embd, n_embd);

        dwq.resize(sz,0); dwk.resize(sz,0); dwv.resize(sz,0); dwo.resize(sz,0);
        mwq.resize(sz,0); vwq.resize(sz,0);
        mwk.resize(sz,0); vwk.resize(sz,0);
        mwv.resize(sz,0); vwv.resize(sz,0);
        mwo.resize(sz,0); vwo.resize(sz,0);
    }
};

struct MLPWeights {
    // fc1: (4*n_embd × n_embd) = (64×16)
    // fc2: (n_embd × 4*n_embd) = (16×64)
    std::vector<float> fc1, fc2;      // [64×16], [16×64]
    std::vector<float> dfc1, dfc2;
    std::vector<float> mfc1, vfc1, mfc2, vfc2;

    void init(int n_embd) {
        int hidden = 4 * n_embd; // 64
        fc1.resize(hidden * n_embd); xavier_init(fc1, n_embd,  hidden);
        fc2.resize(n_embd * hidden); xavier_init(fc2, hidden, n_embd);
        dfc1.resize(hidden*n_embd,0); dfc2.resize(n_embd*hidden,0);
        mfc1.resize(hidden*n_embd,0); vfc1.resize(hidden*n_embd,0);
        mfc2.resize(n_embd*hidden,0); vfc2.resize(n_embd*hidden,0);
    }
};

struct BlockWeights {
    AttentionWeights attn;
    MLPWeights       mlp;
    void init(int n_embd) { attn.init(n_embd); mlp.init(n_embd); }
};

struct ModelWeights {
    // Token embedding wte   [vocab_size × n_embd] = [65 × 16]
    std::vector<float> wte;
    std::vector<float> dwte;
    std::vector<float> mwte, vwte;

    // Position embedding wpe [block_size × n_embd] = [55 × 16]
    std::vector<float> wpe;
    std::vector<float> dwpe;
    std::vector<float> mwpe, vwpe;

    // lm_head: weight-tied to wte (same pointer semantics; we share wte for the
    // output projection and just add gradients back to dwte).

    // Transformer blocks × 4
    std::vector<BlockWeights> blocks;

    Config cfg;

    void init(const Config& c) {
        cfg = c;
        int E = c.n_embd, V = c.vocab_size, T = c.block_size;

        wte.resize(V*E); xavier_init(wte, E, V);
        dwte.resize(V*E,0); mwte.resize(V*E,0); vwte.resize(V*E,0);

        wpe.resize(T*E); xavier_init(wpe, E, T);
        dwpe.resize(T*E,0); mwpe.resize(T*E,0); vwpe.resize(T*E,0);

        blocks.resize(c.n_layer);
        for (auto& b : blocks) b.init(E);
    }

    // Count parameters (matches "Total Parameters: 14,208" in AI.png)
    int count_params() const {
        int E = cfg.n_embd, V = cfg.vocab_size, T = cfg.block_size;
        int layer_params = cfg.n_layer * (4*E*E + 2*(4*E)*E); // 4 * (Q+K+V+wo) * E*E + fc1+fc2
        // = 4 * (4*16*16 + 16*64 + 64*16) = 4 * 3072 = 12288
        return layer_params + V*E + T*E; // + wte + wpe (lm_head is tied)
    }
};

// ─────────────────────────────────────────────────────────────────────────────
// 5.  RMSNorm  (no learnable scale — consistent with 27 matrices in AI.png)
// ─────────────────────────────────────────────────────────────────────────────
// Forward:  y[i] = x[i] / rms(x),  rms = sqrt(mean(x^2) + eps)
static void rmsnorm_fwd(const float* x, float* y, int n, float eps = 1e-5f) {
    float ms = 0.0f;
    for (int i = 0; i < n; i++) ms += x[i]*x[i];
    ms = ms / n + eps;
    float rms_inv = 1.0f / std::sqrt(ms);
    for (int i = 0; i < n; i++) y[i] = x[i] * rms_inv;
}

// Backward:  dx += dy/rms * (1 - y*y/n)   (accumulated into dx)
static void rmsnorm_bwd(const float* x, const float* dy, float* dx, int n, float eps = 1e-5f) {
    float ms = 0.0f;
    for (int i = 0; i < n; i++) ms += x[i]*x[i];
    ms = ms / n + eps;
    float rms = std::sqrt(ms);
    float rms_inv = 1.0f / rms;

    // d(rms_inv)/dms  = -1/(2 * ms^1.5)
    // dy_dot_y = sum(dy * y)   where y = x * rms_inv
    float dy_dot_y = 0.0f;
    for (int i = 0; i < n; i++) dy_dot_y += dy[i] * (x[i] * rms_inv);

    for (int i = 0; i < n; i++) {
        // dx[i] += rms_inv * dy[i]  −  x[i] * dy_dot_y / (n * ms)
        dx[i] += rms_inv * dy[i] - x[i] * dy_dot_y / (float(n) * ms);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 6.  Softmax (in-place, over a length-n array)
// ─────────────────────────────────────────────────────────────────────────────
static void softmax(float* x, int n) {
    float m = *std::max_element(x, x+n);
    float s = 0.0f;
    for (int i = 0; i < n; i++) { x[i] = std::exp(x[i] - m); s += x[i]; }
    for (int i = 0; i < n; i++) x[i] /= s;
}

// ─────────────────────────────────────────────────────────────────────────────
// 7.  Activation caches (store intermediates needed for backward pass)
// ─────────────────────────────────────────────────────────────────────────────
// Each token position t ∈ [0, T) passes through the network independently
// (no batch dimension — we process one sequence at a time for clarity).

struct AttentionCache {
    // Inputs / intermediates for one block's attention sub-layer
    std::vector<float> x_in;       // [T×E]  input  (before RMSNorm)
    std::vector<float> xn;         // [T×E]  after  RMSNorm
    std::vector<float> Q, K, V;    // [T×E]  projected Q/K/V
    std::vector<float> att;        // [H×T×T] attention weights (post-softmax)
    std::vector<float> att_pre;    // [H×T×T] pre-softmax scores (for bwd)
    std::vector<float> sv;         // [T×E]  sum_v (weighted V)
    std::vector<float> y;          // [T×E]  output of wo
};

struct MLPCache {
    std::vector<float> x_in;       // [T×E]  input  (before RMSNorm)
    std::vector<float> xn;         // [T×E]  after  RMSNorm
    std::vector<float> h;          // [T×H]  after fc1  (H = 4*E = 64)
    std::vector<float> h_relu;     // [T×H]  after ReLU
    std::vector<float> y;          // [T×E]  after fc2
};

struct BlockCache {
    AttentionCache attn;
    MLPCache       mlp;
};

struct ForwardCache {
    std::vector<float>      embed;   // [T×E]  wte+wpe
    std::vector<float>      embed_n; // [T×E]  after first RMSNorm
    std::vector<BlockCache> blocks;  // one per transformer block
    std::vector<float>      logits;  // [T×V]  lm_head output (pre-softmax)
    std::vector<float>      probs;   // [T×V]  after softmax
};

// ─────────────────────────────────────────────────────────────────────────────
// 8.  Matrix × vector  utilities
// ─────────────────────────────────────────────────────────────────────────────
// y = W * x     (W: [out × in], x: [in], y: [out])
static void matvec(const float* W, const float* x, float* y, int out, int in) {
    for (int o = 0; o < out; o++) {
        float acc = 0.0f;
        for (int i = 0; i < in; i++) acc += W[o*in + i] * x[i];
        y[o] = acc;
    }
}

// Accumulate  dW += dy * x^T   and   dx += W^T * dy
static void matvec_bwd(
    const float* W, const float* x,
    const float* dy, float* dW, float* dx,
    int out, int in)
{
    for (int o = 0; o < out; o++) {
        for (int i = 0; i < in; i++) {
            dW[o*in + i] += dy[o] * x[i];
        }
    }
    if (dx) {
        for (int i = 0; i < in; i++) {
            float acc = 0.0f;
            for (int o = 0; o < out; o++) acc += W[o*in + i] * dy[o];
            dx[i] += acc;
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 9.  Multi-Head Self-Attention  (forward)
// ─────────────────────────────────────────────────────────────────────────────
// Implements the diagram's ATTENTION SUB-LAYER:
//   xn → Q=xn·Wq, K=xn·Wk, V=xn·Wv
//   split into n_head heads (d_k = 4)
//   scores = Q·K^T / sqrt(d_k), causal mask, softmax
//   weighted sum of V → concat heads → wo → output
static void attention_fwd(
    const Config& cfg,
    const AttentionWeights& W,
    const std::vector<float>& x_in, // [T×E]
    AttentionCache& cache)
{
    int T = cfg.block_size, E = cfg.n_embd;
    int H = cfg.n_head, dk = cfg.d_k();

    cache.x_in = x_in;
    cache.xn.resize(T*E); cache.Q.resize(T*E); cache.K.resize(T*E);
    cache.V.resize(T*E);  cache.att.resize(H*T*T,0); cache.att_pre.resize(H*T*T,0);
    cache.sv.resize(T*E,0); cache.y.resize(T*E,0);

    // RMSNorm per token
    for (int t = 0; t < T; t++)
        rmsnorm_fwd(&x_in[t*E], &cache.xn[t*E], E);

    // Project Q, K, V for all tokens
    for (int t = 0; t < T; t++) {
        matvec(W.wq.data(), &cache.xn[t*E], &cache.Q[t*E], E, E);
        matvec(W.wk.data(), &cache.xn[t*E], &cache.K[t*E], E, E);
        matvec(W.wv.data(), &cache.xn[t*E], &cache.V[t*E], E, E);
    }

    float scale = 1.0f / std::sqrt((float)dk);

    // Per-head scaled dot-product attention
    zero(cache.sv);
    for (int h = 0; h < H; h++) {
        int hoff = h * dk;
        const std::size_t head_offset = static_cast<std::size_t>(h) * static_cast<std::size_t>(T) * static_cast<std::size_t>(T);

        // Compute scores[t1][t2] = Q[t1,h] · K[t2,h] * scale  (causal mask)
        for (int t1 = 0; t1 < T; t1++) {
            for (int t2 = 0; t2 < T; t2++) {
                float score = 0.0f;
                for (int d = 0; d < dk; d++)
                    score += cache.Q[t1*E + hoff+d] * cache.K[t2*E + hoff+d];
                score *= scale;
                // Causal masking: future positions → -inf
                cache.att_pre[head_offset + static_cast<std::size_t>(t1*T + t2)] = (t2 <= t1) ? score : -1e9f;
            }
            // Softmax over t2 for this t1
            float* row = &cache.att_pre[head_offset + static_cast<std::size_t>(t1*T)];
            // compute softmax into cache.att[t1*T + ...]
            float* arow = &cache.att[head_offset + static_cast<std::size_t>(t1*T)];
            float m = row[0];
            for (int t2 = 0; t2 < T; t2++) m = std::max(m, row[t2]);
            float s = 0.0f;
            for (int t2 = 0; t2 <= t1; t2++) { arow[t2] = std::exp(row[t2]-m); s += arow[t2]; }
            for (int t2 = t1+1; t2 < T; t2++) arow[t2] = 0.0f;
            for (int t2 = 0; t2 <= t1; t2++) arow[t2] /= s;
        }

        // Weighted sum of V: sv[t1, h*dk .. (h+1)*dk] = sum_t2 att[t1,t2]*V[t2, h*dk..]
        for (int t1 = 0; t1 < T; t1++) {
            for (int d = 0; d < dk; d++) {
                float acc = 0.0f;
                for (int t2 = 0; t2 <= t1; t2++)
                    acc += cache.att[head_offset + static_cast<std::size_t>(t1*T+t2)] * cache.V[t2*E + hoff+d];
                cache.sv[t1*E + hoff+d] = acc;
            }
        }
    }

    // Output projection: y = sv · wo
    for (int t = 0; t < T; t++)
        matvec(W.wo.data(), &cache.sv[t*E], &cache.y[t*E], E, E);
}

// ─────────────────────────────────────────────────────────────────────────────
// 10.  Multi-Head Self-Attention  (backward)
// ─────────────────────────────────────────────────────────────────────────────
// Returns dx_in gradient (accumulated).
static void attention_bwd(
    const Config& cfg,
    const AttentionWeights& W,
    AttentionWeights& dW,
    const AttentionCache& cache,
    const std::vector<float>& dy,   // [T×E]  gradient of output y
    std::vector<float>& dx_in)      // [T×E]  gradient accumulator for x_in
{
    int T = cfg.block_size, E = cfg.n_embd;
    int H = cfg.n_head, dk = cfg.d_k();

    // 1. grad through output projection wo
    std::vector<float> dsv(T*E, 0.0f);
    std::vector<float> dxn(T*E, 0.0f);

    for (int t = 0; t < T; t++) {
        // dsv[t] += wo^T * dy[t],  dwo += dy[t] * sv[t]^T
        matvec_bwd(W.wo.data(), &cache.sv[t*E],
                   &dy[t*E], dW.dwo.data(), &dsv[t*E], E, E);
    }

    float scale = 1.0f / std::sqrt((float)dk);

    // 2. grad through attention + weighted sum
    std::vector<float> dQ(T*E, 0.0f), dK(T*E, 0.0f), dV(T*E, 0.0f);
    std::vector<float> datt(T*T, 0.0f);

    for (int h = 0; h < H; h++) {
        int hoff = h * dk;
        const std::size_t head_offset = static_cast<std::size_t>(h) * static_cast<std::size_t>(T) * static_cast<std::size_t>(T);

        // dV: dV[t2, h] += sum_t1 att[t1,t2] * dsv[t1, h]
        for (int t2 = 0; t2 < T; t2++) {
            for (int d = 0; d < dk; d++) {
                float acc = 0.0f;
                for (int t1 = t2; t1 < T; t1++)  // t2 <= t1 (causal)
                    acc += cache.att[head_offset + static_cast<std::size_t>(t1*T + t2)] * dsv[t1*E + hoff+d];
                dV[t2*E + hoff+d] += acc;
            }
        }

        // datt[t1,t2] = sum_d dsv[t1,h,d] * V[t2,h,d]
        for (int t1 = 0; t1 < T; t1++) {
            for (int t2 = 0; t2 <= t1; t2++) {
                float acc = 0.0f;
                for (int d = 0; d < dk; d++)
                    acc += dsv[t1*E + hoff+d] * cache.V[t2*E + hoff+d];
                datt[t1*T + t2] = acc;
            }
        }

        // Backward through softmax → pre-softmax scores
        for (int t1 = 0; t1 < T; t1++) {
            const float* a  = &cache.att[head_offset + static_cast<std::size_t>(t1*T)];
            float*       da = &datt[t1*T];
            // Jacobian of softmax: d(loss)/d(score_j) = a_j*(da_j - sum_k a_k*da_k)
            float dot = 0.0f;
            for (int t2 = 0; t2 <= t1; t2++) dot += a[t2] * da[t2];
            for (int t2 = 0; t2 <= t1; t2++) da[t2] = a[t2] * (da[t2] - dot);
            // future tokens have zero gradient (masked)
            for (int t2 = t1+1; t2 < T; t2++) da[t2] = 0.0f;
        }

        // grad through Q·K^T * scale
        // dQ[t1,h] += scale * sum_t2 da[t1,t2] * K[t2,h]
        // dK[t2,h] += scale * sum_t1 da[t1,t2] * Q[t1,h]
        for (int t1 = 0; t1 < T; t1++) {
            for (int d = 0; d < dk; d++) {
                float acc = 0.0f;
                for (int t2 = 0; t2 <= t1; t2++)
                    acc += datt[t1*T + t2] * cache.K[t2*E + hoff+d];
                dQ[t1*E + hoff+d] += scale * acc;
            }
        }
        for (int t2 = 0; t2 < T; t2++) {
            for (int d = 0; d < dk; d++) {
                float acc = 0.0f;
                for (int t1 = t2; t1 < T; t1++)  // causal: t1 >= t2
                    acc += datt[t1*T + t2] * cache.Q[t1*E + hoff+d];
                dK[t2*E + hoff+d] += scale * acc;
            }
        }
    }

    // 3. grad through Q/K/V linear projections → back to xn
    for (int t = 0; t < T; t++) {
        matvec_bwd(W.wq.data(), &cache.xn[t*E], &dQ[t*E], dW.dwq.data(), &dxn[t*E], E, E);
        matvec_bwd(W.wk.data(), &cache.xn[t*E], &dK[t*E], dW.dwk.data(), &dxn[t*E], E, E);
        matvec_bwd(W.wv.data(), &cache.xn[t*E], &dV[t*E], dW.dwv.data(), &dxn[t*E], E, E);
    }

    // 4. grad through RMSNorm → accumulate into dx_in
    for (int t = 0; t < T; t++)
        rmsnorm_bwd(&cache.x_in[t*E], &dxn[t*E], &dx_in[t*E], E);
}

// ─────────────────────────────────────────────────────────────────────────────
// 11.  MLP  (forward)
// ─────────────────────────────────────────────────────────────────────────────
// MLP sub-layer: RMSNorm → fc1 (16→64) → ReLU → fc2 (64→16) → output
static void mlp_fwd(
    const Config& cfg,
    const MLPWeights& W,
    const std::vector<float>& x_in, // [T×E]
    MLPCache& cache)
{
    int T = cfg.block_size, E = cfg.n_embd, Hid = 4*E;

    cache.x_in = x_in;
    cache.xn.resize(T*E);
    cache.h.resize(T*Hid);
    cache.h_relu.resize(T*Hid);
    cache.y.resize(T*E);

    for (int t = 0; t < T; t++) {
        // RMSNorm
        rmsnorm_fwd(&x_in[t*E], &cache.xn[t*E], E);
        // fc1
        matvec(W.fc1.data(), &cache.xn[t*E], &cache.h[t*Hid], Hid, E);
        // ReLU
        for (int i = 0; i < Hid; i++)
            cache.h_relu[t*Hid+i] = std::max(0.0f, cache.h[t*Hid+i]);
        // fc2
        matvec(W.fc2.data(), &cache.h_relu[t*Hid], &cache.y[t*E], E, Hid);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 12.  MLP  (backward)
// ─────────────────────────────────────────────────────────────────────────────
static void mlp_bwd(
    const Config& cfg,
    const MLPWeights& W,
    MLPWeights& dW,
    const MLPCache& cache,
    const std::vector<float>& dy,   // [T×E]
    std::vector<float>& dx_in)      // [T×E] grad accumulator
{
    int T = cfg.block_size, E = cfg.n_embd, Hid = 4*E;

    std::vector<float> dh_relu(T*Hid, 0.0f);
    std::vector<float> dh(T*Hid, 0.0f);
    std::vector<float> dxn(T*E, 0.0f);

    for (int t = 0; t < T; t++) {
        // grad through fc2
        matvec_bwd(W.fc2.data(), &cache.h_relu[t*Hid],
                   &dy[t*E], dW.dfc2.data(), &dh_relu[t*Hid], E, Hid);
        // grad through ReLU
        for (int i = 0; i < Hid; i++)
            dh[t*Hid+i] = (cache.h[t*Hid+i] > 0.0f) ? dh_relu[t*Hid+i] : 0.0f;
        // grad through fc1
        matvec_bwd(W.fc1.data(), &cache.xn[t*E],
                   &dh[t*Hid], dW.dfc1.data(), &dxn[t*E], Hid, E);
        // grad through RMSNorm
        rmsnorm_bwd(&cache.x_in[t*E], &dxn[t*E], &dx_in[t*E], E);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 13.  Full forward pass
// ─────────────────────────────────────────────────────────────────────────────
static float forward(
    const Config& cfg,
    const ModelWeights& W,
    const std::vector<int>& tokens, // [T] integer token ids
    ForwardCache& cache,
    bool compute_loss = true)
{
    int T = cfg.block_size, E = cfg.n_embd, V = cfg.vocab_size;
    if (tokens.size() < static_cast<std::size_t>(T) ||
        (compute_loss && tokens.size() < static_cast<std::size_t>(T + 1))) {
        throw std::invalid_argument("forward: expected T context tokens and T+1 tokens when computing loss");
    }

    // ── Embed: wte + wpe ──────────────────────────────────────────────────────
    cache.embed.resize(T*E);
    for (int t = 0; t < T; t++) {
        int id = tokens[t];
        for (int e = 0; e < E; e++)
            cache.embed[t*E+e] = W.wte[id*E+e] + W.wpe[t*E+e];
    }

    // ── Input RMSNorm ─────────────────────────────────────────────────────────
    cache.embed_n.resize(T*E);
    for (int t = 0; t < T; t++)
        rmsnorm_fwd(&cache.embed[t*E], &cache.embed_n[t*E], E);

    // ── Transformer Blocks × 4 ───────────────────────────────────────────────
    cache.blocks.resize(cfg.n_layer);
    std::vector<float> x = cache.embed_n;  // current activations [T×E]

    for (int l = 0; l < cfg.n_layer; l++) {
        auto& bc = cache.blocks[l];

        // Attention sub-layer + residual
        attention_fwd(cfg, W.blocks[l].attn, x, bc.attn);
        std::vector<float> x_after_attn(T*E);
        for (int i = 0; i < T*E; i++)
            x_after_attn[i] = x[i] + bc.attn.y[i];  // residual add

        // MLP sub-layer + residual
        mlp_fwd(cfg, W.blocks[l].mlp, x_after_attn, bc.mlp);
        std::vector<float> x_after_mlp(T*E);
        for (int i = 0; i < T*E; i++)
            x_after_mlp[i] = x_after_attn[i] + bc.mlp.y[i];  // residual add

        x = x_after_mlp;
    }

    // ── lm_head (weight-tied to wte): logits = x · wte^T ────────────────────
    // lm_head shape: (V × E) — we reuse W.wte as the weight matrix
    cache.logits.resize(T*V);
    for (int t = 0; t < T; t++) {
        for (int v = 0; v < V; v++) {
            float acc = 0.0f;
            for (int e = 0; e < E; e++)
                acc += x[t*E+e] * W.wte[v*E+e];  // wte[v] is the v-th row
            cache.logits[t*V+v] = acc;
        }
    }

    // ── Softmax over vocabulary ───────────────────────────────────────────────
    cache.probs.resize(T*V);
    for (int t = 0; t < T; t++) {
        std::copy(&cache.logits[t*V], &cache.logits[t*V+V], &cache.probs[t*V]);
        softmax(&cache.probs[t*V], V);
    }

    // ── Cross-Entropy Loss (next-token prediction) ────────────────────────────
    if (!compute_loss) return 0.0f;

    float loss = 0.0f;
    for (int t = 0; t < T; t++) {
        int target = tokens[t+1];
        loss -= std::log(cache.probs[t*V + target] + 1e-9f);
    }
    return loss / float(T);
}

// ─────────────────────────────────────────────────────────────────────────────
// 14.  Full backward pass
// ─────────────────────────────────────────────────────────────────────────────
// gradient of cross-entropy + softmax combined:
//   d(loss)/d(logits[t,v]) = probs[t,v] - 1{v == target[t]}
static void backward(
    const Config& cfg,
    ModelWeights& W,
    const std::vector<int>& tokens,
    const ForwardCache& cache)
{
    int T = cfg.block_size, E = cfg.n_embd, V = cfg.vocab_size;
    if (tokens.size() < static_cast<std::size_t>(T + 1)) {
        throw std::invalid_argument("backward: expected T+1 tokens including the final next-token target");
    }

    // ── Grad of loss w.r.t. logits (softmax + cross-entropy combined) ─────────
    // Backward flows from LOSS → Softmax → Logits (lm_head)
    std::vector<float> dlogits(T*V, 0.0f);
    float scale_loss = 1.0f / float(T);
    for (int t = 0; t < T; t++) {
        int target = tokens[t+1];
        for (int v = 0; v < V; v++)
            dlogits[t*V+v] = scale_loss * cache.probs[t*V+v];
        dlogits[t*V + target] -= scale_loss;
    }

    // Reconstruct final x (output of last block) for lm_head backward
    // We need the activations that were input to lm_head.
    // These are the same as what went into the last block's mlp residual add.
    // We re-derive from cache by forward-reconstructing:
    // (In a real implementation store this; here we re-compute for clarity.)
    std::vector<float> x_last(T*E);
    {
        std::vector<float> x = cache.embed_n;
        for (int l = 0; l < cfg.n_layer; l++) {
            auto& bc = cache.blocks[l];
            std::vector<float> xa(T*E);
            for (int i = 0; i < T*E; i++) xa[i] = x[i] + bc.attn.y[i];
            std::vector<float> xm(T*E);
            for (int i = 0; i < T*E; i++) xm[i] = xa[i] + bc.mlp.y[i];
            x = xm;
        }
        x_last = x;
    }

    // ── Grad through lm_head (tied to wte) ────────────────────────────────────
    // lm_head: logits[t,v] = dot(x[t], wte[v])
    // d(loss)/d(wte[v,e]) += sum_t dlogits[t,v] * x_last[t,e]   (lm_head part)
    // d(loss)/d(x_last[t,e]) = sum_v dlogits[t,v] * wte[v,e]    (back to x)
    std::vector<float> dx_last(T*E, 0.0f);

    for (int t = 0; t < T; t++) {
        for (int v = 0; v < V; v++) {
            float dl = dlogits[t*V+v];
            for (int e = 0; e < E; e++) {
                W.dwte[v*E+e] += dl * x_last[t*E+e];   // lm_head weight grad
                dx_last[t*E+e] += dl * W.wte[v*E+e];   // back-prop to x
            }
        }
    }

    // ── Backward through transformer blocks (reverse order) ───────────────────
    // As shown in AI.png section 2: "reverse order of forward operations"
    std::vector<float> dx = dx_last;

    for (int l = cfg.n_layer - 1; l >= 0; l--) {
        auto& bc = cache.blocks[l];

        // Grad through MLP residual add: dx splits to MLP path and residual path
        std::vector<float> dx_mlp_in = dx;         // goes into MLP backward
        std::vector<float> dx_residual_mlp = dx;   // identity residual

        std::vector<float> dx_after_attn(T*E, 0.0f);
        // residual pass-through
        for (int i = 0; i < T*E; i++) dx_after_attn[i] = dx_residual_mlp[i];
        // MLP backward: adds gradient into dx_after_attn
        mlp_bwd(cfg, W.blocks[l].mlp, W.blocks[l].mlp /* grad stored in weights */,
                bc.mlp, dx_mlp_in, dx_after_attn);

        // Grad through attention residual add
        std::vector<float> dx_attn_in = dx_after_attn;    // goes into attn bwd
        std::vector<float> dx_residual_attn = dx_after_attn; // identity

        std::vector<float> dx_block_in(T*E, 0.0f);
        for (int i = 0; i < T*E; i++) dx_block_in[i] = dx_residual_attn[i];
        // Attention backward: adds gradient into dx_block_in
        attention_bwd(cfg, W.blocks[l].attn, W.blocks[l].attn /* grad in weights */,
                      bc.attn, dx_attn_in, dx_block_in);

        dx = dx_block_in;
    }

    // ── Grad through input RMSNorm → embed ────────────────────────────────────
    std::vector<float> d_embed(T*E, 0.0f);
    for (int t = 0; t < T; t++)
        rmsnorm_bwd(&cache.embed[t*E], &dx[t*E], &d_embed[t*E], E);

    // ── Grad through wte (embedding lookup) and wpe ───────────────────────────
    // d(wte[id,e]) += d_embed[t,e] for each (t, id=tokens[t])
    for (int t = 0; t < T; t++) {
        int id = tokens[t];
        for (int e = 0; e < E; e++) {
            W.dwte[id*E+e] += d_embed[t*E+e];
            W.dwpe[t*E+e]  += d_embed[t*E+e];
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 15.  Adam optimiser  (β₁=0.9, β₂=0.999, ε=1e-8)
// ─────────────────────────────────────────────────────────────────────────────
struct Adam {
    float lr = 3e-3f, lr_final = 1e-4f, lr_warmup_iters = 0.0f;
    float b1 = 0.9f, b2 = 0.999f, eps = 1e-8f;
    int   step = 0, lr_total_iters = 0;

    void update(float* w, float* g, float* m, float* v, int n) {
        // step must be > 0; bias-correction denominators (1-b1^step, 1-b2^step) are
        // zero at step=0 and would produce NaN/inf.  Always call update_all() which
        // increments step before dispatching here, or call ++step yourself first.
        if (step <= 0) {
            std::fprintf(stderr, "Adam::update(): step must be > 0 (got %d); "
                                 "call update_all() or increment step first\n", step);
            return;
        }
        float lr_t = lr_schedule(step);
        float b1t = std::pow(b1, static_cast<float>(step));
        float b2t = std::pow(b2, static_cast<float>(step));
        float alpha = lr_t * std::sqrt(1.0f - b2t) / (1.0f - b1t);
        for (int i = 0; i < n; i++) {
            m[i] = b1 * m[i] + (1.0f - b1) * g[i];
            v[i] = b2 * v[i] + (1.0f - b2) * g[i] * g[i];
            w[i] -= alpha * m[i] / (std::sqrt(v[i]) + eps);
            g[i]  = 0.0f; // zero grad after update
        }
    }

    // Cosine learning rate decay (with optional warmup)
    float lr_schedule(int iter) const {
        if (lr_warmup_iters > 0 && iter < lr_warmup_iters) {
            // Linear warmup: lr * (iter / warmup_iters)
            return lr * static_cast<float>(iter) / static_cast<float>(lr_warmup_iters);
        }
        // Cosine decay from current step to end
        float progress = static_cast<float>(iter - lr_warmup_iters) /
                         static_cast<float>(lr_total_iters - lr_warmup_iters);
        if (progress >= 1.0f) return lr_final;
        return lr_final + 0.5f * (lr - lr_final) * (1.0f + std::cos(M_PI * progress));
    }

    void update_all(ModelWeights& W, float max_grad_norm = 1.0f, float max_weight = 10.0f) {
        ++step;
        int E = W.cfg.n_embd, V = W.cfg.vocab_size, T = W.cfg.block_size;

        // ── Gradient clipping: compute global L2 norm across all grads ──
        float grad_norm_sq = 0.0f;
        auto accumulate = [&](const std::vector<float>& g) {
            for (float x : g) grad_norm_sq += x * x;
        };
        accumulate(W.dwte);
        accumulate(W.dwpe);
        for (auto& b : W.blocks) {
            accumulate(b.attn.dwq); accumulate(b.attn.dwk); accumulate(b.attn.dwv);
            accumulate(b.attn.dwo);
            accumulate(b.mlp.dfc1); accumulate(b.mlp.dfc2);
        }
        float grad_norm = std::sqrt(grad_norm_sq);
        float clip_coef = (grad_norm > max_grad_norm) ? max_grad_norm / (grad_norm + 1e-6f) : 1.0f;
        auto scale = [&](std::vector<float>& g) {
            for (float& x : g) x *= clip_coef;
        };
        scale(W.dwte); scale(W.dwpe);
        for (auto& b : W.blocks) {
            scale(b.attn.dwq); scale(b.attn.dwk); scale(b.attn.dwv); scale(b.attn.dwo);
            scale(b.mlp.dfc1); scale(b.mlp.dfc2);
        }

        auto clip_weights = [&](std::vector<float>& w) {
            for (float& x : w) {
                if (x > max_weight) x = max_weight;
                if (x < -max_weight) x = -max_weight;
            }
        };

        update(W.wte.data(), W.dwte.data(), W.mwte.data(), W.vwte.data(), V*E);
        clip_weights(W.wte);
        update(W.wpe.data(), W.dwpe.data(), W.mwpe.data(), W.vwpe.data(), T*E);
        clip_weights(W.wpe);
        for (auto& b : W.blocks) {
            auto& a  = b.attn;
            auto& m  = b.mlp;
            int  sz  = E*E;
            int  szf = 4*E*E;
            update(a.wq.data(), a.dwq.data(), a.mwq.data(), a.vwq.data(), sz);
            clip_weights(a.wq);
            update(a.wk.data(), a.dwk.data(), a.mwk.data(), a.vwk.data(), sz);
            clip_weights(a.wk);
            update(a.wv.data(), a.dwv.data(), a.mwv.data(), a.vwv.data(), sz);
            clip_weights(a.wv);
            update(a.wo.data(), a.dwo.data(), a.mwo.data(), a.vwo.data(), sz);
            clip_weights(a.wo);
            update(m.fc1.data(), m.dfc1.data(), m.mfc1.data(), m.vfc1.data(), szf);
            clip_weights(m.fc1);
            update(m.fc2.data(), m.dfc2.data(), m.mfc2.data(), m.vfc2.data(), szf);
            clip_weights(m.fc2);
        }
    }
};

// ─────────────────────────────────────────────────────────────────────────────
// 16.  Character tokeniser
// ─────────────────────────────────────────────────────────────────────────────
struct Tokenizer {
    std::unordered_map<char,int> c2i;
    std::vector<char>            i2c;

    void build(const std::string& text) {
        std::vector<bool> seen(256, false);
        for (char c : text) seen[(unsigned char)c] = true;
        for (int i = 0; i < 256; i++) {
            if (seen[i]) { c2i[(char)i] = (int)i2c.size(); i2c.push_back((char)i); }
        }
    }

    int  vocab_size() const { return (int)i2c.size(); }
    int  encode(char c) const { auto it = c2i.find(c); return it==c2i.end()?0:it->second; }
    char decode(int i) const  { return i >= 0 && i < (int)i2c.size() ? i2c[i] : '?'; }

    std::vector<int> encode(const std::string& s) const {
        std::vector<int> ids(s.size());
        for (size_t i = 0; i < s.size(); i++) ids[i] = encode(s[i]);
        return ids;
    }
    std::string decode(const std::vector<int>& ids) const {
        std::string s(ids.size(), '?');
        for (size_t i = 0; i < ids.size(); i++) s[i] = decode(ids[i]);
        return s;
    }

    void save(const std::string& path) const {
        std::ofstream f(path); f << std::string(i2c.begin(), i2c.end());
    }
    bool load(const std::string& path) {
        std::ifstream f(path);
        if (!f) return false;
        std::string s; std::getline(f, s, '\0');
        for (int i = 0; i < (int)s.size(); i++) { c2i[s[i]] = i; i2c.push_back(s[i]); }
        return true;
    }
};

// ─────────────────────────────────────────────────────────────────────────────
// 17.  Model save / load  (simple binary format)
//
// File layout (P3-C: ABI documented here — all fields native byte order, which
// is little-endian on x86/x86-64):
//
//   [0] Config struct          (sizeof(Config) bytes, compiler-native packing)
//   [sizeof(Config)] float[]   wte  (vocab_size × n_embd floats)
//   ...                        wpe  (block_size × n_embd floats)
//   per block:                 wq, wk, wv, wo  (n_embd × n_embd each)
//                              fc1 (n_embd × 4*n_embd), fc2 (4*n_embd × n_embd)
//
// Portability limitations:
//   - No magic number or version field — incompatible files give UB on load.
//   - Struct packing is compiler-defined; may differ between MSVC and GCC/Clang.
//   - Byte order is native (little-endian); files are not portable to big-endian.
//   A future version should add a 8-byte header: magic("BNNM") + version(uint32).
// ─────────────────────────────────────────────────────────────────────────────
static bool save_weights(const ModelWeights& W, const std::string& path) {
    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    if (!f.is_open()) {
        std::fprintf(stderr, "save_weights: cannot open %s\n", path.c_str());
        return false;
    }
    // Write magic + version header for format compatibility

    const char magic[] = "BNNM";

    uint32_t version = 1;

    f.write(magic, 4);

    f.write(reinterpret_cast<const char*>(&version), 4);


    auto write_bytes = [&](const void* data, std::size_t bytes, const char* name) {
        if (bytes > static_cast<std::size_t>(std::numeric_limits<std::streamsize>::max())) {
            std::fprintf(stderr, "save_weights: tensor '%s' is too large\n", name);
            return false;
        }
        f.write(static_cast<const char*>(data), static_cast<std::streamsize>(bytes));
        if (!f.good()) {
            std::fprintf(stderr, "save_weights: failed writing '%s' to %s\n", name, path.c_str());
            return false;
        }
        return true;
    };
    auto write_tensor = [&](const std::vector<float>& values, const char* name) {
        return write_bytes(values.data(), values.size() * sizeof(float), name);
    };
    if (!write_bytes(&W.cfg, sizeof(Config), "config") ||
        !write_tensor(W.wte, "wte") || !write_tensor(W.wpe, "wpe")) return false;
    for (std::size_t bi = 0; bi < W.blocks.size(); ++bi) {
        const auto& b = W.blocks[bi];
        const std::string prefix = "block " + std::to_string(bi) + " ";
        if (!write_tensor(b.attn.wq, (prefix + "wq").c_str()) ||
            !write_tensor(b.attn.wk, (prefix + "wk").c_str()) ||
            !write_tensor(b.attn.wv, (prefix + "wv").c_str()) ||
            !write_tensor(b.attn.wo, (prefix + "wo").c_str()) ||
            !write_tensor(b.mlp.fc1, (prefix + "fc1").c_str()) ||
            !write_tensor(b.mlp.fc2, (prefix + "fc2").c_str())) return false;
    }
    f.flush();
    if (!f.good()) {
        std::fprintf(stderr, "save_weights: flush failed for %s\n", path.c_str());
        return false;
    }
    f.close();
    if (f.fail()) {
        std::fprintf(stderr, "save_weights: close failed for %s\n", path.c_str());
        return false;
    }
    std::printf("[save] wrote %s\n", path.c_str());
    return true;
}

// ─────────────────────────────────────────────────────────────────────────────
// 17b. Training state save / load — versioned "BNNS" format v1
// ─────────────────────────────────────────────────────────────────────────────
// Self-describing training-state checkpoint (native byte order). Layout:
//   [0..3]    magic "BNNS"
//   [4..7]    format version = 1 (uint32)
//   [8..]     Config (sizeof(Config) bytes)
//   +0        Adam step (int32)
//   +4        Adam lr              (float)
//   +8        Adam lr_final        (float)
//   +12       Adam lr_warmup_iters (float)
//   +16       Adam lr_total_iters  (int32)
//   +20       Adam b1  (float)
//   +24       Adam b2  (float)
//   +28       Adam eps (float)
//   +32       mt19937 state text length (uint32)
//   ...       mt19937 state text (standard <random> textual form; exact
//             round-trip via operator<< / operator>>)
//   ...       corpus token count at save time (uint64) — defines the batch
//             sampler range; resume must verify it matches the live corpus
//   ...       Adam moment tensors, fixed order:
//               mwte, vwte, mwpe, vwpe,
//               per block: mwq, vwq, mwk, vwk, mwv, vwv, mwo, vwo,
//                          mfc1, vfc1, mfc2, vfc2
// load_training_state() enforces exact-size equality (rejecting truncated
// payloads and trailing bytes), validates the versioned header and the
// stored Config against the live model, and is transactional: live model /
// optimizer / RNG state is mutated only after every field has been read and
// validated successfully. Compatibility: version 1 only.
//
// Scope: this is optimizer + data-sampler state for interrupted-training
// resume. The global init/generation RNG (std::mt19937 rng(42)) is NOT part
// of the training state — it seeds initialisation and generation sampling
// only, and is irrelevant once weights are loaded from a checkpoint.
// ─────────────────────────────────────────────────────────────────────────────

// mt19937 round-trips through its standard-defined textual representation.
static std::string mt19937_serialize(const std::mt19937& engine) {
    std::ostringstream os;
    os << engine;
    return os.str();
}

static bool mt19937_deserialize(const std::string& text, std::mt19937& engine) {
    std::istringstream is(text);
    is >> engine;
    return !is.fail();
}

static bool save_training_state(const ModelWeights& W, const Adam& opt,
                                const std::mt19937& data_rng,
                                std::uint64_t data_token_count,
                                const std::string& path) {
    const std::string rng_text = mt19937_serialize(data_rng);
    if (rng_text.empty() || rng_text.size() > 624u * 12u) {
        std::fprintf(stderr, "save_training_state: unexpected mt19937 state text length (%zu)\n",
                     rng_text.size());
        return false;
    }
    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    if (!f.is_open()) {
        std::fprintf(stderr, "save_training_state: cannot open %s\n", path.c_str());
        return false;
    }

    const char magic[4] = { 'B', 'N', 'N', 'S' };
    const std::uint32_t version = 1;
    const std::int32_t step32 = static_cast<std::int32_t>(opt.step);
    const std::int32_t total32 = static_cast<std::int32_t>(opt.lr_total_iters);
    const std::uint32_t rng_len = static_cast<std::uint32_t>(rng_text.size());

    auto write_bytes = [&](const void* data, std::size_t bytes, const char* what) {
        if (bytes > static_cast<std::size_t>(std::numeric_limits<std::streamsize>::max())) {
            std::fprintf(stderr, "save_training_state: field '%s' is too large\n", what);
            return false;
        }
        f.write(static_cast<const char*>(data), static_cast<std::streamsize>(bytes));
        if (!f.good()) {
            std::fprintf(stderr, "save_training_state: failed writing '%s' to %s\n", what, path.c_str());
            return false;
        }
        return true;
    };
    auto write_tensor = [&](const std::vector<float>& values, const char* name) {
        return write_bytes(values.data(), values.size() * sizeof(float), name);
    };

    bool ok =
        write_bytes(magic, 4, "magic") &&
        write_bytes(&version, 4, "version") &&
        write_bytes(&W.cfg, sizeof(Config), "config") &&
        write_bytes(&step32, 4, "adam step") &&
        write_bytes(&opt.lr, 4, "adam lr") &&
        write_bytes(&opt.lr_final, 4, "adam lr_final") &&
        write_bytes(&opt.lr_warmup_iters, 4, "adam lr_warmup_iters") &&
        write_bytes(&total32, 4, "adam lr_total_iters") &&
        write_bytes(&opt.b1, 4, "adam b1") &&
        write_bytes(&opt.b2, 4, "adam b2") &&
        write_bytes(&opt.eps, 4, "adam eps") &&
        write_bytes(&rng_len, 4, "rng length") &&
        write_bytes(rng_text.data(), rng_text.size(), "rng state") &&
        write_bytes(&data_token_count, 8, "token count") &&
        write_tensor(W.mwte, "mwte") &&
        write_tensor(W.vwte, "vwte") &&
        write_tensor(W.mwpe, "mwpe") &&
        write_tensor(W.vwpe, "vwpe");
    for (std::size_t bi = 0; ok && bi < W.blocks.size(); ++bi) {
        const auto& b = W.blocks[bi];
        const std::string prefix = "block " + std::to_string(bi) + " ";
        ok = write_tensor(b.attn.mwq, (prefix + "mwq").c_str()) &&
             write_tensor(b.attn.vwq, (prefix + "vwq").c_str()) &&
             write_tensor(b.attn.mwk, (prefix + "mwk").c_str()) &&
             write_tensor(b.attn.vwk, (prefix + "vwk").c_str()) &&
             write_tensor(b.attn.mwv, (prefix + "mwv").c_str()) &&
             write_tensor(b.attn.vwv, (prefix + "vwv").c_str()) &&
             write_tensor(b.attn.mwo, (prefix + "mwo").c_str()) &&
             write_tensor(b.attn.vwo, (prefix + "vwo").c_str()) &&
             write_tensor(b.mlp.mfc1, (prefix + "mfc1").c_str()) &&
             write_tensor(b.mlp.vfc1, (prefix + "vfc1").c_str()) &&
             write_tensor(b.mlp.mfc2, (prefix + "mfc2").c_str()) &&
             write_tensor(b.mlp.vfc2, (prefix + "vfc2").c_str());
    }
    if (!ok) return false;
    f.flush();
    if (!f.good()) {
        std::fprintf(stderr, "save_training_state: flush failed for %s\n", path.c_str());
        return false;
    }
    f.close();
    if (f.fail()) {
        std::fprintf(stderr, "save_training_state: close failed for %s\n", path.c_str());
        return false;
    }
    std::printf("[save] wrote training state to %s\n", path.c_str());
    return true;
}

// ─────────────────────────────────────────────────────────────────────────────
// 17c. Training state load — validates header/configuration/exact size, reads
//      transactionally into candidates, commits only on full success.
// ─────────────────────────────────────────────────────────────────────────────
static bool config_equal(const Config& a, const Config& b) {
    return a.vocab_size == b.vocab_size && a.block_size == b.block_size &&
           a.n_embd == b.n_embd && a.n_head == b.n_head && a.n_layer == b.n_layer;
}

static bool load_training_state(ModelWeights& W, Adam& opt,
                                std::mt19937& data_rng,
                                std::uint64_t& data_token_count,
                                const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f.is_open()) {
        std::fprintf(stderr, "load_training_state: cannot open %s\n", path.c_str());
        return false;
    }

    auto read_bytes = [&](void* dst, std::size_t bytes, const char* what) {
        if (bytes > static_cast<std::size_t>(std::numeric_limits<std::streamsize>::max())) {
            std::fprintf(stderr, "load_training_state: field '%s' is too large in %s\n", what, path.c_str());
            return false;
        }
        f.read(static_cast<char*>(dst), static_cast<std::streamsize>(bytes));
        if (!f.good()) {
            std::fprintf(stderr, "load_training_state: failed reading '%s' from %s\n", what, path.c_str());
            return false;
        }
        return true;
    };

    char magic[4] = {0};
    if (!read_bytes(magic, 4, "magic")) return false;
    if (std::strncmp(magic, "BNNS", 4) != 0) {
        std::fprintf(stderr, "load_training_state: invalid training-state magic in %s\n", path.c_str());
        return false;
    }
    std::uint32_t version = 0;
    if (!read_bytes(&version, 4, "version")) return false;
    if (version != 1) {
        std::fprintf(stderr, "load_training_state: unsupported training-state version %u in %s\n",
                     static_cast<unsigned>(version), path.c_str());
        return false;
    }
    Config state_cfg{};
    if (!read_bytes(&state_cfg, sizeof(Config), "config")) return false;
    if (state_cfg.vocab_size <= 0 || state_cfg.block_size <= 0 || state_cfg.n_embd <= 0 ||
        state_cfg.n_head <= 0 || state_cfg.n_layer <= 0 || state_cfg.n_embd % state_cfg.n_head != 0 ||
        state_cfg.n_head > state_cfg.n_embd || state_cfg.vocab_size > 1000000 ||
        state_cfg.block_size > 65536 || state_cfg.n_embd > 4096 || state_cfg.n_layer > 1024) {
        std::fprintf(stderr, "load_training_state: invalid or unsupported configuration in %s\n", path.c_str());
        return false;
    }
    if (!config_equal(state_cfg, W.cfg)) {
        std::fprintf(stderr, "load_training_state: configuration mismatch in %s "
                             "(state V=%d T=%d E=%d H=%d L=%d; live V=%d T=%d E=%d H=%d L=%d)\n",
                     path.c_str(),
                     state_cfg.vocab_size, state_cfg.block_size, state_cfg.n_embd,
                     state_cfg.n_head, state_cfg.n_layer,
                     W.cfg.vocab_size, W.cfg.block_size, W.cfg.n_embd,
                     W.cfg.n_head, W.cfg.n_layer);
        return false;
    }

    // Transactional read of the fixed scalar section (Adam fields) that sits
    // between the Config header and the variable-length RNG text.
    Adam cand_opt;
    {
        std::int32_t step32 = 0, total32 = 0;
        if (!read_bytes(&step32, 4, "adam step") ||
            !read_bytes(&cand_opt.lr, 4, "adam lr") ||
            !read_bytes(&cand_opt.lr_final, 4, "adam lr_final") ||
            !read_bytes(&cand_opt.lr_warmup_iters, 4, "adam lr_warmup_iters") ||
            !read_bytes(&total32, 4, "adam lr_total_iters") ||
            !read_bytes(&cand_opt.b1, 4, "adam b1") ||
            !read_bytes(&cand_opt.b2, 4, "adam b2") ||
            !read_bytes(&cand_opt.eps, 4, "adam eps")) return false;
        cand_opt.step = static_cast<int>(step32);
        cand_opt.lr_total_iters = static_cast<int>(total32);
        if (step32 < 0 || total32 < 0 ||
            !std::isfinite(cand_opt.lr) || cand_opt.lr < 0.0f ||
            !std::isfinite(cand_opt.lr_final) || cand_opt.lr_final < 0.0f ||
            !std::isfinite(cand_opt.lr_warmup_iters) || cand_opt.lr_warmup_iters < 0.0f ||
            !(cand_opt.b1 > 0.0f && cand_opt.b1 < 1.0f) ||
            !(cand_opt.b2 > 0.0f && cand_opt.b2 < 1.0f) ||
            !(cand_opt.eps > 0.0f && std::isfinite(cand_opt.eps))) {
            std::fprintf(stderr, "load_training_state: implausible Adam hyperparameters in %s\n", path.c_str());
            return false;
        }
    }

    std::uint32_t rng_len = 0;
    if (!read_bytes(&rng_len, 4, "rng length")) return false;
    if (rng_len == 0 || rng_len > 624u * 12u) {
        std::fprintf(stderr, "load_training_state: implausible mt19937 state length %u in %s\n",
                     static_cast<unsigned>(rng_len), path.c_str());
        return false;
    }

    // Exact-size expectation from the stored Config: reject truncated payloads
    // and trailing bytes alike, before allocating anything.
    const std::uint64_t E = static_cast<std::uint64_t>(state_cfg.n_embd);
    const std::uint64_t V = static_cast<std::uint64_t>(state_cfg.vocab_size);
    const std::uint64_t T = static_cast<std::uint64_t>(state_cfg.block_size);
    const std::uint64_t L = static_cast<std::uint64_t>(state_cfg.n_layer);
    std::uint64_t moment_floats = 0;
    auto add_product = [&](std::uint64_t a, std::uint64_t b) {
        const std::uint64_t max_value = std::numeric_limits<std::uint64_t>::max();
        if (a != 0 && b > (max_value - moment_floats) / a) return false;
        moment_floats += a * b;
        return true;
    };
    if (!add_product(V, E) || !add_product(V, E) ||   // mwte, vwte
        !add_product(T, E) || !add_product(T, E) ||   // mwpe, vwpe
        !add_product(L, 24ULL * E * E)) {             // 8 attention (E²) + 4 MLP (4E²) moments per block
        std::fprintf(stderr, "load_training_state: checkpoint dimensions overflow\n");
        return false;
    }
    const std::uint64_t fixed_bytes = 44ULL + sizeof(Config) + 8ULL; // header + Adam fields + rng_len + token count
    const std::uint64_t max_file_bytes = 4ULL * 1024ULL * 1024ULL * 1024ULL;
    if (moment_floats > (max_file_bytes - fixed_bytes - rng_len) / 4ULL) {
        std::fprintf(stderr, "load_training_state: training state exceeds 4 GiB safety limit\n");
        return false;
    }
    const std::uint64_t expected_bytes = fixed_bytes + rng_len + moment_floats * 4ULL;
    f.seekg(0, std::ios::end);
    const std::streamoff actual_size = f.tellg();
    if (actual_size < 0 || static_cast<std::uint64_t>(actual_size) != expected_bytes) {
        std::fprintf(stderr, "load_training_state: size mismatch for %s (expected %llu bytes, got %lld)\n",
                     path.c_str(), static_cast<unsigned long long>(expected_bytes),
                     static_cast<long long>(actual_size));
        return false;
    }
    // Rewind to the start of the RNG text (magic + version + config + Adam
    // fields + rng length have already been consumed; seekg(end) moved us).
    f.seekg(static_cast<std::streamoff>(44 + sizeof(Config)), std::ios::beg);
    if (!f.good()) {
        std::fprintf(stderr, "load_training_state: cannot seek to payload in %s\n", path.c_str());
        return false;
    }

    std::mt19937 cand_rng;
    std::uint64_t cand_tokens = 0;
    {
        std::string rng_text(rng_len, '\0');
        if (!read_bytes(&rng_text[0], rng_len, "rng state")) return false;
        if (!read_bytes(&cand_tokens, 8, "token count")) return false;
        if (!mt19937_deserialize(rng_text, cand_rng)) {
            std::fprintf(stderr, "load_training_state: malformed mt19937 state in %s\n", path.c_str());
            return false;
        }
    }

    ModelWeights candidate;
    candidate.init(state_cfg);
    auto read_tensor = [&](std::vector<float>& values, const char* name) {
        const std::size_t bytes = values.size() * sizeof(float);
        if (bytes > static_cast<std::size_t>(std::numeric_limits<std::streamsize>::max())) {
            std::fprintf(stderr, "load_training_state: tensor '%s' is too large\n", name);
            return false;
        }
        f.read(reinterpret_cast<char*>(values.data()), static_cast<std::streamsize>(bytes));
        if (!f.good()) {
            std::fprintf(stderr, "load_training_state: failed reading tensor '%s' from %s\n", name, path.c_str());
            return false;
        }
        return true;
    };
    if (!read_tensor(candidate.mwte, "mwte") || !read_tensor(candidate.vwte, "vwte") ||
        !read_tensor(candidate.mwpe, "mwpe") || !read_tensor(candidate.vwpe, "vwpe")) return false;
    for (std::size_t bi = 0; bi < candidate.blocks.size(); ++bi) {
        auto& b = candidate.blocks[bi];
        const std::string prefix = "block " + std::to_string(bi) + " ";
        if (!read_tensor(b.attn.mwq, (prefix + "mwq").c_str()) ||
            !read_tensor(b.attn.vwq, (prefix + "vwq").c_str()) ||
            !read_tensor(b.attn.mwk, (prefix + "mwk").c_str()) ||
            !read_tensor(b.attn.vwk, (prefix + "vwk").c_str()) ||
            !read_tensor(b.attn.mwv, (prefix + "mwv").c_str()) ||
            !read_tensor(b.attn.vwv, (prefix + "vwv").c_str()) ||
            !read_tensor(b.attn.mwo, (prefix + "mwo").c_str()) ||
            !read_tensor(b.attn.vwo, (prefix + "vwo").c_str()) ||
            !read_tensor(b.mlp.mfc1, (prefix + "mfc1").c_str()) ||
            !read_tensor(b.mlp.vfc1, (prefix + "vfc1").c_str()) ||
            !read_tensor(b.mlp.mfc2, (prefix + "mfc2").c_str()) ||
            !read_tensor(b.mlp.vfc2, (prefix + "vfc2").c_str())) return false;
    }

    // Commit: every field read and validated — now mutate live state.
    W.mwte = std::move(candidate.mwte);
    W.vwte = std::move(candidate.vwte);
    W.mwpe = std::move(candidate.mwpe);
    W.vwpe = std::move(candidate.vwpe);
    for (std::size_t bi = 0; bi < W.blocks.size(); ++bi) {
        auto& dst = W.blocks[bi];
        auto& src = candidate.blocks[bi];
        dst.attn.mwq = std::move(src.attn.mwq);
        dst.attn.vwq = std::move(src.attn.vwq);
        dst.attn.mwk = std::move(src.attn.mwk);
        dst.attn.vwk = std::move(src.attn.vwk);
        dst.attn.mwv = std::move(src.attn.mwv);
        dst.attn.vwv = std::move(src.attn.vwv);
        dst.attn.mwo = std::move(src.attn.mwo);
        dst.attn.vwo = std::move(src.attn.vwo);
        dst.mlp.mfc1 = std::move(src.mlp.mfc1);
        dst.mlp.vfc1 = std::move(src.mlp.vfc1);
        dst.mlp.mfc2 = std::move(src.mlp.mfc2);
        dst.mlp.vfc2 = std::move(src.mlp.vfc2);
    }
    opt = cand_opt;
    data_rng = std::move(cand_rng);
    data_token_count = cand_tokens;
    std::printf("[load] loaded training state from %s\n", path.c_str());
    return true;
}

static bool load_weights(ModelWeights& W, const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f.is_open()) {
        std::fprintf(stderr, "load_weights: cannot open %s\n", path.c_str());
        return false;
    }
    // Read and validate versioned checkpoint header
    char magic[4] = {0};
    uint32_t version = 0;
    f.read(magic, 4);
    if (std::strncmp(magic, "BNNM", 4) != 0) {
        std::fprintf(stderr, "load_weights: invalid checkpoint magic in %s\n", path.c_str());
        return false;
    }
    f.read(reinterpret_cast<char*>(&version), 4);
    if (version != 1) {
        std::fprintf(stderr, "load_weights: unsupported checkpoint version %u in %s\n", version, path.c_str());
        return false;
    }
    Config cfg{};
    f.read(reinterpret_cast<char*>(&cfg), static_cast<std::streamsize>(sizeof(Config)));
    if (!f.good()) {
        std::fprintf(stderr, "load_weights: incomplete configuration header in %s\n", path.c_str());
        return false;
    }
    if (cfg.vocab_size <= 0 || cfg.block_size <= 0 || cfg.n_embd <= 0 ||
        cfg.n_head <= 0 || cfg.n_layer <= 0 || cfg.n_embd % cfg.n_head != 0 ||
        cfg.n_head > cfg.n_embd || cfg.vocab_size > 1000000 ||
        cfg.block_size > 65536 || cfg.n_embd > 4096 || cfg.n_layer > 1024) {
        std::fprintf(stderr, "load_weights: invalid or unsupported configuration in %s\n", path.c_str());
        return false;
    }

    // Calculate exact checkpoint size with overflow checks before any tensor allocation.
    std::uint64_t float_count = 0;
    auto add_product = [&](std::uint64_t a, std::uint64_t b) {
        const std::uint64_t max_value = std::numeric_limits<std::uint64_t>::max();
        if (a != 0 && b > (max_value - float_count) / a) return false;
        float_count += a * b;
        return true;
    };
    const std::uint64_t E = static_cast<std::uint64_t>(cfg.n_embd);
    const std::uint64_t V = static_cast<std::uint64_t>(cfg.vocab_size);
    const std::uint64_t T = static_cast<std::uint64_t>(cfg.block_size);
    const std::uint64_t L = static_cast<std::uint64_t>(cfg.n_layer);
    if (!add_product(V, E) || !add_product(T, E) || !add_product(L, 12U * E * E)) {
        std::fprintf(stderr, "load_weights: checkpoint dimensions overflow\n");
        return false;
    }
    const std::uint64_t max_file_bytes = 1024ULL * 1024ULL * 1024ULL;
    if (float_count > (max_file_bytes - sizeof(Config)) / sizeof(float)) {
        std::fprintf(stderr, "load_weights: checkpoint exceeds 1 GiB safety limit\n");
        return false;
    }
    // 8 bytes: magic("BNNM") + version(uint32)
    const std::uint64_t expected_bytes =
        static_cast<std::uint64_t>(sizeof(Config)) + 8 + float_count * sizeof(float);
    f.seekg(0, std::ios::end);
    const std::streamoff actual_size = f.tellg();
    if (actual_size < 0 || static_cast<std::uint64_t>(actual_size) != expected_bytes) {
        std::fprintf(stderr, "load_weights: checkpoint size mismatch for %s (expected %llu bytes)\n",
                     path.c_str(), static_cast<unsigned long long>(expected_bytes));
        return false;
    }
    // Payload starts after magic (4) + version (4) + Config. The historic
    // seek to sizeof(Config) alone read every tensor 8 bytes early — corrupting
    // all loaded weights — and was fixed after the interrupted-vs-uninterrupted
    // equivalence test surfaced it (Issue #40, 2026-11-13).
    f.seekg(static_cast<std::streamoff>(8 + sizeof(Config)), std::ios::beg);
    if (!f.good()) {
        std::fprintf(stderr, "load_weights: cannot seek to payload in %s\n", path.c_str());
        return false;
    }

    // Commit only after the complete payload has been read successfully.
    ModelWeights candidate;
    candidate.init(cfg);
    auto read_tensor = [&](std::vector<float>& values, const char* name) {
        const std::size_t bytes = values.size() * sizeof(float);
        if (bytes > static_cast<std::size_t>(std::numeric_limits<std::streamsize>::max())) {
            std::fprintf(stderr, "load_weights: tensor '%s' is too large\n", name);
            return false;
        }
        f.read(reinterpret_cast<char*>(values.data()), static_cast<std::streamsize>(bytes));
        if (!f.good()) {
            std::fprintf(stderr, "load_weights: failed reading tensor '%s' from %s\n", name, path.c_str());
            return false;
        }
        return true;
    };
    if (!read_tensor(candidate.wte, "wte") || !read_tensor(candidate.wpe, "wpe")) return false;
    for (std::size_t bi = 0; bi < candidate.blocks.size(); ++bi) {
        auto& b = candidate.blocks[bi];
        const std::string prefix = "block " + std::to_string(bi) + " ";
        if (!read_tensor(b.attn.wq, (prefix + "wq").c_str()) ||
            !read_tensor(b.attn.wk, (prefix + "wk").c_str()) ||
            !read_tensor(b.attn.wv, (prefix + "wv").c_str()) ||
            !read_tensor(b.attn.wo, (prefix + "wo").c_str()) ||
            !read_tensor(b.mlp.fc1, (prefix + "fc1").c_str()) ||
            !read_tensor(b.mlp.fc2, (prefix + "fc2").c_str())) return false;
    }
    W = std::move(candidate);
    std::printf("[load] loaded %s\n", path.c_str());
    return true;
}

// ─────────────────────────────────────────────────────────────────────────────
// 18.  Text generation  (greedy or temperature-based sampling)
// ─────────────────────────────────────────────────────────────────────────────
static std::string generate(
    const Config& cfg,
    const ModelWeights& W,
    const Tokenizer& tok,
    const std::string& prompt,
    int max_new_tokens = 200,
    float temperature  = 1.0f)
{
    int T = cfg.block_size, V = cfg.vocab_size;

    // Seed context with prompt (right-aligned; truncate keeping suffix if longer than T).
    // This ensures ctx[T-1] always contains the last relevant token (last prompt token
    // when no new tokens have been generated, or the most recently generated token thereafter),
    // which is the position we sample from below.
    std::vector<int> ctx(T, 0);
    std::vector<int> enc = tok.encode(prompt);
    int enc_size = static_cast<int>(enc.size());
    int copy_count = std::min(enc_size, T);
    for (int i = 0; i < copy_count; i++) {
        ctx[T - copy_count + i] = enc[enc_size - copy_count + i];
    }

    std::string out = prompt;
    ForwardCache cache;

    for (int step = 0; step < max_new_tokens; step++) {
        forward(cfg, W, ctx, cache, false);

        // Last token's logits → sample
        // With right-aligned prompting, ctx[T-1] always contains the last relevant token:
        //   - At step 0: the last prompt token (correct starting point for generation)
        //   - At step s (s > 0): the most recently generated token (correct for continuation)
        // We therefore always sample from position T-1.
        int last = T - 1;

        std::vector<float> logits(V);
        std::copy(&cache.logits[last*V], &cache.logits[last*V+V], logits.begin());

        int next_id;
        if (temperature <= 0.0f) {
            // Greedy
            next_id = (int)(std::max_element(logits.begin(), logits.end()) - logits.begin());
        } else {
            // Temperature-scaled softmax sampling
            for (auto& l : logits) l /= temperature;
            softmax(logits.data(), V);
            std::discrete_distribution<int> dist(logits.begin(), logits.end());
            next_id = dist(rng);
        }

        out += tok.decode(next_id);

        // Shift context window left and append new token
        std::rotate(ctx.begin(), ctx.begin()+1, ctx.end());
        ctx[T-1] = next_id;
    }
    return out;
}

// ─────────────────────────────────────────────────────────────────────────────
// 19.  Training loop
// ─────────────────────────────────────────────────────────────────────────────
static bool train(
    const std::string& data_path,
    const std::string& model_path = "model.bin",
    int   max_iters   = 5000,
    int   eval_every  = 100,
    float lr          = 3e-3f,
    int   n_embd      = 128,
    int   n_head      = 8,
    int   n_layer     = 6,
    int   block_size  = 128,
    bool  resume_from_state = false,
    int   total_iters_override = 0)
{
    std::ifstream f(data_path);
    if (!f) { std::fprintf(stderr, "Cannot open %s\n", data_path.c_str()); return false; }
    std::string text((std::istreambuf_iterator<char>(f)), {});
    std::printf("[data] loaded %zu characters\n", text.size());

    Tokenizer tok;
    tok.build(text);
    std::printf("[tok]  vocab_size = %d\n", tok.vocab_size());
    tok.save("vocab.bin");
    std::vector<int> data = tok.encode(text);

    Config cfg;
    cfg.vocab_size = tok.vocab_size();
    cfg.n_embd     = n_embd;
    cfg.n_head     = n_head;
    cfg.n_layer    = n_layer;
    cfg.block_size = block_size;
    ModelWeights W;
    W.init(cfg);

    Adam opt;
    opt.lr = lr;
    opt.lr_total_iters = total_iters_override > 0 ? total_iters_override : max_iters;
    ForwardCache cache;
    std::mt19937 data_rng(0);
    const std::uint64_t corpus_tokens = static_cast<std::uint64_t>(data.size());

    if (resume_from_state) {
        // Weights load first: the checkpoint's Config becomes authoritative for
        // the resumed run, and optimizer-state tensors are dimension-checked
        // against it by load_training_state().
        if (!load_weights(W, model_path)) {
            std::fprintf(stderr, "resume: cannot load model weights from %s\n", model_path.c_str());
            return false;
        }
        cfg = W.cfg;
        const std::string state_path = model_path + ".state";
        std::uint64_t stored_tokens = 0;
        if (!load_training_state(W, opt, data_rng, stored_tokens, state_path)) {
            std::fprintf(stderr, "resume: cannot load training state from %s\n", state_path.c_str());
            return false;
        }
        if (stored_tokens != corpus_tokens) {
            std::fprintf(stderr, "resume: training state was saved with a %llu-token corpus, but the live corpus has %llu tokens; refusing to resume with a different sampler range\n",
                         static_cast<unsigned long long>(stored_tokens),
                         static_cast<unsigned long long>(corpus_tokens));
            return false;
        }
        if (total_iters_override > 0) {
            opt.lr_total_iters = total_iters_override;
            std::printf("[resume] LR schedule horizon overridden to %d iterations\n", opt.lr_total_iters);
        }
        std::printf("[resume] step=%d lr=%g lr_total_iters=%d\n", opt.step, opt.lr, opt.lr_total_iters);
    }
    std::printf("[model] parameters = %d (~%.2fM)\n", W.count_params(),
                W.count_params() / 1e6f);

    int T = cfg.block_size;

    if (static_cast<int>(data.size()) < T + 2) {
        std::fprintf(stderr, "train: corpus too short (%zu tokens after tokenisation); need at least %d tokens for block_size=%d\n",
                     data.size(), T + 2, T);
        return false;
    }

    for (int iter = 0; iter < max_iters; iter++) {
        std::uniform_int_distribution<int> pos_dist(0, static_cast<int>(data.size()) - T - 2);
        int start_pos = pos_dist(data_rng);
        std::vector<int> seq(data.begin() + start_pos, data.begin() + start_pos + T + 1);
        float loss = forward(cfg, W, seq, cache, true);
        if (!std::isfinite(loss)) {
            std::fprintf(stderr, "NaN loss at iter %d\n", iter + 1);
            return false;
        }
        backward(cfg, W, seq, cache);
        bool grad_nan = false;
        auto check_grad = [&](const std::vector<float>& g, const char* name) {
            for (float x : g) {
                if (!std::isfinite(x)) { grad_nan = true; std::fprintf(stderr, "NaN in %s at iter %d\n", name, iter + 1); return; }
            }
        };
        check_grad(W.dwte, "dwte");
        check_grad(W.dwpe, "dwpe");
        for (auto& b : W.blocks) {
            check_grad(b.attn.dwq, "dwq");
            check_grad(b.attn.dwk, "dwk");
            check_grad(b.attn.dwv, "dwv");
            check_grad(b.attn.dwo, "dwo");
            check_grad(b.mlp.dfc1, "dfc1");
            check_grad(b.mlp.dfc2, "dfc2");
            if (grad_nan) break;
        }
        if (grad_nan) return false;
        opt.update_all(W);
        if (opt.step % eval_every == 0) {
            std::printf("[iter %5d] loss = %.4f\n", opt.step, loss);
        }
    }

    if (!save_weights(W, model_path)) return false;
    const std::string state_path = model_path + ".state";
    if (!save_training_state(W, opt, data_rng, corpus_tokens, state_path)) return false;
    std::printf("[done] model saved to %s (training state: %s)\n", model_path.c_str(), state_path.c_str());
    return true;
}

// ─────────────────────────────────────────────────────────────────────────────
// 20.  Main
// ─────────────────────────────────────────────────────────────────────────────
int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::printf(
            "Usage:\n"
            "  %s train  <data.txt> [model.bin] [iters] [lr] [n_embd] [n_head] [n_layer] [block_size] [total_iters]\n"
            "  %s resume <data.txt> <model.bin> [extra_iters] [total_iters]\n"
            "  %s gen    <model.bin> <vocab.bin> \"<seed>\" [temperature] [max_new]\n"
            "  train  writes <model.bin> plus full optimizer/sampler state in <model.bin>.state;\n"
            "  resume continues training from that state. Persisted optimizer hyperparameters\n"
            "  (lr, schedule horizon, beta1/beta2, eps, step) are authoritative on resume;\n"
            "  total_iters may extend the schedule horizon.\n",
            argv[0], argv[0], argv[0]);
        return 1;
    }

    std::string mode = argv[1];

    if (mode == "train") {
        if (argc < 3) { std::fprintf(stderr, "train: need <data.txt>\n"); return 1; }
        std::string data_path  = argv[2];
        std::string model_path = (argc > 3) ? argv[3] : "model.bin";
        int   iters = (argc > 4) ? std::stoi(argv[4]) : 5000;
        float lr    = (argc > 5) ? std::stof(argv[5]) : 3e-3f;
        int   n_embd  = (argc > 6) ? std::stoi(argv[6]) : 128;
        int   n_head  = (argc > 7) ? std::stoi(argv[7]) : 8;
        int   n_layer = (argc > 8) ? std::stoi(argv[8]) : 6;
        int   block_size = (argc > 9) ? std::stoi(argv[9]) : 128;
        int   total_iters = (argc > 10) ? std::stoi(argv[10]) : 0;
        if (!train(data_path, model_path, iters, 100, lr, n_embd, n_head, n_layer, block_size, false, total_iters)) return 1;

    } else if (mode == "resume") {
        if (argc < 4) { std::fprintf(stderr, "resume: need <data.txt> <model.bin> [extra_iters] [total_iters]\n"); return 1; }
        std::string data_path  = argv[2];
        std::string model_path = argv[3];
        int extra_iters = (argc > 4) ? std::stoi(argv[4]) : 1000;
        int total_iters = (argc > 5) ? std::stoi(argv[5]) : 0;
        // Persisted optimizer hyperparameters (lr, schedule horizon, beta1/beta2,
        // eps, step) and the checkpoint configuration are authoritative on
        // resume; the train-mode CLI lr/dimension arguments do not apply here.
        if (!train(data_path, model_path, extra_iters, 100, 0.0f, 128, 8, 6, 128, true, total_iters)) return 1;

    } else if (mode == "gen") {
        if (argc < 5) { std::fprintf(stderr, "gen: need <model.bin> <vocab.bin> \"<seed>\"\n"); return 1; }
        std::string model_path = argv[2];
        std::string vocab_path = argv[3];
        std::string seed       = argv[4];
        float temperature      = (argc > 5) ? std::stof(argv[5]) : 0.8f;
        int   max_new          = (argc > 6) ? std::stoi(argv[6]) : 300;

        ModelWeights W;
        if (!load_weights(W, model_path)) { std::fprintf(stderr, "Cannot load %s\n", model_path.c_str()); return 1; }

        Tokenizer tok;
        if (!tok.load(vocab_path)) { std::fprintf(stderr, "Cannot load %s\n", vocab_path.c_str()); return 1; }

        std::string out = generate(W.cfg, W, tok, seed, max_new, temperature);
        std::printf("%s\n", out.c_str());

    } else {
        std::fprintf(stderr, "Unknown mode: %s\n", mode.c_str());
        return 1;
    }

    return 0;
}
