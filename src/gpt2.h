#ifndef GPT2_H
#define GPT2_H

#include <stddef.h>
#include <stdint.h>

/*
 * Full GPT-2 weight struct (parity with OpenAI gpt2-small: 12L/12H/768D).
 *
 * Weight layout per layer (matching convert_checkpoint.py write order):
 *   ln_1_gamma, ln_1_beta          (D each)
 *   c_attn_w (D × 3D), c_attn_b (3D)
 *   c_proj_w (D × D),  c_proj_b (D)
 *   ln_2_gamma, ln_2_beta          (D each)
 *   mlp_fc_w (D × 4D), mlp_fc_b (4D)
 *   mlp_proj_w (4D × D), mlp_proj_b (D)
 *
 * Global:
 *   wte (V × D), wpe (T × D)
 *   ln_f_gamma, ln_f_beta          (D each)
 */

typedef struct {
    /* layer norms */
    float *ln_1_gamma;   /* D */
    float *ln_1_beta;    /* D */
    float *ln_2_gamma;   /* D */
    float *ln_2_beta;    /* D */
    /* attention */
    float *c_attn_w;     /* D × 3D  (row-major: out=3D, in=D) */
    float *c_attn_b;     /* 3D */
    float *c_proj_w;     /* D × D   (out=D, in=D) */
    float *c_proj_b;     /* D */
    /* MLP */
    float *mlp_fc_w;     /* D × 4D  (out=4D, in=D) */
    float *mlp_fc_b;     /* 4D */
    float *mlp_proj_w;   /* 4D × D  (out=D, in=4D) */
    float *mlp_proj_b;   /* D */
} GPT2Layer;

typedef struct {
    int n_vocab;   /* V: 50257 */
    int n_ctx;     /* T: 1024  */
    int n_embd;    /* D: 768   */
    int n_layer;   /* L: 12    */
    int n_head;    /* H: 12    */
    /* global embeddings */
    float *wte;    /* V × D */
    float *wpe;    /* T × D */
    /* per-layer */
    GPT2Layer *layers; /* n_layer */
    /* final layer norm */
    float *ln_f_gamma; /* D */
    float *ln_f_beta;  /* D */
} gpt2_model;

/*
 * gpt2_entropy holds per-token entropy from the last generate call
 * (allocated by caller: n_tokens floats).
 */
typedef struct {
    float *per_token; /* entropy in nats for each generated token */
    int    n;
} gpt2_entropy;

gpt2_model *gpt2_model_new(int n_vocab, int n_ctx, int n_embd, int n_layer, int n_head);
void        gpt2_model_free(gpt2_model *m);

/*
 * Load from binary produced by convert_checkpoint.py.
 * Format: all weights written contiguously in float32, in the order:
 *   wte, wpe,
 *   [for each layer]: ln_1_gamma, ln_1_beta,
 *                     c_attn_w, c_attn_b, c_proj_w, c_proj_b,
 *                     ln_2_gamma, ln_2_beta,
 *                     mlp_fc_w, mlp_fc_b, mlp_proj_w, mlp_proj_b,
 *   ln_f_gamma, ln_f_beta
 * Returns 0 on success, -1 on error.
 */
int gpt2_model_load_weights(gpt2_model *m, const char *filename);

/*
 * Forward pass: compute logits for next token given context tokens[0..t-1].
 * out_logits must be preallocated to n_vocab floats.
 * Returns 0 on success.
 */
int gpt2_model_forward(gpt2_model *m, const int *tokens, int t, float *out_logits);

/*
 * Autoregressive generation. Greedy (argmax) by default.
 * If entropy != NULL, fills per-token entropy (caller allocates max_new_tokens floats).
 * Returns number of tokens generated, or -1 on error.
 */
int gpt2_generate(gpt2_model *m, int *context_tokens, int context_len,
                  int max_new_tokens, int *out_tokens, gpt2_entropy *entropy);

#endif /* GPT2_H */
