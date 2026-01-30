\
    #include "gpt2.h"
    #include <stdlib.h>
    #include <stdio.h>
    #include <string.h>
    #include <math.h>

    static float *alloc_f(size_t n) {
        float *p = (float*)malloc(sizeof(float)*n);
        if (!p) { fprintf(stderr, "alloc failed\n"); exit(1); }
        return p;
    }

    gpt2_model *gpt2_model_new(int n_vocab, int n_ctx, int n_embd, int n_layer, int n_head) {
        gpt2_model *m = (gpt2_model*)calloc(1, sizeof(gpt2_model));
        m->n_vocab = n_vocab; m->n_ctx = n_ctx; m->n_embd = n_embd; m->n_layer = n_layer; m->n_head = n_head;
        m->w_emb = alloc_f((size_t)n_vocab * n_embd);
        m->w_ln_f_gamma = alloc_f(n_embd);
        m->w_ln_f_beta  = alloc_f(n_embd);
        m->w_dense_w = alloc_f((size_t)n_embd * n_embd);
        m->w_dense_b = alloc_f(n_embd);
        return m;
    }

    void gpt2_model_free(gpt2_model *m) {
        if (!m) return;
        free(m->w_emb); free(m->w_ln_f_gamma); free(m->w_ln_f_beta);
        free(m->w_dense_w); free(m->w_dense_b); free(m);
    }

    static void matmul_rxc_c(const float *A, const float *x, float *y, int r, int c) {
        for (int i = 0; i < r; ++i) {
            float s = 0.0f;
            const float *Ai = A + (size_t)i * c;
            for (int j = 0; j < c; ++j) s += Ai[j] * x[j];
            y[i] = s;
        }
    }

    static void add_bias(float *vec, const float *bias, int n) {
        for (int i = 0; i < n; ++i) vec[i] += bias[i];
    }

    static void gelu_inplace(float *x, int n) {
        for (int i = 0; i < n; ++i) {
            float v = x[i];
            x[i] = 0.5f * v * (1.0f + tanhf(0.79788456f * (v + 0.044715f * v * v * v)));
        }
    }

    static void layernorm(const float *x, float *out, const float *gamma, const float *beta, int n) {
        float mean = 0.0f;
        for (int i = 0; i < n; ++i) mean += x[i];
        mean /= n;
        float var = 0.0f;
        for (int i = 0; i < n; ++i) { float d = x[i] - mean; var += d * d; }
        var /= n;
        float denom = 1.0f / sqrtf(var + 1e-5f);
        for (int i = 0; i < n; ++i) out[i] = (x[i] - mean) * denom * gamma[i] + beta[i];
    }

    static void tiny_block(gpt2_model *m, const float *emb, float *out) {
        int D = m->n_embd;
        float *ln = (float*)malloc(sizeof(float)*D);
        layernorm(emb, ln, m->w_ln_f_gamma, m->w_ln_f_beta, D);
        float *tmp = (float*)malloc(sizeof(float)*D);
        matmul_rxc_c(m->w_dense_w, ln, tmp, D, D);
        add_bias(tmp, m->w_dense_b, D);
        gelu_inplace(tmp, D);
        float *tmp2 = (float*)malloc(sizeof(float)*D);
        matmul_rxc_c(m->w_dense_w, tmp, tmp2, D, D);
        add_bias(tmp2, m->w_dense_b, D);
        for (int i = 0; i < D; ++i) out[i] = emb[i] + tmp2[i];
        free(ln); free(tmp); free(tmp2);
    }

    int gpt2_model_load_weights(gpt2_model *m, const char *filename) {
        FILE *f = fopen(filename, "rb");
        if (!f) { perror("open weights"); return -1; }
        size_t n;
        n = (size_t)m->n_vocab * m->n_embd;
        if (fread(m->w_emb, sizeof(float), n, f) != n) { fclose(f); return -1; }
        n = (size_t)m->n_embd;
        if (fread(m->w_ln_f_gamma, sizeof(float), n, f) != n) { fclose(f); return -1; }
        if (fread(m->w_ln_f_beta, sizeof(float), n, f) != n) { fclose(f); return -1; }
        n = (size_t)m->n_embd * m->n_embd;
        if (fread(m->w_dense_w, sizeof(float), n, f) != n) { fclose(f); return -1; }
        n = (size_t)m->n_embd;
        if (fread(m->w_dense_b, sizeof(float), n, f) != n) { fclose(f); return -1; }
        fclose(f);
        return 0;
    }

    int gpt2_model_forward(gpt2_model *m, const int *tokens, int t, float *out_logits) {
        if (t <= 0) return -1;
        int D = m->n_embd;
        float *hidden = (float*)calloc(D, sizeof(float));
        for (int i = 0; i < t; ++i) {
            int id = tokens[i];
            if (id < 0 || id >= m->n_vocab) { free(hidden); return -1; }
            float *emb = m->w_emb + (size_t)id * D;
            for (int j = 0; j < D; ++j) hidden[j] += emb[j];
        }
        for (int j = 0; j < D; ++j) hidden[j] /= t;
        float *hidden2 = (float*)malloc(sizeof(float)*D);
        tiny_block(m, hidden, hidden2);
        for (int v = 0; v < m->n_vocab; ++v) {
            float s = 0.0f;
            float *ve = m->w_emb + (size_t)v * D;
            for (int j = 0; j < D; ++j) s += ve[j] * hidden2[j];
            out_logits[v] = s;
        }
        free(hidden); free(hidden2);
        return 0;
    }

    static void softmax_inplace(float *x, int n) {
        float m = x[0];
        for (int i = 1; i < n; ++i) if (x[i] > m) m = x[i];
        double sum = 0.0;
        for (int i = 0; i < n; ++i) { sum += exp((double)(x[i] - m)); }
        for (int i = 0; i < n; ++i) x[i] = (float)(exp((double)(x[i] - m)) / sum);
    }

    int gpt2_generate(gpt2_model *m, int *context_tokens, int context_len, int max_new_tokens, int *out_tokens) {
        int n = 0;
        int *buf = (int*)malloc(sizeof(int) * (context_len + max_new_tokens));
        memcpy(buf, context_tokens, sizeof(int) * context_len);
        int cur_len = context_len;
        float *logits = (float*)malloc(sizeof(float) * m->n_vocab);
        for (int step = 0; step < max_new_tokens; ++step) {
            if (gpt2_model_forward(m, buf, cur_len, logits) != 0) { free(buf); free(logits); return -1; }
            softmax_inplace(logits, m->n_vocab);
            int best = 0; float bestv = logits[0];
            for (int i = 1; i < m->n_vocab; ++i) if (logits[i] > bestv) { bestv = logits[i]; best = i; }
            out_tokens[n++] = best;
            buf[cur_len++] = best;
            if (cur_len > m->n_ctx) {
                memmove(buf, buf + (cur_len - m->n_ctx), sizeof(int) * m->n_ctx);
                cur_len = m->n_ctx;
            }
        }
        free(buf); free(logits);
        return n;
    }
