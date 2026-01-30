\
    #ifndef GPT2_H
    #define GPT2_H

    #include <stddef.h>
    #include <stdint.h>

    typedef struct {
        int n_vocab;
        int n_ctx;
        int n_embd;
        int n_layer;
        int n_head;
        float *w_emb;
        float *w_ln_f_gamma;
        float *w_ln_f_beta;
        float *w_dense_w;
        float *w_dense_b;
    } gpt2_model;

    gpt2_model *gpt2_model_new(int n_vocab, int n_ctx, int n_embd, int n_layer, int n_head);
    void gpt2_model_free(gpt2_model *m);
    int gpt2_model_load_weights(gpt2_model *m, const char *filename);
    int gpt2_model_forward(gpt2_model *m, const int *tokens, int t, float *out_logits);
    int gpt2_generate(gpt2_model *m, int *context_tokens, int context_len, int max_new_tokens, int *out_tokens);

    #endif // GPT2_H
