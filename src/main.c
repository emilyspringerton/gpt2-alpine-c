\
    #include "gpt2.h"
    #include <stdio.h>
    #include <stdlib.h>

    int main(int argc, char **argv) {
        if (argc < 2) {
            printf("Usage: %s <weights.bin> [seed_token_ids...]\n", argv[0]);
            return 1;
        }
        const char *weights = argv[1];
        const int N_VOCAB = 50257;
        const int N_CTX = 1024;
        const int N_EMBD = 768;
        const int N_LAYER = 12;
        const int N_HEAD = 12;
        gpt2_model *m = gpt2_model_new(N_VOCAB, N_CTX, N_EMBD, N_LAYER, N_HEAD);
        if (gpt2_model_load_weights(m, weights) != 0) {
            fprintf(stderr, "Failed to load weights from %s\n", weights);
            gpt2_model_free(m);
            return 2;
        }
        int context[1024]; int context_len = 0;
        for (int i = 2; i < argc && context_len < 1024; ++i) context[context_len++] = atoi(argv[i]);
        if (context_len == 0) context[context_len++] = 50256 % N_VOCAB;
        int out[512];
        int generated = gpt2_generate(m, context, context_len, 16, out);
        if (generated < 0) { fprintf(stderr, "Generation failed\n"); gpt2_model_free(m); return 3; }
        printf("Generated %d tokens:\\n", generated);
        for (int i = 0; i < generated; ++i) printf("%d ", out[i]);
        printf("\\n");
        gpt2_model_free(m);
        return 0;
    }
