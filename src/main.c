#include "gpt2.h"
#include "tokenizer.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void usage(const char *prog)
{
    fprintf(stderr,
        "Usage: %s <weights.bin> [options] [seed_token_ids...]\n"
        "Options:\n"
        "  --prompt TEXT    encode TEXT as context and decode output as text\n"
        "  --tokens N       generate N tokens (default 64)\n"
        "  --entropy        print per-token entropy (nats) after generation\n"
        "  --entropy-stats  print mean/max entropy and exit (no text output)\n",
        prog);
}

int main(int argc, char **argv)
{
    if (argc < 2) { usage(argv[0]); return 1; }

    const char *weights_path = argv[1];
    int max_tokens    = 64;
    int do_entropy    = 0;
    int only_stats    = 0;
    const char *prompt = NULL;

    int context[1024];
    int context_len = 0;

    for (int i = 2; i < argc; ++i) {
        if (strcmp(argv[i], "--tokens") == 0 && i + 1 < argc) {
            max_tokens = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--prompt") == 0 && i + 1 < argc) {
            prompt = argv[++i];
        } else if (strcmp(argv[i], "--entropy") == 0) {
            do_entropy = 1;
        } else if (strcmp(argv[i], "--entropy-stats") == 0) {
            do_entropy = 1;
            only_stats = 1;
        } else {
            /* treat as token id */
            int tok = atoi(argv[i]);
            if (context_len < 1024) context[context_len++] = tok;
        }
    }

    /* Load tokenizer when using --prompt (weights dir inferred from weights path) */
    int use_text = (prompt != NULL);
    if (use_text) {
        /* tokenizer.json lives next to the weights file */
        char tok_path[1024];
        const char *slash = strrchr(weights_path, '/');
        if (slash) {
            int dir_len = (int)(slash - weights_path);
            snprintf(tok_path, sizeof(tok_path), "%.*s/tokenizer.bin", dir_len, weights_path);
        } else {
            snprintf(tok_path, sizeof(tok_path), "weights/tokenizer.bin");
        }
        tokenizer_load(tok_path);
        gpt2_encode(prompt, context, &context_len);
        if (context_len == 0) {
            context[context_len++] = 50256; /* fallback: endoftext */
        }
    }

    if (context_len == 0) {
        /* GPT-2 endoftext token as default seed */
        context[context_len++] = 50256;
    }

    const int N_VOCAB = 50257, N_CTX = 1024, N_EMBD = 768;
    const int N_LAYER = 12,    N_HEAD = 12;

    gpt2_model *m = gpt2_model_new(N_VOCAB, N_CTX, N_EMBD, N_LAYER, N_HEAD);
    if (gpt2_model_load_weights(m, weights_path) != 0) {
        fprintf(stderr, "Failed to load weights from %s\n", weights_path);
        gpt2_model_free(m);
        return 2;
    }

    int *out_tokens = (int *)malloc(sizeof(int) * max_tokens);
    float *ent_buf  = do_entropy ? (float *)malloc(sizeof(float) * max_tokens) : NULL;

    gpt2_entropy ent = { ent_buf, 0 };
    int n = gpt2_generate(m, context, context_len, max_tokens, out_tokens,
                          do_entropy ? &ent : NULL);
    if (n < 0) {
        fprintf(stderr, "Generation failed\n");
        gpt2_model_free(m); free(out_tokens); free(ent_buf);
        return 3;
    }

    if (!only_stats) {
        if (use_text) {
            char *text = gpt2_decode(out_tokens, n);
            if (text) { printf("%s\n", text); free(text); }
        } else {
            printf("Generated %d tokens:\n", n);
            for (int i = 0; i < n; ++i) printf("%d ", out_tokens[i]);
            printf("\n");
        }
    }

    if (do_entropy && ent_buf) {
        float sum = 0.0f, mx = ent_buf[0];
        for (int i = 0; i < n; ++i) {
            sum += ent_buf[i];
            if (ent_buf[i] > mx) mx = ent_buf[i];
        }
        float mean = sum / (float)n;

        if (only_stats) {
            printf("entropy_mean_nats=%.4f entropy_max_nats=%.4f tokens=%d\n",
                   mean, mx, n);
        } else {
            printf("\nEntropy (nats):\n");
            for (int i = 0; i < n; ++i)
                printf("  token[%d]=%d  H=%.4f\n", i, out_tokens[i], ent_buf[i]);
            printf("mean=%.4f  max=%.4f\n", mean, mx);
        }
    }

    gpt2_model_free(m);
    if (use_text) tokenizer_free();
    free(out_tokens);
    free(ent_buf);
    return 0;
}
