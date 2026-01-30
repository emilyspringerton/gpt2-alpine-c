#include "tokenizer.h"

TokenStr decoder[VOCAB_SIZE];

int num_merges;
Merge *merges;

void tokenizer_load(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "Failed to open %s\n", path);
        exit(1);
    }
    
    int vs;
    fread(&vs, sizeof(int), 1, f);
    if (vs != VOCAB_SIZE) {
        fprintf(stderr, "Vocab size mismatch\n");
        exit(1);
    }
    
    for (int i = 0; i < VOCAB_SIZE; i++) {
        fread(&decoder[i].len, sizeof(int), 1, f);
        decoder[i].bytes = malloc(decoder[i].len);
        fread(decoder[i].bytes, 1, decoder[i].len, f);
    }
    
    fread(&num_merges, sizeof(int), 1, f);
    merges = malloc(num_merges * sizeof(Merge));
    fread(merges, sizeof(Merge), num_merges, f);
    
    fclose(f);
}

void tokenizer_free() {
    for (int i = 0; i < VOCAB_SIZE; i++) {
        free(decoder[i].bytes);
    }
    free(merges);
}

void gpt2_encode(const char *text, int *ids, int *len) {
    int text_len = strlen(text);
    int *tokens = malloc(text_len * sizeof(int));
    int num_tokens = 0;
    for (int i = 0; i < text_len; ) {
        if ((unsigned char)text[i] < 128) {
            tokens[num_tokens++] = (unsigned char)text[i];
            i++;
        } else {
            // Simple UTF-8 handling; for GPT2 BPE, we treat as bytes
            tokens[num_tokens++] = (unsigned char)text[i];
            i++;
        }
    }
    
    while (1) {
        int min_rank = num_merges + 1;
        int selected_merge = -1;
        int merge_pos = -1;
        for (int i = 0; i < num_tokens - 1; i++) {
            int p1 = tokens[i];
            int p2 = tokens[i + 1];
            for (int m = 0; m < num_merges; m++) {
                if (merges[m].first == p1 && merges[m].second == p2) {
                    if (m < min_rank) {
                        min_rank = m;
                        selected_merge = m;
                        merge_pos = i;
                    }
                    break;
                }
            }
        }
        if (selected_merge == -1) break;
        
        tokens[merge_pos] = merges[selected_merge].replacement;
        for (int i = merge_pos + 1; i < num_tokens - 1; i++) {
            tokens[i] = tokens[i + 1];
        }
        num_tokens--;
    }
    
    memcpy(ids, tokens, num_tokens * sizeof(int));
    *len = num_tokens;
    free(tokens);
}

char *gpt2_decode(const int *ids, int len) {
    int total_len = 0;
    for (int i = 0; i < len; i++) {
        total_len += decoder[ids[i]].len;
    }
    char *result = malloc(total_len + 1);
    int pos = 0;
    for (int i = 0; i < len; i++) {
        memcpy(result + pos, decoder[ids[i]].bytes, decoder[ids[i]].len);
        pos += decoder[ids[i]].len;
    }
    result[total_len] = '\0';
    return result;
}
