#ifndef TOKENIZER_H
#define TOKENIZER_H

#include "gpt2.h"  // For VOCAB_SIZE, EOS_TOKEN

typedef struct {
    int len;
    char *bytes;
} TokenStr;

extern TokenStr decoder[VOCAB_SIZE];

typedef struct {
    int first;
    int second;
    int replacement;
} Merge;

extern int num_merges;
extern Merge *merges;

void tokenizer_load(const char *path);
void tokenizer_free();

void gpt2_encode(const char *text, int *ids, int *len);
char *gpt2_decode(const int *ids, int len);

#endif
