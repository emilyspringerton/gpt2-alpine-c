#include "gpt2.h"

/* Thin wrapper: the real loading logic lives in gpt2.c. */
int load_weights_into_model(gpt2_model *m, const char *path)
{
    return gpt2_model_load_weights(m, path);
}
