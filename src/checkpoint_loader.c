\
    #include "gpt2.h"
    #include <stdio.h>

    int load_weights_into_model(gpt2_model *m, const char *path) {
        return gpt2_model_load_weights(m, path);
    }
