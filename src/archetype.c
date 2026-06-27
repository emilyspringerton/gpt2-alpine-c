#include "archetype.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

/*
 * archetype_classify — score text_tokens against each archetype prefix.
 *
 * For archetype A with prefix tokens P = [p0..pm-1]:
 *   context = [p0..pm-1, t0..tn-1]
 *   For each text token t_i (i = 0..n-1):
 *     ctx_len = m + i
 *     logits  = gpt2_forward(context[0..ctx_len-1])
 *     nll_i   = -log(softmax(logits)[t_i])   (in nats)
 *   perplexity_nats = mean(nll_i)
 */
int archetype_classify(gpt2_model *m,
                       const int *text_tokens, int text_len,
                       ArchetypeScore scores[ARCH_COUNT])
{
    if (!m || !text_tokens || text_len <= 0) return -1;

    /* max context: n_ctx tokens */
    int max_ctx = m->n_ctx;
    int *ctx = (int *)malloc(sizeof(int) * max_ctx);
    float *logits = (float *)malloc(sizeof(float) * m->n_vocab);
    if (!ctx || !logits) { free(ctx); free(logits); return -1; }

    for (int ai = 0; ai < ARCH_COUNT; ai++) {
        const ArchetypeProfile *arch = &ARCHETYPES[ai];
        scores[ai].id = arch->id;

        /* encode prefix */
        int prefix_buf[128];
        int prefix_len = 0;
        gpt2_encode(arch->prefix, prefix_buf, &prefix_len);

        /* clamp: leave room for at least one text token */
        if (prefix_len >= max_ctx - 1) prefix_len = max_ctx - 2;

        /* build context: prefix + text */
        for (int i = 0; i < prefix_len; i++) ctx[i] = prefix_buf[i];
        int avail = max_ctx - prefix_len;
        int use   = text_len < avail ? text_len : avail;
        for (int i = 0; i < use; i++) ctx[prefix_len + i] = text_tokens[i];

        /* accumulate NLL over text tokens */
        double nll_sum = 0.0;
        int counted = 0;
        for (int ti = 0; ti < use; ti++) {
            int ctx_len = prefix_len + ti; /* predict text_tokens[ti] from prefix+prev */
            if (ctx_len < 1) { ctx_len = 1; }
            if (ctx_len > max_ctx) break;

            if (gpt2_model_forward(m, ctx, ctx_len, logits) != 0) continue;

            /* log-sum-exp for numerical stability */
            int target = text_tokens[ti];
            float max_l = logits[0];
            for (int v = 1; v < m->n_vocab; v++)
                if (logits[v] > max_l) max_l = logits[v];

            double lse = 0.0;
            for (int v = 0; v < m->n_vocab; v++)
                lse += exp((double)(logits[v] - max_l));
            double log_z = (double)max_l + log(lse);
            double log_p = (double)logits[target] - log_z;

            nll_sum += -log_p;
            counted++;
        }

        scores[ai].perplexity_nats = (counted > 0)
            ? (float)(nll_sum / counted)
            : 1e9f;
    }

    free(ctx);
    free(logits);
    return 0;
}

static int score_cmp(const void *a, const void *b)
{
    float da = ((const ArchetypeScore *)a)->perplexity_nats;
    float db = ((const ArchetypeScore *)b)->perplexity_nats;
    return (da < db) ? -1 : (da > db) ? 1 : 0;
}

void archetype_print_scores(const ArchetypeScore scores[ARCH_COUNT])
{
    /* sort a local copy */
    ArchetypeScore sorted[ARCH_COUNT];
    memcpy(sorted, scores, sizeof(sorted));
    qsort(sorted, ARCH_COUNT, sizeof(ArchetypeScore), score_cmp);

    printf("archetype classification (lower = better match):\n");
    for (int i = 0; i < ARCH_COUNT; i++) {
        const ArchetypeProfile *p = &ARCHETYPES[sorted[i].id];
        printf("  [%d] %-14s  %.4f nats  (%s / %.2f Hz)\n",
               i + 1, p->display, sorted[i].perplexity_nats, p->mode, p->hz);
    }
    printf("best match: %s\n", ARCHETYPES[sorted[0].id].display);
}
