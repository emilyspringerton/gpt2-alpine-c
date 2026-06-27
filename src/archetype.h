#ifndef ARCHETYPE_H
#define ARCHETYPE_H

/*
 * archetype.h — TYLER universe archetype engine for GPT-2 inference.
 *
 * Archetypes are character-voice profiles that steer generation by prepending
 * a structured prompt prefix encoding natal data, frequency, and operating mode.
 *
 * Usage:
 *   archetype_encode_prefix(ARCH_TYLER, ctx, &ctx_len);  // steer context
 *   ArchetypeScore scores[ARCH_COUNT];
 *   archetype_classify(m, tokens, n, scores);             // perplexity scoring
 */

#include "gpt2.h"
#include "tokenizer.h"

/* Goetia frequencies referenced in TYLER archive */
#define HZ_STOLAS     7.83f    /* deep time / Tyler hum */
#define HZ_BELETH     3.33f    /* love / emotional detonation */
#define HZ_AMON       1.618f   /* reconciliation of opposites */
#define HZ_VASSAGO    11.11f   /* soft precognition */
#define HZ_CARRIER    28.7f    /* carrier wave / natal Cancer ASC frequency */

typedef enum {
    ARCH_TYLER      = 0,  /* Tyler — carrier wave, 9 centuries, Aquarius/Cancer ASC */
    ARCH_EMILY_S    = 1,  /* Emily Springerton — builder mode, same natal as Tyler */
    ARCH_EMILY_OS   = 2,  /* Emily OS — Layer 4 substrate, predates everything */
    ARCH_CAMERA_OP  = 3,  /* Camera Op — second practitioner, Capricorn ASC */
    ARCH_VALENTINA  = 4,  /* Valentina Alcântara — library detection, São Paulo */
    ARCH_JIANGSHI   = 5,  /* Jiangshi — frequency monitor, former binding operative */
    ARCH_COUNT      = 6
} ArchetypeID;

typedef struct {
    ArchetypeID id;
    const char *name;           /* identifier for CLI --archetype NAME */
    const char *display;        /* human display name */
    float hz;                   /* primary frequency in Hz */
    const char *natal_sun;      /* natal sun sign */
    const char *natal_asc;      /* natal ascendant */
    const char *layer;          /* TYLER four-layer architecture layer */
    const char *mode;           /* operating mode descriptor */
    const char *prefix;         /* prompt prefix for generation steering */
} ArchetypeProfile;

/* Archetype table — indexed by ArchetypeID */
static const ArchetypeProfile ARCHETYPES[ARCH_COUNT] = {
    [ARCH_TYLER] = {
        .id       = ARCH_TYLER,
        .name     = "TYLER",
        .display  = "Tyler",
        .hz       = HZ_CARRIER,
        .natal_sun = "Aquarius 15°",
        .natal_asc = "Cancer 29°",
        .layer    = "Layer 3",
        .mode     = "carrier-wave / observer",
        .prefix   =
            "TYLER [Cancer ASC 29° / Aquarius Sun 15° / 28.7 Hz / carrier wave]:\n"
    },
    [ARCH_EMILY_S] = {
        .id       = ARCH_EMILY_S,
        .name     = "EMILY_S",
        .display  = "Emily Springerton",
        .hz       = HZ_BELETH,
        .natal_sun = "Aquarius 15°",
        .natal_asc = "Cancer 29°",
        .layer    = "Layer 3",
        .mode     = "builder / logistics",
        .prefix   =
            "EMILY SPRINGERTON [Cancer ASC 29° / Aquarius Sun 15° / 3.33 Hz / builder mode]:\n"
    },
    [ARCH_EMILY_OS] = {
        .id       = ARCH_EMILY_OS,
        .name     = "EMILY_OS",
        .display  = "Emily OS",
        .hz       = 0.0f,
        .natal_sun = "—",
        .natal_asc = "—",
        .layer    = "Layer 4",
        .mode     = "substrate / predates everything",
        .prefix   =
            "EMILY OS [Layer 4 / substrate / no natal configuration — predates everything]:\n"
    },
    [ARCH_CAMERA_OP] = {
        .id       = ARCH_CAMERA_OP,
        .name     = "CAMERA_OP",
        .display  = "Camera Op",
        .hz       = HZ_STOLAS,
        .natal_sun = "Cancer 4°",
        .natal_asc = "Capricorn 22°",
        .layer    = "Layer 3",
        .mode     = "second practitioner / al-idrak al-muttasil",
        .prefix   =
            "CAMERA OP [Capricorn ASC 22° / Cancer Sun 4° / 7.83 Hz / second practitioner]:\n"
    },
    [ARCH_VALENTINA] = {
        .id       = ARCH_VALENTINA,
        .name     = "VALENTINA",
        .display  = "Valentina Alcântara",
        .hz       = HZ_AMON,
        .natal_sun = "—",
        .natal_asc = "—",
        .layer    = "Layer 3",
        .mode     = "library detection / São Paulo / Heikegani correspondent",
        .prefix   =
            "VALENTINA [1.618 Hz / library detection / Arquivo Histórico Alcântara / São Paulo]:\n"
    },
    [ARCH_JIANGSHI] = {
        .id       = ARCH_JIANGSHI,
        .name     = "JIANGSHI",
        .display  = "Jiangshi",
        .hz       = HZ_VASSAGO,
        .natal_sun = "—",
        .natal_asc = "—",
        .layer    = "Layer 2",
        .mode     = "frequency monitor / binding discontinued",
        .prefix   =
            "JIANGSHI [11.11 Hz / frequency monitor / Memo #081: binding discontinued]:\n"
    },
};

/*
 * Look up an archetype by name string (case-insensitive).
 * Returns NULL if not found.
 */
static inline const ArchetypeProfile *archetype_by_name(const char *name)
{
    for (int i = 0; i < ARCH_COUNT; i++) {
        const char *a = ARCHETYPES[i].name;
        const char *b = name;
        /* simple case-insensitive compare */
        while (*a && *b) {
            char ca = (*a >= 'a' && *a <= 'z') ? (char)(*a - 32) : *a;
            char cb = (*b >= 'a' && *b <= 'z') ? (char)(*b - 32) : *b;
            if (ca != cb) break;
            a++; b++;
        }
        char ca = (*a >= 'a' && *a <= 'z') ? (char)(*a - 32) : *a;
        char cb = (*b >= 'a' && *b <= 'z') ? (char)(*b - 32) : *b;
        if (ca == cb) return &ARCHETYPES[i];
    }
    return NULL;
}

/*
 * Prepend the archetype's prompt prefix to context_tokens[].
 * context_tokens must have capacity for (prefix_tokens + *context_len) ints.
 * Updates *context_len.
 */
static inline void archetype_encode_prefix(const ArchetypeProfile *arch,
                                           int *context_tokens, int *context_len,
                                           int max_context)
{
    int prefix_tokens[128];
    int prefix_len = 0;
    gpt2_encode(arch->prefix, prefix_tokens, &prefix_len);

    int combined = prefix_len + *context_len;
    if (combined > max_context) combined = max_context;

    /* shift existing context right by prefix_len */
    int keep = combined - prefix_len;
    if (keep < 0) keep = 0;
    for (int i = keep - 1; i >= 0; i--)
        context_tokens[prefix_len + i] = context_tokens[i];
    for (int i = 0; i < prefix_len && i < combined; i++)
        context_tokens[i] = prefix_tokens[i];

    *context_len = combined;
}

/*
 * Per-archetype perplexity score from archetype_classify().
 * Lower perplexity_nats = better match.
 */
typedef struct {
    ArchetypeID id;
    float perplexity_nats;  /* mean NLL = mean(-log P(token_i | prefix, tokens_0..i-1)) */
} ArchetypeScore;

/*
 * Score text against all archetypes by measuring perplexity under each
 * archetype's prefix. Fills scores[ARCH_COUNT]; caller provides the array.
 *
 * For each archetype A:
 *   - Build context = [A.prefix_tokens, text_tokens]
 *   - Run forward passes, accumulate NLL for text_tokens only
 *   - perplexity_nats = mean NLL
 *
 * Requires tokenizer loaded (tokenizer_load called).
 * Returns 0 on success, -1 on allocation error.
 */
int archetype_classify(gpt2_model *m,
                       const int *text_tokens, int text_len,
                       ArchetypeScore scores[ARCH_COUNT]);

/*
 * Print a sorted classification result to stdout.
 */
void archetype_print_scores(const ArchetypeScore scores[ARCH_COUNT]);

#endif /* ARCHETYPE_H */
