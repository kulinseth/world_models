/* Copyright (c) 2026 Bytedance Ltd. and/or its affiliate
 * Licensed under the Apache License, Version 2.0. See LICENSE. */

#include "tokenizer.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    char  *piece;
    int    len;
    float  score;
    int    type; /* 1 normal, 2 unknown, 3 control, 4 user, 6 byte */
} sp_piece;

struct bn_tokenizer {
    sp_piece *pieces;
    int       n_pieces;
    int       unk_id, eos_id, pad_id;
    int       max_piece_len;
    float     min_score;
    /* open-addressing hash: piece string -> id */
    int32_t *slots;
    size_t   nslots;
};

/* ----------------------------- protobuf parse ----------------------------- */

static uint64_t read_varint(const uint8_t **p, const uint8_t *end) {
    uint64_t v = 0;
    int shift = 0;
    while (*p < end) {
        uint8_t b = *(*p)++;
        v |= (uint64_t)(b & 0x7F) << shift;
        if (!(b & 0x80)) break;
        shift += 7;
    }
    return v;
}

static void skip_field(const uint8_t **p, const uint8_t *end, int wire) {
    switch (wire) {
    case 0: read_varint(p, end); break;
    case 1: *p += 8; break;
    case 2: {
        uint64_t len = read_varint(p, end);
        *p += len;
        break;
    }
    case 5: *p += 4; break;
    default: *p = end; break;
    }
}

static void parse_sentencepiece(bn_tokenizer *tk, const uint8_t *p, const uint8_t *end) {
    char *piece = NULL;
    int len = 0;
    float score = 0.0f;
    int type = 1;
    while (p < end) {
        uint64_t tag = read_varint(&p, end);
        int field = (int)(tag >> 3), wire = (int)(tag & 7);
        if (field == 1 && wire == 2) {
            uint64_t l = read_varint(&p, end);
            piece = (char *)malloc(l + 1);
            memcpy(piece, p, l);
            piece[l] = '\0';
            len = (int)l;
            p += l;
        } else if (field == 2 && wire == 5) {
            memcpy(&score, p, 4);
            p += 4;
        } else if (field == 3 && wire == 0) {
            type = (int)read_varint(&p, end);
        } else {
            skip_field(&p, end, wire);
        }
    }
    int id = tk->n_pieces++;
    tk->pieces = (sp_piece *)realloc(tk->pieces, sizeof(sp_piece) * (size_t)tk->n_pieces);
    tk->pieces[id].piece = piece ? piece : strdup("");
    tk->pieces[id].len = len;
    tk->pieces[id].score = score;
    tk->pieces[id].type = type;
}

static uint64_t fnv1a_n(const char *s, int n) {
    uint64_t h = 1469598103934665603ULL;
    for (int i = 0; i < n; i++) {
        h ^= (uint8_t)s[i];
        h *= 1099511628211ULL;
    }
    return h;
}

static int piece_lookup(const bn_tokenizer *tk, const char *s, int n) {
    uint64_t h = fnv1a_n(s, n) & (tk->nslots - 1);
    while (tk->slots[h] >= 0) {
        const sp_piece *pc = &tk->pieces[tk->slots[h]];
        if (pc->len == n && !memcmp(pc->piece, s, (size_t)n)) return tk->slots[h];
        h = (h + 1) & (tk->nslots - 1);
    }
    return -1;
}

bn_tokenizer *bn_tokenizer_load(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "tokenizer: cannot open %s\n", path);
        return NULL;
    }
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    uint8_t *buf = (uint8_t *)malloc((size_t)n);
    if (fread(buf, 1, (size_t)n, f) != (size_t)n) {
        fclose(f);
        free(buf);
        return NULL;
    }
    fclose(f);

    bn_tokenizer *tk = (bn_tokenizer *)calloc(1, sizeof(bn_tokenizer));
    tk->unk_id = -1;
    tk->eos_id = -1;
    tk->pad_id = 0;
    tk->min_score = 0.0f;

    const uint8_t *p = buf, *end = buf + n;
    while (p < end) {
        uint64_t tag = read_varint(&p, end);
        int field = (int)(tag >> 3), wire = (int)(tag & 7);
        if (field == 1 && wire == 2) { /* repeated SentencePiece pieces */
            uint64_t len = read_varint(&p, end);
            parse_sentencepiece(tk, p, p + len);
            p += len;
        } else {
            skip_field(&p, end, wire);
        }
    }
    free(buf);

    tk->nslots = 16;
    while (tk->nslots < (size_t)tk->n_pieces * 2 + 8) tk->nslots *= 2;
    tk->slots = (int32_t *)malloc(sizeof(int32_t) * tk->nslots);
    for (size_t i = 0; i < tk->nslots; i++) tk->slots[i] = -1;
    for (int i = 0; i < tk->n_pieces; i++) {
        sp_piece *pc = &tk->pieces[i];
        if (pc->type == 2) tk->unk_id = i;
        if (!strcmp(pc->piece, "</s>")) tk->eos_id = i;
        if (!strcmp(pc->piece, "<pad>")) tk->pad_id = i;
        if (pc->score < tk->min_score) tk->min_score = pc->score;
        if (pc->len > tk->max_piece_len) tk->max_piece_len = pc->len;
        if (pc->type == 1 || pc->type == 4) { /* only scoreable pieces match */
            uint64_t h = fnv1a_n(pc->piece, pc->len) & (tk->nslots - 1);
            while (tk->slots[h] >= 0) h = (h + 1) & (tk->nslots - 1);
            tk->slots[h] = i;
        }
    }
    if (tk->unk_id < 0) tk->unk_id = 2;
    if (tk->eos_id < 0) tk->eos_id = 1;
    fprintf(stderr, "tokenizer: %d pieces (unk=%d eos=%d pad=%d)\n",
            tk->n_pieces, tk->unk_id, tk->eos_id, tk->pad_id);
    return tk;
}

void bn_tokenizer_free(bn_tokenizer *tk) {
    if (!tk) return;
    for (int i = 0; i < tk->n_pieces; i++) free(tk->pieces[i].piece);
    free(tk->pieces);
    free(tk->slots);
    free(tk);
}

int bn_tokenizer_eos_id(const bn_tokenizer *tk) { return tk->eos_id; }
int bn_tokenizer_pad_id(const bn_tokenizer *tk) { return tk->pad_id; }

static int utf8_char_len(uint8_t c) {
    if (c < 0x80) return 1;
    if ((c >> 5) == 0x6) return 2;
    if ((c >> 4) == 0xE) return 3;
    if ((c >> 3) == 0x1E) return 4;
    return 1;
}

int bn_tokenize(const bn_tokenizer *tk, const char *text,
                int32_t *ids, int32_t *mask, int max_length) {
    /* SentencePiece pre-tokenization: add a dummy prefix space and replace
     * each space with the metaspace character U+2581. */
    size_t in_len = strlen(text);
    char *s = (char *)malloc(in_len * 3 + 4);
    size_t sl = 0;
    const char *meta = "\xE2\x96\x81";
    memcpy(s + sl, meta, 3);
    sl += 3;
    for (size_t i = 0; i < in_len; i++) {
        if (text[i] == ' ') {
            memcpy(s + sl, meta, 3);
            sl += 3;
        } else {
            s[sl++] = text[i];
        }
    }
    s[sl] = '\0';

    /* Viterbi over byte positions */
    int n = (int)sl;
    float *best = (float *)malloc(sizeof(float) * (size_t)(n + 1));
    int *back = (int *)malloc(sizeof(int) * (size_t)(n + 1));       /* prev position */
    int *back_id = (int *)malloc(sizeof(int) * (size_t)(n + 1));    /* piece id */
    best[0] = 0.0f;
    for (int i = 1; i <= n; i++) {
        best[i] = -1e30f;
        back[i] = -1;
        back_id[i] = -1;
    }
    const float unk_penalty = tk->min_score - 10.0f;
    for (int i = 0; i < n; i++) {
        if (best[i] <= -1e29f) continue;
        int max_l = tk->max_piece_len;
        if (max_l > n - i) max_l = n - i;
        for (int l = 1; l <= max_l; l++) {
            int id = piece_lookup(tk, s + i, l);
            if (id < 0) continue;
            float sc = best[i] + tk->pieces[id].score;
            if (sc > best[i + l]) {
                best[i + l] = sc;
                back[i + l] = i;
                back_id[i + l] = id;
            }
        }
        /* unknown fallback: one UTF-8 character as <unk> */
        int cl = utf8_char_len((uint8_t)s[i]);
        if (cl > n - i) cl = n - i;
        float sc = best[i] + unk_penalty;
        if (sc > best[i + cl]) {
            best[i + cl] = sc;
            back[i + cl] = i;
            back_id[i + cl] = tk->unk_id;
        }
    }

    /* walk back */
    int *rev = (int *)malloc(sizeof(int) * (size_t)(n + 1));
    int count = 0, pos = n;
    while (pos > 0 && back[pos] >= 0) {
        rev[count++] = back_id[pos];
        pos = back[pos];
    }
    free(best);
    free(back);
    free(back_id);
    free(s);

    int out = 0;
    for (int i = count - 1; i >= 0 && out < max_length - 1; i--) {
        /* merge consecutive unk like sentencepiece does */
        if (out > 0 && ids[out - 1] == tk->unk_id && rev[i] == tk->unk_id) continue;
        ids[out++] = rev[i];
    }
    free(rev);
    ids[out++] = tk->eos_id;
    int real_len = out;
    for (; out < max_length; out++) ids[out] = tk->pad_id;
    if (mask)
        for (int i = 0; i < max_length; i++) mask[i] = i < real_len ? 1 : 0;
    return real_len;
}
