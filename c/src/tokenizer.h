/* Copyright (c) 2026 Bytedance Ltd. and/or its affiliate
 * Licensed under the Apache License, Version 2.0. See LICENSE. */

/* SentencePiece unigram tokenizer (replaces AutoTokenizer for the UMT5
 * text encoder). Parses spiece.model (a SentencePiece ModelProto) directly
 * and runs Viterbi segmentation.
 *
 * Note: the precompiled NFKC normalization charsmap is not applied; input is
 * assumed to be already-clean UTF-8 (the pipeline's prompt cleaning collapses
 * whitespace, which covers the common cases). */

#ifndef BERNINI_TOKENIZER_H
#define BERNINI_TOKENIZER_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct bn_tokenizer bn_tokenizer;

bn_tokenizer *bn_tokenizer_load(const char *spiece_model_path);
void          bn_tokenizer_free(bn_tokenizer *tk);

/* Tokenize like the T5 tokenizer with padding="max_length", truncation=True,
 * add_special_tokens=True: writes exactly `max_length` ids and 0/1 mask
 * entries. Returns the unpadded length (<= max_length). */
int bn_tokenize(const bn_tokenizer *tk, const char *text,
                int32_t *ids, int32_t *mask, int max_length);

int bn_tokenizer_eos_id(const bn_tokenizer *tk);
int bn_tokenizer_pad_id(const bn_tokenizer *tk);

#ifdef __cplusplus
}
#endif
#endif /* BERNINI_TOKENIZER_H */
