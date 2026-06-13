/* Copyright (c) 2026 Bytedance Ltd. and/or its affiliate
 * Licensed under the Apache License, Version 2.0. See LICENSE. */

/* bjson: minimal JSON DOM parser, just enough for model config files and
 * safetensors headers (objects, arrays, strings, numbers, bools, null). */

#ifndef BERNINI_BJSON_H
#define BERNINI_BJSON_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    BJ_NULL, BJ_BOOL, BJ_NUMBER, BJ_STRING, BJ_ARRAY, BJ_OBJECT
} bj_type;

typedef struct bj_value bj_value;
struct bj_value {
    bj_type type;
    double  num;            /* BJ_NUMBER / BJ_BOOL (0/1) */
    char   *str;            /* BJ_STRING (owned) */
    /* BJ_ARRAY: items[0..count); BJ_OBJECT: keys[i] -> items[i] */
    char     **keys;
    bj_value **items;
    size_t     count;
};

/* Parse a JSON document from a NUL-terminated buffer. Returns NULL on error. */
bj_value *bj_parse(const char *text);
bj_value *bj_parse_file(const char *path);
void      bj_free(bj_value *v);

/* Object lookup; returns NULL if missing or wrong container type. */
bj_value *bj_get(const bj_value *obj, const char *key);

/* Typed getters with defaults. */
double      bj_get_num(const bj_value *obj, const char *key, double dflt);
int         bj_get_bool(const bj_value *obj, const char *key, int dflt);
const char *bj_get_str(const bj_value *obj, const char *key, const char *dflt);
/* Copy a float array field into out (up to max); returns count copied. */
size_t      bj_get_farray(const bj_value *obj, const char *key, float *out, size_t max);
size_t      bj_get_iarray(const bj_value *obj, const char *key, int *out, size_t max);

#ifdef __cplusplus
}
#endif
#endif /* BERNINI_BJSON_H */
