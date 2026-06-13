/* Copyright (c) 2026 Bytedance Ltd. and/or its affiliate
 * Licensed under the Apache License, Version 2.0. See LICENSE. */

#include "bjson.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    const char *p;
} bj_parser;

static void skip_ws(bj_parser *ps) {
    while (*ps->p == ' ' || *ps->p == '\t' || *ps->p == '\n' || *ps->p == '\r') ps->p++;
}

static bj_value *bj_alloc(bj_type t) {
    bj_value *v = (bj_value *)calloc(1, sizeof(bj_value));
    v->type = t;
    return v;
}

static bj_value *parse_value(bj_parser *ps);

static char *parse_string_raw(bj_parser *ps) {
    if (*ps->p != '"') return NULL;
    ps->p++;
    size_t cap = 32, len = 0;
    char *s = (char *)malloc(cap);
    while (*ps->p && *ps->p != '"') {
        char c = *ps->p++;
        if (c == '\\') {
            char e = *ps->p++;
            switch (e) {
            case 'n': c = '\n'; break;
            case 't': c = '\t'; break;
            case 'r': c = '\r'; break;
            case 'b': c = '\b'; break;
            case 'f': c = '\f'; break;
            case '/': c = '/'; break;
            case '\\': c = '\\'; break;
            case '"': c = '"'; break;
            case 'u': {
                /* decode \uXXXX to UTF-8 (no surrogate pairing for brevity;
                 * config files only carry BMP text) */
                unsigned cp = 0;
                for (int i = 0; i < 4 && *ps->p; i++) {
                    char h = *ps->p++;
                    cp <<= 4;
                    if (h >= '0' && h <= '9') cp |= (unsigned)(h - '0');
                    else if (h >= 'a' && h <= 'f') cp |= (unsigned)(h - 'a' + 10);
                    else if (h >= 'A' && h <= 'F') cp |= (unsigned)(h - 'A' + 10);
                }
                if (len + 4 >= cap) { cap = cap * 2 + 4; s = (char *)realloc(s, cap); }
                if (cp < 0x80) s[len++] = (char)cp;
                else if (cp < 0x800) {
                    s[len++] = (char)(0xC0 | (cp >> 6));
                    s[len++] = (char)(0x80 | (cp & 0x3F));
                } else {
                    s[len++] = (char)(0xE0 | (cp >> 12));
                    s[len++] = (char)(0x80 | ((cp >> 6) & 0x3F));
                    s[len++] = (char)(0x80 | (cp & 0x3F));
                }
                continue;
            }
            default: c = e; break;
            }
        }
        if (len + 1 >= cap) { cap *= 2; s = (char *)realloc(s, cap); }
        s[len++] = c;
    }
    if (*ps->p == '"') ps->p++;
    s[len] = '\0';
    return s;
}

static void obj_push(bj_value *v, char *key, bj_value *item) {
    v->keys = (char **)realloc(v->keys, sizeof(char *) * (v->count + 1));
    v->items = (bj_value **)realloc(v->items, sizeof(bj_value *) * (v->count + 1));
    v->keys[v->count] = key;
    v->items[v->count] = item;
    v->count++;
}

static bj_value *parse_value(bj_parser *ps) {
    skip_ws(ps);
    char c = *ps->p;
    if (c == '{') {
        ps->p++;
        bj_value *v = bj_alloc(BJ_OBJECT);
        skip_ws(ps);
        if (*ps->p == '}') { ps->p++; return v; }
        for (;;) {
            skip_ws(ps);
            char *key = parse_string_raw(ps);
            if (!key) { bj_free(v); return NULL; }
            skip_ws(ps);
            if (*ps->p != ':') { free(key); bj_free(v); return NULL; }
            ps->p++;
            bj_value *item = parse_value(ps);
            if (!item) { free(key); bj_free(v); return NULL; }
            obj_push(v, key, item);
            skip_ws(ps);
            if (*ps->p == ',') { ps->p++; continue; }
            if (*ps->p == '}') { ps->p++; return v; }
            bj_free(v);
            return NULL;
        }
    }
    if (c == '[') {
        ps->p++;
        bj_value *v = bj_alloc(BJ_ARRAY);
        skip_ws(ps);
        if (*ps->p == ']') { ps->p++; return v; }
        for (;;) {
            bj_value *item = parse_value(ps);
            if (!item) { bj_free(v); return NULL; }
            obj_push(v, NULL, item);
            skip_ws(ps);
            if (*ps->p == ',') { ps->p++; continue; }
            if (*ps->p == ']') { ps->p++; return v; }
            bj_free(v);
            return NULL;
        }
    }
    if (c == '"') {
        bj_value *v = bj_alloc(BJ_STRING);
        v->str = parse_string_raw(ps);
        if (!v->str) { bj_free(v); return NULL; }
        return v;
    }
    if (!strncmp(ps->p, "true", 4)) { ps->p += 4; bj_value *v = bj_alloc(BJ_BOOL); v->num = 1; return v; }
    if (!strncmp(ps->p, "false", 5)) { ps->p += 5; bj_value *v = bj_alloc(BJ_BOOL); v->num = 0; return v; }
    if (!strncmp(ps->p, "null", 4)) { ps->p += 4; return bj_alloc(BJ_NULL); }
    /* number */
    {
        char *end = NULL;
        double d = strtod(ps->p, &end);
        if (end == ps->p) return NULL;
        ps->p = end;
        bj_value *v = bj_alloc(BJ_NUMBER);
        v->num = d;
        return v;
    }
}

bj_value *bj_parse(const char *text) {
    bj_parser ps = {text};
    bj_value *v = parse_value(&ps);
    return v;
}

bj_value *bj_parse_file(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    char *buf = (char *)malloc((size_t)n + 1);
    if (fread(buf, 1, (size_t)n, f) != (size_t)n) { fclose(f); free(buf); return NULL; }
    fclose(f);
    buf[n] = '\0';
    bj_value *v = bj_parse(buf);
    free(buf);
    return v;
}

void bj_free(bj_value *v) {
    if (!v) return;
    if (v->type == BJ_ARRAY || v->type == BJ_OBJECT) {
        for (size_t i = 0; i < v->count; i++) {
            if (v->keys && v->keys[i]) free(v->keys[i]);
            bj_free(v->items[i]);
        }
        free(v->keys);
        free(v->items);
    }
    free(v->str);
    free(v);
}

bj_value *bj_get(const bj_value *obj, const char *key) {
    if (!obj || obj->type != BJ_OBJECT) return NULL;
    for (size_t i = 0; i < obj->count; i++)
        if (obj->keys[i] && !strcmp(obj->keys[i], key)) return obj->items[i];
    return NULL;
}

double bj_get_num(const bj_value *obj, const char *key, double dflt) {
    bj_value *v = bj_get(obj, key);
    if (!v || (v->type != BJ_NUMBER && v->type != BJ_BOOL)) return dflt;
    return v->num;
}

int bj_get_bool(const bj_value *obj, const char *key, int dflt) {
    bj_value *v = bj_get(obj, key);
    if (!v || (v->type != BJ_BOOL && v->type != BJ_NUMBER)) return dflt;
    return v->num != 0;
}

const char *bj_get_str(const bj_value *obj, const char *key, const char *dflt) {
    bj_value *v = bj_get(obj, key);
    if (!v || v->type != BJ_STRING) return dflt;
    return v->str;
}

size_t bj_get_farray(const bj_value *obj, const char *key, float *out, size_t max) {
    bj_value *v = bj_get(obj, key);
    if (!v || v->type != BJ_ARRAY) return 0;
    size_t n = v->count < max ? v->count : max;
    for (size_t i = 0; i < n; i++) out[i] = (float)v->items[i]->num;
    return n;
}

size_t bj_get_iarray(const bj_value *obj, const char *key, int *out, size_t max) {
    bj_value *v = bj_get(obj, key);
    if (!v || v->type != BJ_ARRAY) return 0;
    size_t n = v->count < max ? v->count : max;
    for (size_t i = 0; i < n; i++) out[i] = (int)v->items[i]->num;
    return n;
}
