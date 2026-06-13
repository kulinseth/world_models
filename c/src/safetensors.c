/* Copyright (c) 2026 Bytedance Ltd. and/or its affiliate
 * Licensed under the Apache License, Version 2.0. See LICENSE. */

#include "safetensors.h"
#include "bjson.h"

#include <dirent.h>
#include <fcntl.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

typedef enum { ST_F32, ST_F16, ST_BF16, ST_F64, ST_I64, ST_I32, ST_OTHER } st_dtype;

typedef struct {
    char    *key;
    int      shard;
    st_dtype dtype;
    int      ndim;
    int64_t  shape[BT_MAX_DIMS];
    int64_t  begin, end; /* byte offsets into the shard data section */
} st_entry;

typedef struct {
    char    *path;
    int      fd;
    uint8_t *map;
    size_t   map_size;
    size_t   data_off; /* start of the tensor data section */
} st_shard;

struct st_store {
    st_shard *shards;
    int       nshards;
    st_entry *entries;
    size_t    nentries, cap;
    /* open-addressing hash of entry indices keyed by entry key */
    int32_t  *slots;
    size_t    nslots;
};

static uint64_t fnv1a(const char *s) {
    uint64_t h = 1469598103934665603ULL;
    while (*s) {
        h ^= (uint8_t)*s++;
        h *= 1099511628211ULL;
    }
    return h;
}

static void st_index_build(st_store *st) {
    st->nslots = 16;
    while (st->nslots < st->nentries * 2 + 8) st->nslots *= 2;
    st->slots = (int32_t *)malloc(sizeof(int32_t) * st->nslots);
    for (size_t i = 0; i < st->nslots; i++) st->slots[i] = -1;
    for (size_t i = 0; i < st->nentries; i++) {
        uint64_t h = fnv1a(st->entries[i].key) & (st->nslots - 1);
        while (st->slots[h] >= 0) h = (h + 1) & (st->nslots - 1);
        st->slots[h] = (int32_t)i;
    }
}

static const st_entry *st_find(const st_store *st, const char *key) {
    if (!st->slots) return NULL;
    uint64_t h = fnv1a(key) & (st->nslots - 1);
    while (st->slots[h] >= 0) {
        const st_entry *e = &st->entries[st->slots[h]];
        if (!strcmp(e->key, key)) return e;
        h = (h + 1) & (st->nslots - 1);
    }
    return NULL;
}

static st_dtype parse_dtype(const char *s) {
    if (!strcmp(s, "F32")) return ST_F32;
    if (!strcmp(s, "F16")) return ST_F16;
    if (!strcmp(s, "BF16")) return ST_BF16;
    if (!strcmp(s, "F64")) return ST_F64;
    if (!strcmp(s, "I64")) return ST_I64;
    if (!strcmp(s, "I32")) return ST_I32;
    return ST_OTHER;
}

static int st_load_shard(st_store *st, const char *path) {
    int fd = open(path, O_RDONLY);
    if (fd < 0) {
        fprintf(stderr, "safetensors: cannot open %s\n", path);
        return -1;
    }
    struct stat sb;
    fstat(fd, &sb);
    uint8_t *map = (uint8_t *)mmap(NULL, (size_t)sb.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (map == MAP_FAILED) {
        close(fd);
        fprintf(stderr, "safetensors: mmap failed for %s\n", path);
        return -1;
    }
    uint64_t hdr_len = 0;
    memcpy(&hdr_len, map, 8);
    char *hdr = (char *)malloc(hdr_len + 1);
    memcpy(hdr, map + 8, hdr_len);
    hdr[hdr_len] = '\0';
    bj_value *doc = bj_parse(hdr);
    free(hdr);
    if (!doc) {
        munmap(map, (size_t)sb.st_size);
        close(fd);
        fprintf(stderr, "safetensors: bad header in %s\n", path);
        return -1;
    }

    int shard_idx = st->nshards;
    st->shards = (st_shard *)realloc(st->shards, sizeof(st_shard) * (size_t)(st->nshards + 1));
    st_shard *sh = &st->shards[st->nshards++];
    sh->path = strdup(path);
    sh->fd = fd;
    sh->map = map;
    sh->map_size = (size_t)sb.st_size;
    sh->data_off = 8 + (size_t)hdr_len;

    for (size_t i = 0; i < doc->count; i++) {
        const char *key = doc->keys[i];
        bj_value *meta = doc->items[i];
        if (!strcmp(key, "__metadata__") || meta->type != BJ_OBJECT) continue;
        if (st->nentries == st->cap) {
            st->cap = st->cap ? st->cap * 2 : 256;
            st->entries = (st_entry *)realloc(st->entries, sizeof(st_entry) * st->cap);
        }
        st_entry *e = &st->entries[st->nentries++];
        memset(e, 0, sizeof(*e));
        e->key = strdup(key);
        e->shard = shard_idx;
        e->dtype = parse_dtype(bj_get_str(meta, "dtype", ""));
        bj_value *shape = bj_get(meta, "shape");
        e->ndim = 0;
        if (shape && shape->type == BJ_ARRAY) {
            for (size_t d = 0; d < shape->count && d < BT_MAX_DIMS; d++)
                e->shape[e->ndim++] = (int64_t)shape->items[d]->num;
        }
        bj_value *offs = bj_get(meta, "data_offsets");
        if (offs && offs->type == BJ_ARRAY && offs->count == 2) {
            e->begin = (int64_t)offs->items[0]->num;
            e->end = (int64_t)offs->items[1]->num;
        }
    }
    bj_free(doc);
    return 0;
}

static int ends_with(const char *s, const char *suf) {
    size_t ls = strlen(s), lf = strlen(suf);
    return ls >= lf && !strcmp(s + ls - lf, suf);
}

static int cmp_str(const void *a, const void *b) {
    return strcmp(*(const char *const *)a, *(const char *const *)b);
}

st_store *st_open_dir(const char *dir) {
    DIR *d = opendir(dir);
    if (!d) {
        fprintf(stderr, "safetensors: cannot open dir %s\n", dir);
        return NULL;
    }
    char **files = NULL;
    size_t nfiles = 0;
    struct dirent *de;
    while ((de = readdir(d)) != NULL) {
        if (ends_with(de->d_name, ".safetensors")) {
            files = (char **)realloc(files, sizeof(char *) * (nfiles + 1));
            files[nfiles++] = strdup(de->d_name);
        }
    }
    closedir(d);
    if (nfiles == 0) {
        free(files);
        return NULL;
    }
    qsort(files, nfiles, sizeof(char *), cmp_str);

    st_store *st = (st_store *)calloc(1, sizeof(st_store));
    char path[4096];
    for (size_t i = 0; i < nfiles; i++) {
        snprintf(path, sizeof(path), "%s/%s", dir, files[i]);
        st_load_shard(st, path);
        free(files[i]);
    }
    free(files);
    st_index_build(st);
    return st;
}

void st_close(st_store *st) {
    if (!st) return;
    for (int i = 0; i < st->nshards; i++) {
        munmap(st->shards[i].map, st->shards[i].map_size);
        close(st->shards[i].fd);
        free(st->shards[i].path);
    }
    for (size_t i = 0; i < st->nentries; i++) free(st->entries[i].key);
    free(st->entries);
    free(st->shards);
    free(st->slots);
    free(st);
}

int st_has(const st_store *st, const char *key) { return st_find(st, key) != NULL; }

bt_t *st_get(const st_store *st, const char *key) {
    const st_entry *e = st_find(st, key);
    if (!e) return NULL;
    const st_shard *sh = &st->shards[e->shard];
    const uint8_t *src = sh->map + sh->data_off + (size_t)e->begin;
    int64_t shape0[1] = {1};
    bt_t *t = e->ndim ? bt_new(e->ndim, e->shape) : bt_new(1, shape0);
    int64_t n = t->numel;
    switch (e->dtype) {
    case ST_F32:
        memcpy(t->data, src, sizeof(float) * (size_t)n);
        break;
    case ST_BF16: {
        const uint16_t *p = (const uint16_t *)src;
        for (int64_t i = 0; i < n; i++) {
            uint32_t bits = ((uint32_t)p[i]) << 16;
            memcpy(&t->data[i], &bits, 4);
        }
        break;
    }
    case ST_F16: {
        const uint16_t *p = (const uint16_t *)src;
        for (int64_t i = 0; i < n; i++) {
            uint16_t hv = p[i];
            uint32_t sign = (uint32_t)(hv >> 15) << 31;
            uint32_t exp = (hv >> 10) & 0x1F;
            uint32_t man = hv & 0x3FF;
            uint32_t bits;
            if (exp == 0) {
                if (man == 0) bits = sign;
                else { /* subnormal */
                    int sft = 0;
                    while (!(man & 0x400)) { man <<= 1; sft++; }
                    man &= 0x3FF;
                    bits = sign | ((uint32_t)(127 - 15 - sft) << 23) | (man << 13);
                }
            } else if (exp == 31) {
                bits = sign | 0x7F800000u | (man << 13);
            } else {
                bits = sign | ((exp - 15 + 127) << 23) | (man << 13);
            }
            memcpy(&t->data[i], &bits, 4);
        }
        break;
    }
    case ST_F64: {
        const double *p = (const double *)src;
        for (int64_t i = 0; i < n; i++) t->data[i] = (float)p[i];
        break;
    }
    case ST_I64: {
        const int64_t *p = (const int64_t *)src;
        for (int64_t i = 0; i < n; i++) t->data[i] = (float)p[i];
        break;
    }
    case ST_I32: {
        const int32_t *p = (const int32_t *)src;
        for (int64_t i = 0; i < n; i++) t->data[i] = (float)p[i];
        break;
    }
    default:
        fprintf(stderr, "safetensors: unsupported dtype for key %s\n", key);
        bt_free(t);
        return NULL;
    }
    return t;
}

char *st_resolve(const st_store *st, const char *prefix, const char *name) {
    size_t need = strlen(prefix) + strlen(name) + 8;
    char *key = (char *)malloc(need);
    /* Prefer the EMA copy when both exist, mirroring weights._select_keys. */
    snprintf(key, need, "ema.%s%s", prefix, name);
    if (st_find(st, key)) return key;
    snprintf(key, need, "%s%s", prefix, name);
    if (st_find(st, key)) return key;
    free(key);
    return NULL;
}

int st_pick_prefix(const st_store *st, const char *const *prefixes, int n) {
    for (int p = 0; p < n; p++) {
        size_t plen = strlen(prefixes[p]);
        for (size_t i = 0; i < st->nentries; i++) {
            const char *base = st->entries[i].key;
            if (!strncmp(base, "ema.", 4)) base += 4;
            if (!strncmp(base, prefixes[p], plen)) return p;
        }
    }
    return -1;
}

bt_t *st_get_pfx(const st_store *st, const char *prefix, const char *fmt, ...) {
    char name[512];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(name, sizeof(name), fmt, ap);
    va_end(ap);
    char *key = st_resolve(st, prefix ? prefix : "", name);
    if (!key) {
        fprintf(stderr, "safetensors: missing tensor '%s%s'\n", prefix ? prefix : "", name);
        return NULL;
    }
    bt_t *t = st_get(st, key);
    free(key);
    return t;
}
