/* Copyright (c) 2026 Bytedance Ltd. and/or its affiliate
 * Licensed under the Apache License, Version 2.0. See LICENSE. */

/* safetensors: mmap-based reader for (sharded) .safetensors checkpoints.
 * Mirrors bernini/weights.py: index.json weight maps, key-prefix candidates
 * and the EMA-copy preference are all handled here. */

#ifndef BERNINI_SAFETENSORS_H
#define BERNINI_SAFETENSORS_H

#include "bt.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct st_store st_store;

/* Open every .safetensors shard in `dir` (using *.safetensors.index.json when
 * present). Returns NULL when no shard is found. */
st_store *st_open_dir(const char *dir);
void      st_close(st_store *st);

int st_has(const st_store *st, const char *key);

/* Fetch a tensor by exact key, converted to float32. Caller frees. */
bt_t *st_get(const st_store *st, const char *key);

/* Resolve `name` through candidate key prefixes the way weights.py does:
 * for the chosen prefix, prefer "ema." + prefix + name over prefix + name.
 * Returns a malloc'd source key, or NULL when absent. */
char *st_resolve(const st_store *st, const char *prefix, const char *name);

/* First prefix (of n candidates) that matches at least one key, or -1.
 * Mirrors the prefix-candidate loop of load_transformer_state_dict. */
int st_pick_prefix(const st_store *st, const char *const *prefixes, int n);

/* Convenience: resolve + fetch, with error logging. NULL when missing. */
bt_t *st_get_pfx(const st_store *st, const char *prefix, const char *fmt, ...);

#ifdef __cplusplus
}
#endif
#endif /* BERNINI_SAFETENSORS_H */
