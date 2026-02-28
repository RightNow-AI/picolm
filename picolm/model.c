#include "model.h"
#include "tensor.h"
#include "quant.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

/* ---- GGUF metadata value types ---- */
enum {
    GGUF_META_UINT8   = 0,
    GGUF_META_INT8    = 1,
    GGUF_META_UINT16  = 2,
    GGUF_META_INT16   = 3,
    GGUF_META_UINT32  = 4,
    GGUF_META_INT32   = 5,
    GGUF_META_FLOAT32 = 6,
    GGUF_META_BOOL    = 7,
    GGUF_META_STRING  = 8,
    GGUF_META_ARRAY   = 9,
    GGUF_META_UINT64  = 10,
    GGUF_META_INT64   = 11,
    GGUF_META_FLOAT64 = 12,
};

/* ---- Helpers for reading GGUF binary format ---- */

typedef struct {
    const uint8_t *data;
    size_t pos;
    size_t size;
} reader_t;

static uint32_t read_u32(reader_t *r) {
    uint32_t v;
    if (r->pos + 4 > r->size) { fprintf(stderr, "GGUF read out of bounds\n"); exit(1); }
    memcpy(&v, r->data + r->pos, 4);
    r->pos += 4;
    return v;
}

static int32_t read_i32(reader_t *r) {
    int32_t v;
    if (r->pos + 4 > r->size) { fprintf(stderr, "GGUF read out of bounds\n"); exit(1); }
    memcpy(&v, r->data + r->pos, 4);
    r->pos += 4;
    return v;
}

static uint64_t read_u64(reader_t *r) {
    uint64_t v;
    if (r->pos + 8 > r->size) { fprintf(stderr, "GGUF read out of bounds\n"); exit(1); }
    memcpy(&v, r->data + r->pos, 8);
    r->pos += 8;
    return v;
}

static uint16_t read_u16(reader_t *r) {
    uint16_t v;
    if (r->pos + 2 > r->size) { fprintf(stderr, "GGUF read out of bounds\n"); exit(1); }
    memcpy(&v, r->data + r->pos, 2);
    r->pos += 2;
    return v;
}

static uint8_t read_u8(reader_t *r) {
    uint8_t v;
    if (r->pos + 1 > r->size) { fprintf(stderr, "GGUF read out of bounds\n"); exit(1); }
    v = r->data[r->pos];
    r->pos += 1;
    return v;
}

static float read_f32(reader_t *r) {
    float v;
    if (r->pos + 4 > r->size) { fprintf(stderr, "GGUF read out of bounds\n"); exit(1); }
    memcpy(&v, r->data + r->pos, 4);
    r->pos += 4;
    return v;
}

typedef struct {
    char *str;
    uint64_t len;
} gguf_str_t;

static gguf_str_t read_gguf_string(reader_t *r) {
    gguf_str_t s;
    s.len = read_u64(r);
    if (r->pos + (size_t)s.len > r->size) { fprintf(stderr, "GGUF string out of bounds\n"); exit(1); }
    s.str = (char *)malloc((size_t)s.len + 1);
    if (!s.str) { fprintf(stderr, "OOM reading gguf string\n"); exit(1); }
    memcpy(s.str, r->data + r->pos, (size_t)s.len);
    s.str[s.len] = '\0';
    r->pos += (size_t)s.len;
    return s;
}

static int str_eq(gguf_str_t s, const char *lit) {
    size_t n = strlen(lit);
    return (s.len == (uint64_t)n) && (memcmp(s.str, lit, n) == 0);
}

static uint64_t skip_meta_value(reader_t *r, uint32_t vtype, int *is_numeric) {
    *is_numeric = 1;
    switch (vtype) {
        case GGUF_META_UINT8:   return read_u8(r);
        case GGUF_META_INT8:    return (uint64_t)(int64_t)(int8_t)read_u8(r);
        case GGUF_META_UINT16:  return read_u16(r);
        case GGUF_META_INT16:   return (uint64_t)(int64_t)(int16_t)read_u16(r);
        case GGUF_META_UINT32:  return read_u32(r);
        case GGUF_META_INT32:   return (uint64_t)(int64_t)read_i32(r);
        case GGUF_META_UINT64:  return read_u64(r);
        case GGUF_META_INT64:   return read_u64(r);
        case GGUF_META_FLOAT32: { read_f32(r); *is_numeric = 0; return 0; }
        case GGUF_META_FLOAT64: { r->pos += 8; *is_numeric = 0; return 0; }
        case GGUF_META_BOOL:    return read_u8(r);
        case GGUF_META_STRING:  { read_gguf_string(r); *is_numeric = 0; return 0; }
        case GGUF_META_ARRAY: {
            *is_numeric = 0;
            uint32_t arr_type = read_u32(r);
            uint64_t arr_len  = read_u64(r);
            int dummy;
            for (uint64_t i = 0; i < arr_len; i++) {
                skip_meta_value(r, arr_type, &dummy);
            }
            return 0;
        }
        default:
            fprintf(stderr, "Unknown GGUF metadata type: %u\n", vtype);
            exit(1);
    }
}

/* ---- GGUF Parser ---- */

/* Align 'x' up to the next multiple of 'a'. Works for any a >= 1. */
static size_t align_up_any(size_t x, size_t a) {
    if (a == 0) return x; /* defensive */
    size_t rem = x % a;
    return rem ? (x + (a - rem)) : x;
}

static int parse_gguf(model_t *m, int max_seq_len) {
    reader_t r = { .data = (const uint8_t *)m->mmap_addr, .pos = 0, .size = m->mmap_size };
    model_config_t *cfg = &m->config;

    /* Start from a known state in case some GGUF metadata fields are missing. */
    memset(cfg, 0, sizeof(*cfg));

    uint32_t magic = read_u32(&r);
    if (magic != GGUF_MAGIC) {
        fprintf(stderr, "Invalid GGUF magic: 0x%08X\n", magic);
        return -1;
    }

    uint32_t version = read_u32(&r);
    if (version < 2 || version > 3) {
        fprintf(stderr, "Unsupported GGUF version: %u\n", version);
        return -1;
    }

    uint64_t n_tensors  = read_u64(&r);
    uint64_t n_metadata = read_u64(&r);

    cfg->alignment = 32;
    cfg->rope_freq_base = 10000.0f;
    cfg->max_seq_len = 2048;
    cfg->weight_type = GGUF_TYPE_F16;
    m->tok_bos_id = 1;
    m->tok_eos_id = 2;

    for (uint64_t i = 0; i < n_metadata; i++) {
        gguf_str_t key = read_gguf_string(&r);
        uint32_t vtype = read_u32(&r);

        if (str_eq(key, "llama.embedding_length") || str_eq(key, "general.embedding_length")) {
            int dummy; cfg->n_embd = (int)skip_meta_value(&r, vtype, &dummy);
        } else if (str_eq(key, "llama.feed_forward_length") || str_eq(key, "general.feed_forward_length")) {
            int dummy; cfg->n_ffn = (int)skip_meta_value(&r, vtype, &dummy);
        } else if (str_eq(key, "llama.attention.head_count")) {
            int dummy; cfg->n_heads = (int)skip_meta_value(&r, vtype, &dummy);
        } else if (str_eq(key, "llama.attention.head_count_kv")) {
            int dummy; cfg->n_kv_heads = (int)skip_meta_value(&r, vtype, &dummy);
        } else if (str_eq(key, "llama.block_count")) {
            int dummy; cfg->n_layers = (int)skip_meta_value(&r, vtype, &dummy);
        } else if (str_eq(key, "llama.context_length")) {
            int dummy; cfg->max_seq_len = (int)skip_meta_value(&r, vtype, &dummy);
        } else if (str_eq(key, "llama.rope.freq_base")) {
            if (vtype == GGUF_META_FLOAT32) {
                cfg->rope_freq_base = read_f32(&r);
            } else {
                int dummy; skip_meta_value(&r, vtype, &dummy);
            }
        } else if (str_eq(key, "general.alignment")) {
            int dummy; cfg->alignment = (int)skip_meta_value(&r, vtype, &dummy);
        } else if (str_eq(key, "llama.vocab_size")) {
            int dummy; cfg->vocab_size = (int)skip_meta_value(&r, vtype, &dummy);
        } else if (str_eq(key, "tokenizer.ggml.bos_token_id")) {
            int dummy; m->tok_bos_id = (uint32_t)skip_meta_value(&r, vtype, &dummy);
        } else if (str_eq(key, "tokenizer.ggml.eos_token_id")) {
            int dummy; m->tok_eos_id = (uint32_t)skip_meta_value(&r, vtype, &dummy);
        } else if (str_eq(key, "tokenizer.ggml.tokens")) {
            if (vtype != GGUF_META_ARRAY) {
                int dummy; skip_meta_value(&r, vtype, &dummy);
            } else {
                uint32_t arr_type = read_u32(&r);
                uint64_t arr_len  = read_u64(&r);
                m->tok_tokens_data = r.data + r.pos;
                m->tok_n_tokens = arr_len;
                int dummy;
                for (uint64_t j = 0; j < arr_len; j++) {
                    skip_meta_value(&r, arr_type, &dummy);
                }
            }
        } else if (str_eq(key, "tokenizer.ggml.scores")) {
            if (vtype != GGUF_META_ARRAY) {
                int dummy; skip_meta_value(&r, vtype, &dummy);
            } else {
                uint32_t arr_type = read_u32(&r);
                uint64_t arr_len  = read_u64(&r);
                (void)arr_type;
                m->tok_scores_data = r.data + r.pos;
                m->tok_n_scores = arr_len;
                r.pos += arr_len * 4;
            }
        } else {
            int dummy; skip_meta_value(&r, vtype, &dummy);
        }
    }

    /* ---- Validate required config ---- */
    if (cfg->n_embd <= 0 || cfg->n_heads <= 0 || cfg->n_layers <= 0 || cfg->vocab_size <= 0) {
        fprintf(stderr, "Invalid model config in GGUF metadata (n_embd=%d n_heads=%d n_layers=%d vocab_size=%d)\n",
                cfg->n_embd, cfg->n_heads, cfg->n_layers, cfg->vocab_size);
        return -1;
    }
    if (cfg->n_kv_heads <= 0) {
        /* Some GGUFs omit head_count_kv; default to MHA. */
        cfg->n_kv_heads = cfg->n_heads;
    }
    if (cfg->max_seq_len <= 0) {
        cfg->max_seq_len = 2048;
    }
    if (cfg->alignment <= 0) {
        cfg->alignment = 32;
    }

    if (max_seq_len > 0 && max_seq_len < cfg->max_seq_len) {
        cfg->max_seq_len = max_seq_len;
    }
    if (cfg->n_embd % cfg->n_heads != 0) {
        fprintf(stderr, "Invalid head configuration: n_embd=%d not divisible by n_heads=%d\n", cfg->n_embd, cfg->n_heads);
        return -1;
    }
    cfg->head_dim = cfg->n_embd / cfg->n_heads;

    /* Parse tensor info entries */
    typedef struct {
        gguf_str_t name;
        uint32_t   n_dims;
        uint64_t   dims[4];
        uint32_t   type;
        uint64_t   offset;
    } tensor_info_t;

    tensor_info_t *tinfos = (tensor_info_t *)malloc(n_tensors * sizeof(tensor_info_t));
    if (!tinfos) { fprintf(stderr, "OOM allocating tensor info\n"); return -1; }

    for (uint64_t i = 0; i < n_tensors; i++) {
        tinfos[i].name   = read_gguf_string(&r);
        tinfos[i].n_dims = read_u32(&r);
        for (uint32_t d = 0; d < tinfos[i].n_dims; d++) {
            tinfos[i].dims[d] = read_u64(&r);
        }
        tinfos[i].type   = read_u32(&r);
        tinfos[i].offset = read_u64(&r);
    }

    size_t alignment = (size_t)cfg->alignment;
    size_t tensor_data_base = align_up_any((size_t)r.pos, alignment);

    model_weights_t *w = &m->weights;
    memset(w, 0, sizeof(*w));

    for (uint64_t i = 0; i < n_tensors; i++) {
        const void *ptr = (const uint8_t *)m->mmap_addr + tensor_data_base + tinfos[i].offset;
        gguf_type_t qtype = (gguf_type_t)tinfos[i].type;

        if (str_eq(tinfos[i].name, "token_embd.weight")) {
            w->token_embd = ptr; w->type_token_embd = qtype;
        } else if (str_eq(tinfos[i].name, "output_norm.weight")) {
            w->output_norm = ptr; w->type_output_norm = qtype;
        } else if (str_eq(tinfos[i].name, "output.weight")) {
            w->output = ptr; w->type_output = qtype;
        } else {
            int layer = -1;
            char suffix[64] = {0};

            if (tinfos[i].name.len > 4 && memcmp(tinfos[i].name.str, "blk.", 4) == 0) {
                const char *p = tinfos[i].name.str + 4;
                const char *end = tinfos[i].name.str + tinfos[i].name.len;
                layer = 0;
                while (p < end && *p >= '0' && *p <= '9') {
                    layer = layer * 10 + (*p - '0');
                    p++;
                }
                if (p < end && *p == '.') p++;
                size_t slen = (size_t)(end - p);
                if (slen < sizeof(suffix)) {
                    memcpy(suffix, p, slen);
                    suffix[slen] = '\0';
                }
            }

            if (layer >= 0 && layer < MAX_LAYERS) {
                layer_weights_t *lw = &w->layers[layer];
                if (strcmp(suffix, "attn_norm.weight") == 0) {
                    lw->attn_norm = ptr; lw->type_attn_norm = qtype;
                } else if (strcmp(suffix, "attn_q.weight") == 0) {
                    lw->attn_q = ptr; lw->type_attn_q = qtype;
                } else if (strcmp(suffix, "attn_k.weight") == 0) {
                    lw->attn_k = ptr; lw->type_attn_k = qtype;
                } else if (strcmp(suffix, "attn_v.weight") == 0) {
                    lw->attn_v = ptr; lw->type_attn_v = qtype;
                } else if (strcmp(suffix, "attn_output.weight") == 0) {
                    lw->attn_out = ptr; lw->type_attn_out = qtype;
                } else if (strcmp(suffix, "ffn_norm.weight") == 0) {
                    lw->ffn_norm = ptr; lw->type_ffn_norm = qtype;
                } else if (strcmp(suffix, "ffn_gate.weight") == 0) {
                    lw->ffn_gate = ptr; lw->type_ffn_gate = qtype;
                } else if (strcmp(suffix, "ffn_down.weight") == 0) {
                    lw->ffn_down = ptr; lw->type_ffn_down = qtype;
                } else if (strcmp(suffix, "ffn_up.weight") == 0) {
                    lw->ffn_up = ptr; lw->type_ffn_up = qtype;
                }
            }
        }
    }

    free(tinfos);
    return 0;
}

/* ---- mmap helpers and rest of file unchanged ---- */

#ifdef _WIN32
static int map_file(model_t *m, const char *path) {
    HANDLE hFile = CreateFileA(path, GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
    if (hFile == INVALID_HANDLE_VALUE) {
        fprintf(stderr, "Failed to open %s\n", path);
        return -1;
    }
    LARGE_INTEGER size;
    if (!GetFileSizeEx(hFile, &size)) {
        CloseHandle(hFile);
        fprintf(stderr, "Failed to stat %s\n", path);
        return -1;
    }
    HANDLE hMap = CreateFileMappingA(hFile, NULL, PAGE_READONLY, 0, 0, NULL);
    if (!hMap) {
        CloseHandle(hFile);
        fprintf(stderr, "Failed to map %s\n", path);
        return -1;
    }
    void *addr = MapViewOfFile(hMap, FILE_MAP_READ, 0, 0, 0);
    if (!addr) {
        CloseHandle(hMap);
        CloseHandle(hFile);
        fprintf(stderr, "Failed to map view %s\n", path);
        return -1;
    }
    m->mmap_addr = addr;
    m->mmap_size = (size_t)size.QuadPart;
    m->win_file = hFile;
    m->win_map = hMap;
    return 0;
}

static void unmap_file(model_t *m) {
    if (m->mmap_addr) UnmapViewOfFile(m->mmap_addr);
    if (m->win_map) CloseHandle(m->win_map);
    if (m->win_file) CloseHandle(m->win_file);
    m->mmap_addr = NULL;
    m->mmap_size = 0;
}
#else
static int map_file(model_t *m, const char *path) {
    int fd = open(path, O_RDONLY);
    if (fd < 0) { perror("open"); return -1; }
    struct stat st;
    if (fstat(fd, &st) != 0) { perror("fstat"); close(fd); return -1; }
    void *addr = mmap(NULL, (size_t)st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (addr == MAP_FAILED) { perror("mmap"); close(fd); return -1; }
    close(fd);
    m->mmap_addr = addr;
    m->mmap_size = (size_t)st.st_size;
    return 0;
}

static void unmap_file(model_t *m) {
    if (m->mmap_addr) munmap(m->mmap_addr, m->mmap_size);
    m->mmap_addr = NULL;
    m->mmap_size = 0;
}
#endif

int model_load(model_t *m, const char *path, int max_seq_len) {
    memset(m, 0, sizeof(*m));
    if (map_file(m, path) != 0) return -1;
    if (parse_gguf(m, max_seq_len) != 0) {
        unmap_file(m);
        return -1;
    }
    return 0;
}

void model_free(model_t *m) {
    unmap_file(m);
}
