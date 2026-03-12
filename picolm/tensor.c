#include "tensor.h"
#include <math.h>
#include <string.h>
#include <stdio.h>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <pthread.h>
#endif

/* ---- Scratch buffer (kept for dequantize_row in model.c) ---- */

static float *scratch_buf = NULL;
static int    scratch_size = 0;

void tensor_init_scratch(float *buf, int size) {
    scratch_buf  = buf;
    scratch_size = size;
}

/* ---- Threading for matmul ---- */

static int n_threads = 1;

void tensor_set_threads(int t) {
    if (t < 1) t = 1;
    if (t > MAX_THREADS) t = MAX_THREADS;
    n_threads = t;
}

int tensor_get_threads(void) {
    return n_threads;
}

typedef struct {
    float       *out;
    const float *x;
    const char  *W;
    size_t       row_bytes;
    int          n;        /* input dimension */
    int          start;    /* first output row */
    int          end;      /* one past last output row */
    gguf_type_t  qtype;
} matmul_task_t;

static
#ifdef _WIN32
DWORD WINAPI
#else
void *
#endif
matmul_worker(void *arg) {
    matmul_task_t *t = (matmul_task_t *)arg;
    for (int i = t->start; i < t->end; i++) {
        t->out[i] = vec_dot(t->W + (size_t)i * t->row_bytes,
                            t->x, t->n, t->qtype);
    }
#ifdef _WIN32
    return 0;
#else
    return NULL;
#endif
}

void matmul(float *out, const float *x, const void *W, int n, int d, gguf_type_t qtype) {
    size_t row_bytes = gguf_type_row_size(qtype, n);
    const char *wptr = (const char *)W;

    if (n_threads <= 1 || d < 4) {
        for (int i = 0; i < d; i++) {
            out[i] = vec_dot(wptr + (size_t)i * row_bytes, x, n, qtype);
        }
        return;
    }

    int nt = n_threads;
    if (nt > d) nt = d;

    matmul_task_t tasks[MAX_THREADS];
#ifdef _WIN32
    HANDLE threads[MAX_THREADS];
#else
    pthread_t threads[MAX_THREADS];
#endif

    int rows_per = d / nt;
    int extra = d % nt;
    int row = 0;

    for (int t = 0; t < nt; t++) {
        tasks[t].out = out;
        tasks[t].x = x;
        tasks[t].W = wptr;
        tasks[t].row_bytes = row_bytes;
        tasks[t].n = n;
        tasks[t].qtype = qtype;
        tasks[t].start = row;
        row += rows_per + (t < extra ? 1 : 0);
        tasks[t].end = row;
    }

    for (int t = 1; t < nt; t++) {
#ifdef _WIN32
        threads[t] = CreateThread(NULL, 0, matmul_worker, &tasks[t], 0, NULL);
#else
        pthread_create(&threads[t], NULL, matmul_worker, &tasks[t]);
#endif
    }

    matmul_worker(&tasks[0]);

    for (int t = 1; t < nt; t++) {
#ifdef _WIN32
        WaitForSingleObject(threads[t], INFINITE);
        CloseHandle(threads[t]);
#else
        pthread_join(threads[t], NULL);
#endif
    }
}

/* ================================================================
 * SIMD-accelerated basic operations
 * ================================================================ */

void rmsnorm(float *out, const float *x, const float *weight, int size) {
    float ss = 0.0f;

#ifdef PICOLM_NEON
    float32x4_t acc = vdupq_n_f32(0);
    int i = 0;
    for (; i + 3 < size; i += 4) {
        float32x4_t v = vld1q_f32(x + i);
        acc = vmlaq_f32(acc, v, v);
    }
    ss = vaddvq_f32_compat(acc);
    for (; i < size; i++) ss += x[i] * x[i];
#elif defined(PICOLM_SSE2)
    __m128 acc = _mm_setzero_ps();
    int i = 0;
    for (; i + 3 < size; i += 4) {
        __m128 v = _mm_loadu_ps(x + i);
        acc = _mm_add_ps(acc, _mm_mul_ps(v, v));
    }
    ss = hsum_sse(acc);
    for (; i < size; i++) ss += x[i] * x[i];
#else
    for (int i = 0; i < size; i++) ss += x[i] * x[i];
#endif

    ss = 1.0f / sqrtf(ss / (float)size + 1e-5f);

#ifdef PICOLM_NEON
    float32x4_t scale = vdupq_n_f32(ss);
    i = 0;
    for (; i + 3 < size; i += 4) {
        float32x4_t v = vld1q_f32(x + i);
        float32x4_t w = vld1q_f32(weight + i);
        vst1q_f32(out + i, vmulq_f32(vmulq_f32(v, scale), w));
    }
    for (; i < size; i++) out[i] = x[i] * ss * weight[i];
#elif defined(PICOLM_SSE2)
    __m128 scale = _mm_set1_ps(ss);
    i = 0;
    for (; i + 3 < size; i += 4) {
        __m128 v = _mm_loadu_ps(x + i);
        __m128 w = _mm_loadu_ps(weight + i);
        _mm_storeu_ps(out + i, _mm_mul_ps(_mm_mul_ps(v, scale), w));
    }
    for (; i < size; i++) out[i] = x[i] * ss * weight[i];
#else
    for (int i = 0; i < size; i++) out[i] = x[i] * ss * weight[i];
#endif
}

void softmax(float *x, int size) {
    float max_val = x[0];
    for (int i = 1; i < size; i++) {
        if (x[i] > max_val) max_val = x[i];
    }
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        x[i] = expf(x[i] - max_val);
        sum += x[i];
    }
    float inv = 1.0f / sum;

#ifdef PICOLM_NEON
    float32x4_t inv_v = vdupq_n_f32(inv);
    int i = 0;
    for (; i + 3 < size; i += 4) {
        vst1q_f32(x + i, vmulq_f32(vld1q_f32(x + i), inv_v));
    }
    for (; i < size; i++) x[i] *= inv;
#elif defined(PICOLM_SSE2)
    __m128 inv_v = _mm_set1_ps(inv);
    int i = 0;
    for (; i + 3 < size; i += 4) {
        _mm_storeu_ps(x + i, _mm_mul_ps(_mm_loadu_ps(x + i), inv_v));
    }
    for (; i < size; i++) x[i] *= inv;
#else
    for (int i = 0; i < size; i++) x[i] *= inv;
#endif
}

/* Rotary position encoding using pre-computed cos/sin tables */
void rope(float *q, float *k, int head_dim, int n_heads, int n_kv_heads,
          const float *cos_pos, const float *sin_pos) {
    int half = head_dim / 2;

    /* Apply RoPE to all query heads */
    for (int h = 0; h < n_heads; h++) {
        float *qh = q + h * head_dim;
#ifdef PICOLM_NEON
        int i = 0;
        for (; i + 3 < half; i += 4) {
            /* Load pairs: (q0,q1), (q2,q3), ... as interleaved */
            float32x4x2_t qv = vld2q_f32(qh + i * 2);
            float32x4_t cv = vld1q_f32(cos_pos + i);
            float32x4_t sv = vld1q_f32(sin_pos + i);
            /* q_even = q0*cos - q1*sin, q_odd = q0*sin + q1*cos */
            float32x4_t new_even = vmlsq_f32(vmulq_f32(qv.val[0], cv), qv.val[1], sv);
            float32x4_t new_odd  = vmlaq_f32(vmulq_f32(qv.val[0], sv), qv.val[1], cv);
            float32x4x2_t result = {{ new_even, new_odd }};
            vst2q_f32(qh + i * 2, result);
        }
        for (; i < half; i++) {
            float q0 = qh[i * 2];
            float q1 = qh[i * 2 + 1];
            qh[i * 2]     = q0 * cos_pos[i] - q1 * sin_pos[i];
            qh[i * 2 + 1] = q0 * sin_pos[i] + q1 * cos_pos[i];
        }
#else
        for (int i = 0; i < half; i++) {
            float q0 = qh[i * 2];
            float q1 = qh[i * 2 + 1];
            qh[i * 2]     = q0 * cos_pos[i] - q1 * sin_pos[i];
            qh[i * 2 + 1] = q0 * sin_pos[i] + q1 * cos_pos[i];
        }
#endif
    }

    /* Apply RoPE to all KV heads */
    for (int h = 0; h < n_kv_heads; h++) {
        float *kh = k + h * head_dim;
#ifdef PICOLM_NEON
        int i = 0;
        for (; i + 3 < half; i += 4) {
            float32x4x2_t kv = vld2q_f32(kh + i * 2);
            float32x4_t cv = vld1q_f32(cos_pos + i);
            float32x4_t sv = vld1q_f32(sin_pos + i);
            float32x4_t new_even = vmlsq_f32(vmulq_f32(kv.val[0], cv), kv.val[1], sv);
            float32x4_t new_odd  = vmlaq_f32(vmulq_f32(kv.val[0], sv), kv.val[1], cv);
            float32x4x2_t result = {{ new_even, new_odd }};
            vst2q_f32(kh + i * 2, result);
        }
        for (; i < half; i++) {
            float k0 = kh[i * 2];
            float k1 = kh[i * 2 + 1];
            kh[i * 2]     = k0 * cos_pos[i] - k1 * sin_pos[i];
            kh[i * 2 + 1] = k0 * sin_pos[i] + k1 * cos_pos[i];
        }
#else
        for (int i = 0; i < half; i++) {
            float k0 = kh[i * 2];
            float k1 = kh[i * 2 + 1];
            kh[i * 2]     = k0 * cos_pos[i] - k1 * sin_pos[i];
            kh[i * 2 + 1] = k0 * sin_pos[i] + k1 * cos_pos[i];
        }
#endif
    }
}

void silu(float *x, int size) {
    for (int i = 0; i < size; i++) {
        x[i] = x[i] / (1.0f + expf(-x[i]));
    }
}

void elemwise_mul(float *out, const float *a, const float *b, int size) {
#ifdef PICOLM_NEON
    int i = 0;
    for (; i + 3 < size; i += 4) {
        vst1q_f32(out + i, vmulq_f32(vld1q_f32(a + i), vld1q_f32(b + i)));
    }
    for (; i < size; i++) out[i] = a[i] * b[i];
#elif defined(PICOLM_SSE2)
    int i = 0;
    for (; i + 3 < size; i += 4) {
        _mm_storeu_ps(out + i, _mm_mul_ps(_mm_loadu_ps(a + i), _mm_loadu_ps(b + i)));
    }
    for (; i < size; i++) out[i] = a[i] * b[i];
#else
    for (int i = 0; i < size; i++) out[i] = a[i] * b[i];
#endif
}

void vec_add(float *a, const float *b, int size) {
#ifdef PICOLM_NEON
    int i = 0;
    for (; i + 3 < size; i += 4) {
        vst1q_f32(a + i, vaddq_f32(vld1q_f32(a + i), vld1q_f32(b + i)));
    }
    for (; i < size; i++) a[i] += b[i];
#elif defined(PICOLM_SSE2)
    int i = 0;
    for (; i + 3 < size; i += 4) {
        _mm_storeu_ps(a + i, _mm_add_ps(_mm_loadu_ps(a + i), _mm_loadu_ps(b + i)));
    }
    for (; i < size; i++) a[i] += b[i];
#else
    for (int i = 0; i < size; i++) a[i] += b[i];
#endif
}
