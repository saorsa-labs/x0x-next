/*
 * PROTOTYPE / EXPERIMENT -- NOT PRODUCTION CODE, NOT PART OF x0x.
 *
 * How much CPU does a "learned" decision cost, compared with the cryptography
 * a post-quantum network already has to do?  Single core, scalar C, -O2.
 * Build: gcc -O2 -o decision_cost decision_cost.c -lm
 */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

static double now_ns(void) {
    struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec * 1e9 + t.tv_nsec;
}
static uint32_t rs = 12345;
static inline uint32_t rnd(void) { rs ^= rs << 13; rs ^= rs >> 17; rs ^= rs << 5; return rs; }

/* 1. EWMA update of a latency estimate (fixed point) */
static inline int32_t ewma(int32_t est, int32_t sample) { return est + ((sample - est) >> 3); }

/* 2. UCB1 choice among K arms, then update (float) */
#define K 32
static double mean_[K], cnt_[K];
static int ucb_select(double t) {
    double lt = log(t + 1.0), best = -1e300; int bi = 0;
    for (int i = 0; i < K; i++) {
        double v = cnt_[i] < 1 ? 1e300 : mean_[i] + sqrt(2.0 * lt / cnt_[i]);
        if (v > best) { best = v; bi = i; }
    }
    return bi;
}

/* 3. isotonic-regression lookup: binary search over 64 monotone blocks */
static int32_t iso_x[64], iso_y[64];
static inline int32_t iso_lookup(int32_t x) {
    int lo = 0, hi = 63;
    while (lo < hi) { int mid = (lo + hi + 1) >> 1; if (iso_x[mid] <= x) lo = mid; else hi = mid - 1; }
    return iso_y[lo];
}

/* 4. tiny integer MLP: in -> hid -> out, int8 weights, ReLU */
static void mlp(const int8_t *w1, const int8_t *w2, const int8_t *in, int n_in, int n_h, int n_out, int32_t *out) {
    int32_t h[256];
    for (int j = 0; j < n_h; j++) {
        int32_t a = 0; const int8_t *w = w1 + j * n_in;
        for (int i = 0; i < n_in; i++) a += w[i] * in[i];
        h[j] = a > 0 ? (a >> 7) : 0;
    }
    for (int k = 0; k < n_out; k++) {
        int32_t a = 0; const int8_t *w = w2 + k * n_h;
        for (int j = 0; j < n_h; j++) a += w[j] * h[j];
        out[k] = a;
    }
}

/* 5. boosted trees: 50 trees, depth 4, int features */
typedef struct { uint8_t feat[15]; int16_t thr[15]; int16_t leaf[16]; } tree_t;
static tree_t trees[50];
static int32_t gbdt(const int16_t *x) {
    int32_t s = 0;
    for (int t = 0; t < 50; t++) {
        int n = 0;
        for (int d = 0; d < 4; d++) n = 2 * n + 1 + (x[trees[t].feat[n]] > trees[t].thr[n]);
        s += trees[t].leaf[n - 15];
    }
    return s;
}

int main(void) {
    volatile int64_t sink = 0;
    int8_t *w1s = malloc(32 * 64), *w2s = malloc(64 * 16), *w1l = malloc(128 * 256), *w2l = malloc(256 * 64);
    int8_t in[128]; int16_t feat[32]; int32_t out[64];
    for (int i = 0; i < 32 * 64; i++) w1s[i] = rnd(); for (int i = 0; i < 64 * 16; i++) w2s[i] = rnd();
    for (int i = 0; i < 128 * 256; i++) w1l[i] = rnd(); for (int i = 0; i < 256 * 64; i++) w2l[i] = rnd();
    for (int i = 0; i < 128; i++) in[i] = rnd(); for (int i = 0; i < 32; i++) feat[i] = rnd();
    for (int i = 0; i < 64; i++) { iso_x[i] = i * 1000; iso_y[i] = i * 7; }
    for (int t = 0; t < 50; t++) { for (int n = 0; n < 15; n++) { trees[t].feat[n] = rnd() % 32; trees[t].thr[n] = rnd(); } for (int l = 0; l < 16; l++) trees[t].leaf[l] = rnd(); }
    for (int i = 0; i < K; i++) { mean_[i] = (rnd() % 1000) / 1000.0; cnt_[i] = 1 + rnd() % 50; }

    double t0; long n;

    n = 200000000; int32_t e = 1000; t0 = now_ns();
    for (long i = 0; i < n; i++) e = ewma(e, (int32_t)(rnd() & 4095));
    printf("ewma_update_ns %.2f\n", (now_ns() - t0) / n); sink += e;

    n = 5000000; t0 = now_ns();
    for (long i = 0; i < n; i++) { int a = ucb_select((double)i); double r = (rnd() & 1023) / 1024.0; cnt_[a] += 1; mean_[a] += (r - mean_[a]) / cnt_[a]; sink += a; }
    printf("ucb32_select_update_ns %.1f\n", (now_ns() - t0) / n);

    n = 100000000; t0 = now_ns();
    for (long i = 0; i < n; i++) sink += iso_lookup((int32_t)(rnd() % 64000));
    printf("isotonic_lookup_ns %.2f\n", (now_ns() - t0) / n);

    n = 2000000; t0 = now_ns();
    for (long i = 0; i < n; i++) { feat[i & 31] = rnd(); sink += gbdt(feat); }
    printf("gbdt_50x4_ns %.1f\n", (now_ns() - t0) / n);

    n = 1000000; t0 = now_ns();
    for (long i = 0; i < n; i++) { in[i & 31] = rnd(); mlp(w1s, w2s, in, 32, 64, 16, out); sink += out[0]; }
    printf("mlp_int8_32x64x16_3072w_ns %.1f\n", (now_ns() - t0) / n);

    n = 50000; t0 = now_ns();
    for (long i = 0; i < n; i++) { in[i & 127] = rnd(); mlp(w1l, w2l, in, 128, 256, 64, out); sink += out[0]; }
    printf("mlp_int8_128x256x64_49152w_ns %.1f\n", (now_ns() - t0) / n);

    fprintf(stderr, "(sink %lld)\n", (long long)sink);
    return 0;
}
