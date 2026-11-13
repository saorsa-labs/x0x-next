/* PROTOTYPE / EXPERIMENT -- NOT PRODUCTION CODE, NOT PART OF x0x.
 *
 * What does it cost a node to check a written policy?
 *
 * Part 1, the default form (one weight per step). Every check is read off the weights:
 *   directions are signs, trade-off limits are inequalities. No table is built.
 *   Timed here for 8 features of 16 levels each (120 step weights and a bias).
 *
 * Part 2, the exception (a small network with hidden units, so features can interact).
 *   The node builds the table: the policy's answer on EVERY input. Timed for a
 *   6-input network with 8 and 4 hidden units (int8 weights, int16 biases, 64-bit
 *   accumulators) on 4, 5 and 6 features of 16 levels each, then:
 *     - the direction check on every neighbouring pair of cells,
 *     - the gate check,
 *     - the tie-break limit (band of each group over the two tie-break features),
 *     - the diff against a second table (cells that changed).
 *
 * Single core, scalar C.   gcc -O2 -o atlas_cost atlas_cost.c && ./atlas_cost
 */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>

#define NIN 6
#define H1 8
#define H2 4

static int8_t W1[H1][NIN], W2[H2][H1], W3[H2];
static int16_t b1[H1], b2[H2], b3;

static inline int64_t net(const int64_t *x) {
    int64_t h1[H1], h2[H2], y = b3;
    for (int i = 0; i < H1; i++) { int64_t s = b1[i]; for (int j = 0; j < NIN; j++) s += (int64_t)W1[i][j] * x[j]; h1[i] = s > 0 ? s : 0; }
    for (int i = 0; i < H2; i++) { int64_t s = b2[i]; for (int j = 0; j < H1; j++) s += (int64_t)W2[i][j] * h1[j]; h2[i] = s > 0 ? s : 0; }
    for (int i = 0; i < H2; i++) y += (int64_t)W3[i] * h2[i];
    return y > 0 ? y : 0;
}

static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + 1e-9 * t.tv_nsec; }

/* ---- part 1: one weight per step ------------------------------------------------- */
#define SF 8          /* features */
#define SL 16         /* levels   */
static int32_t SW[SF][SL - 1];
static int SDIR[SF] = {-1, -1, +1, +1, +1, -1, +1, +1};
static int SROLE[SF] = {1, 1, 0, 2, 2, 1, 0, 2};   /* 1 primary, 2 tie-break, 0 other */

static int64_t gcd64(int64_t a, int64_t b) { while (b) { int64_t t = a % b; a = b; b = t; } return a < 0 ? -a : a; }

static int read_checks(void) {
    int64_t pmin = INT64_MAX, pmax = 0, swing = 0, g = 0;
    for (int f = 0; f < SF; f++)
        for (int k = 0; k < SL - 1; k++) {
            int64_t w = SW[f][k], a = w < 0 ? -w : w;
            if (w * SDIR[f] < 0) return 0;                       /* wrong direction */
            if (SROLE[f] == 1) { if (a == 0) return 0; if (a < pmin) pmin = a; if (a > pmax) pmax = a; }
            else if (SROLE[f] == 2) swing += a;
            if (SROLE[f] != 2) g = gcd64(g, a);
        }
    if (pmax > 32 * pmin) return 0;                              /* one step worth too much */
    if (swing && swing >= g) return 0;                           /* tie-break could outweigh a real step */
    return 1;
}

int main(void) {
    /* ---- part 1 ---- */
    for (int f = 0; f < SF; f++) for (int k = 0; k < SL - 1; k++)
        SW[f][k] = SDIR[f] * (SROLE[f] == 2 ? (k < 1 ? 1 : 0) : 8 * (1 + (k + f) % 4));
    double t0 = now(); long ok = 0; int reps = 2000000;
    for (int r = 0; r < reps; r++) { SW[0][r % 15] ^= 0; ok += read_checks(); }
    double t1 = now();
    printf("ONE WEIGHT PER STEP: %d features x %d levels = %.0f possible inputs, %d weights. All checks read off the weights: %.0f ns per policy (passed %ld of %d). No table.\n",
           SF, SL, 4294967296.0, SF * (SL - 1) + 1, 1e9 * (t1 - t0) / reps, ok, reps);

    /* ---- part 1b: scoring one candidate with step weights (cumulative tables, 8 lookups and adds) ---- */
    {
        static int32_t CUM[SF][SL];
        for (int f = 0; f < SF; f++) { CUM[f][0] = 0; for (int k = 1; k < SL; k++) CUM[f][k] = CUM[f][k - 1] + SW[f][k - 1]; }
        uint32_t s = 2463534242u; int64_t acc = 0; size_t n = 50000000;
        double e0 = now();
        for (size_t i = 0; i < n; i++) {
            s ^= s << 13; s ^= s >> 17; s ^= s << 5;                 /* the input: 8 levels from one random word */
            int32_t y = 1000;
            for (int f = 0; f < SF; f++) y += CUM[f][(s >> (4 * f)) & 15];
            acc += y;
        }
        double e1 = now();
        printf("ONE WEIGHT PER STEP: scoring one candidate (8 features) takes %.1f ns, including generating the input (checksum %lld).\n", 1e9 * (e1 - e0) / n, (long long)acc);
    }

    /* ---- part 2 ---- */
    uint32_t r = 12345;
    for (int i = 0; i < H1; i++) { b1[i] = 40; for (int j = 0; j < NIN; j++) { r = r * 1664525u + 1013904223u; W1[i][j] = (int8_t)((r >> 24) % 15) - 7; } }
    for (int i = 0; i < H2; i++) { b2[i] = 3; for (int j = 0; j < H1; j++) { r = r * 1664525u + 1013904223u; W2[i][j] = (int8_t)((r >> 24) % 9) - 4; } }
    for (int i = 0; i < H2; i++) W3[i] = (int8_t)(i + 1);
    b3 = 0;
    for (int F = 4; F <= NIN; F++) {                 /* 2^16, 2^20, 2^24 cells */
        size_t n = (size_t)1 << (4 * F);
        int64_t *S = malloc(n * sizeof(int64_t)), *T = malloc(n * sizeof(int64_t));
        double a0 = now();
        for (size_t c = 0; c < n; c++) {
            int64_t x[NIN] = {0};
            for (int f = 0; f < F; f++) x[f] = (c >> (4 * f)) & 15;
            S[c] = net(x);
        }
        double a1 = now();
        size_t viol = 0;
        for (int f = 0; f < F; f++) {
            size_t stride = (size_t)1 << (4 * f);
            for (size_t c = 0; c < n; c++)
                if (((c >> (4 * f)) & 15) < 15 && S[c + stride] < S[c]) viol++;
        }
        double a2 = now();
        /* gate: feature 2 below level 2 must score zero, everything else above zero */
        size_t gate_bad = 0;
        for (size_t c = 0; c < n; c++) { int inel = ((c >> 8) & 15) < 2; if (inel ? S[c] != 0 : S[c] <= 0) gate_bad++; }
        double a3 = now();
        /* tie-break limit: features 0 and 1 are the tie-break pair here (the low 8 bits),
           so each group is 256 consecutive cells; record each group's band */
        size_t groups = n >> 8, band_bad = 0; int64_t prev_hi = INT64_MIN;
        int64_t *lo = malloc(groups * sizeof(int64_t)), *hi = malloc(groups * sizeof(int64_t));
        for (size_t g = 0; g < groups; g++) {
            int64_t l = INT64_MAX, h = INT64_MIN;
            for (size_t c = g << 8; c < (g + 1) << 8; c++) { if (S[c] < l) l = S[c]; if (S[c] > h) h = S[c]; }
            lo[g] = l; hi[g] = h;
        }
        for (size_t g = 0; g + 1 < groups; g++) if (hi[g] >= lo[g + 1] && !(lo[g] == lo[g + 1] && hi[g] == hi[g + 1])) band_bad++;
        (void)prev_hi;
        double a4 = now();
        /* diff against a second table (here: the same weights with one changed) */
        W1[0][1] += 1;
        for (size_t c = 0; c < n; c++) { int64_t x[NIN] = {0}; for (int f = 0; f < F; f++) x[f] = (c >> (4 * f)) & 15; T[c] = net(x); }
        W1[0][1] -= 1;
        size_t changed = 0; for (size_t c = 0; c < n; c++) if (S[c] != T[c]) changed++;
        double a5 = now();
        printf("NETWORK WITH HIDDEN UNITS, %d features x 16 levels = %9zu cells: table %.3f s (%.1f ns per cell); directions %.3f s; gate %.3f s; tie-break bands %.3f s; second table and diff %.3f s; everything %.3f s; memory for two tables %zu MB\n",
               F, n, a1 - a0, 1e9 * (a1 - a0) / n, a2 - a1, a3 - a2, a4 - a3, a5 - a4, a5 - a0, 2 * n * 8 / (1 << 20));
        (void)viol; (void)gate_bad; (void)band_bad; (void)changed;
        fprintf(stderr, "  (arbitrary weights: %zu wrong-way pairs, %zu gate failures, %zu overlapping bands, %zu cells changed)\n", viol, gate_bad, band_bad, changed);
        free(S); free(T); free(lo); free(hi);
    }
    return 0;
}
