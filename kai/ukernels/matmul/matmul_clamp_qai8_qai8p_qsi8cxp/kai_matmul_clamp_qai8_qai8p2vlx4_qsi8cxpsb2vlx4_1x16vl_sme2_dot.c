//
// SPDX-FileCopyrightText: Copyright 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//
// EXP-3: RHS-reuse fix. Process TILE_M=4 rows simultaneously per inner loop pass
// so RHS is loaded once per k-group and reused across 4 rows (vs 32× in EXP-2).
// LHS x8p2vlx4 and RHS qsi8cxpsb2vlx4 share 128-byte k-group stride on M4
// (m_step=n_step=32, kr=4). Total RHS loads: (m_step/4) × k_groups vs m_step × k_groups.

#if (!defined(__aarch64__) || !defined(__ARM_FEATURE_SVE2)) && !defined(_M_ARM64)
#error This file must be compiled for AArch64, FEAT_SVE2.
#else  // Architectural features check.

#include "kai_matmul_clamp_qai8_qai8p2vlx4_qsi8cxpsb2vlx4_1x16vl_sme2_dot.h"

#include <stddef.h>
#include <stdint.h>

#include "kai/kai_common.h"

static const size_t kai_mr = 2;
static const size_t kai_nr = 2;
static const size_t kai_kr = 4;
static const size_t kai_sr = 1;

#define TILE_M 4

size_t kai_get_m_step_matmul_clamp_qai8_qai8p2vlx4_qsi8cxpsb2vlx4_1x16vl_sme2_dot(void) {
    return kai_mr * kai_get_sme_vector_length_u8() / kai_kr;
}

size_t kai_get_n_step_matmul_clamp_qai8_qai8p2vlx4_qsi8cxpsb2vlx4_1x16vl_sme2_dot(void) {
    return kai_nr * kai_get_sme_vector_length_u8() / kai_kr;
}

size_t kai_get_mr_matmul_clamp_qai8_qai8p2vlx4_qsi8cxpsb2vlx4_1x16vl_sme2_dot(void) {
    return kai_mr * kai_get_sme_vector_length_u8() / kai_kr;
}

size_t kai_get_nr_matmul_clamp_qai8_qai8p2vlx4_qsi8cxpsb2vlx4_1x16vl_sme2_dot(void) {
    return kai_nr * kai_get_sme_vector_length_u8() / kai_kr;
}

size_t kai_get_kr_matmul_clamp_qai8_qai8p2vlx4_qsi8cxpsb2vlx4_1x16vl_sme2_dot(void) {
    return kai_kr;
}

size_t kai_get_sr_matmul_clamp_qai8_qai8p2vlx4_qsi8cxpsb2vlx4_1x16vl_sme2_dot(void) {
    return kai_sr;
}

size_t kai_get_lhs_packed_offset_matmul_clamp_qai8_qai8p2vlx4_qsi8cxpsb2vlx4_1x16vl_sme2_dot(
    size_t m_idx, size_t k) {
    KAI_ASSUME(m_idx % kai_get_m_step_matmul_clamp_qai8_qai8p2vlx4_qsi8cxpsb2vlx4_1x16vl_sme2_dot() == 0);
    return m_idx * kai_roundup(k, kai_kr) * sizeof(int8_t);
}

static size_t kai_get_rhs_packed_stride(size_t n_step_val, size_t k) {
    return n_step_val * (sizeof(int32_t) + kai_roundup(k, kai_kr) * sizeof(int8_t) + sizeof(float));
}

size_t kai_get_rhs_packed_offset_matmul_clamp_qai8_qai8p2vlx4_qsi8cxpsb2vlx4_1x16vl_sme2_dot(
    size_t n_idx, size_t k) {
    const size_t n_step_val =
        kai_get_n_step_matmul_clamp_qai8_qai8p2vlx4_qsi8cxpsb2vlx4_1x16vl_sme2_dot();
    KAI_ASSUME(n_idx % n_step_val == 0);
    return (n_idx / n_step_val) * kai_get_rhs_packed_stride(n_step_val, k);
}

size_t kai_get_dst_offset_matmul_clamp_qai8_qai8p2vlx4_qsi8cxpsb2vlx4_1x16vl_sme2_dot(
    size_t m_idx, size_t n_idx, size_t dst_stride_row) {
    KAI_ASSUME(m_idx % kai_get_m_step_matmul_clamp_qai8_qai8p2vlx4_qsi8cxpsb2vlx4_1x16vl_sme2_dot() == 0);
    KAI_ASSUME(n_idx % kai_get_n_step_matmul_clamp_qai8_qai8p2vlx4_qsi8cxpsb2vlx4_1x16vl_sme2_dot() == 0);
    return m_idx * dst_stride_row + n_idx * sizeof(int8_t);
}

size_t kai_get_dst_size_matmul_clamp_qai8_qai8p2vlx4_qsi8cxpsb2vlx4_1x16vl_sme2_dot(size_t m, size_t n) {
    return m * n * sizeof(int8_t);
}

void kai_run_matmul_clamp_qai8_qai8p2vlx4_qsi8cxpsb2vlx4_1x16vl_sme2_dot(
    size_t m, size_t n, size_t k, const void* lhs_packed, const void* rhs_packed,
    void* dst, size_t dst_stride_row, size_t dst_stride_col,
    const struct kai_matmul_requantize32_params* params) {
    KAI_UNUSED(dst_stride_col);

    const size_t m_step_val =
        kai_get_m_step_matmul_clamp_qai8_qai8p2vlx4_qsi8cxpsb2vlx4_1x16vl_sme2_dot();
    const size_t n_step_val =
        kai_get_n_step_matmul_clamp_qai8_qai8p2vlx4_qsi8cxpsb2vlx4_1x16vl_sme2_dot();
    const size_t k_rounded = kai_roundup(k, kai_kr);
    const size_t vl32 = kai_get_sme_vector_length_u8() / sizeof(int32_t);
    const uint64_t k_groups_total = (uint64_t)(k_rounded / kai_kr);
    const uint64_t lhs_kgrp_stride = (uint64_t)(m_step_val * kai_kr);
    const uint64_t rhs_kgrp_stride = (uint64_t)(n_step_val * kai_kr);

    kai_commit_za();

    __asm__ volatile(".inst 0xd503477f\n" ::: "memory");  // smstart

    const int32_t ozp       = params->output_zero_point;
    const int32_t clamp_min = (int32_t)params->min_value;
    const int32_t clamp_max = (int32_t)params->max_value;

    for (size_t m_blk = 0; m_blk < m; m_blk += m_step_val) {
        const size_t m_count = (m_blk + m_step_val <= m) ? m_step_val : (m - m_blk);
        const int8_t* lhs_blk = (const int8_t*)lhs_packed + m_blk * k_rounded;

        for (size_t n_blk = 0; n_blk < n; n_blk += n_step_val) {
            const size_t n_count = (n_blk + n_step_val <= n) ? n_step_val : (n - n_blk);
            const int8_t* rhs_blk = (const int8_t*)rhs_packed +
                (n_blk / n_step_val) * kai_get_rhs_packed_stride(n_step_val, k);
            const int32_t* rhs_bias  = (const int32_t*)rhs_blk;
            const int8_t*  rhs_data  = (const int8_t*)(rhs_bias + n_step_val);
            const float*   rhs_scale = (const float*)(rhs_data + n_step_val * k_rounded);

            const uint64_t nc_lo = (n_count < vl32) ? n_count : vl32;
            const uint64_t nc_hi = (n_count > vl32) ? (n_count - vl32) : 0;

            // Process TILE_M rows at a time: RHS loaded once per k-group, reused across rows.
            for (size_t r_tile = 0; r_tile < m_count; r_tile += TILE_M) {
                const size_t tile_rows = (r_tile + TILE_M <= m_count) ? TILE_M : (m_count - r_tile);

                // Tail rows alias to row 0 (same lhs → same acc result, written to same dst).
                const int8_t* lp0 = lhs_blk + (r_tile + 0) * kai_kr;
                const int8_t* lp1 = (tile_rows > 1) ? lhs_blk + (r_tile + 1) * kai_kr : lp0;
                const int8_t* lp2 = (tile_rows > 2) ? lhs_blk + (r_tile + 2) * kai_kr : lp0;
                const int8_t* lp3 = (tile_rows > 3) ? lhs_blk + (r_tile + 3) * kai_kr : lp0;

                int8_t* dp0 = (int8_t*)dst + (m_blk + r_tile + 0) * dst_stride_row + n_blk;
                int8_t* dp1 = (tile_rows > 1) ?
                    (int8_t*)dst + (m_blk + r_tile + 1) * dst_stride_row + n_blk : dp0;
                int8_t* dp2 = (tile_rows > 2) ?
                    (int8_t*)dst + (m_blk + r_tile + 2) * dst_stride_row + n_blk : dp0;
                int8_t* dp3 = (tile_rows > 3) ?
                    (int8_t*)dst + (m_blk + r_tile + 3) * dst_stride_row + n_blk : dp0;

                const int8_t* rp  = rhs_data;
                uint64_t       cnt = k_groups_total;

                __asm__ volatile(
                    "ptrue p0.b\n"
                    // Initialize accumulators from bias (same bias for all rows).
                    "mov x9, %[bias]\n"
                    "ld1w {z0.s}, p0/z, [x9]\n"
                    "ld1w {z1.s}, p0/z, [x9, #1, MUL VL]\n"
                    "mov z2.d, z0.d\n"
                    "mov z3.d, z1.d\n"
                    "mov z4.d, z0.d\n"
                    "mov z5.d, z1.d\n"
                    "mov z6.d, z0.d\n"
                    "mov z7.d, z1.d\n"
                    // k-group loop: load RHS once, SDOT into all 4 rows.
                    "cbz %[cnt], 1f\n"
                    "0:\n"
                    "ld1b {z8.b}, p0/z, [%[rp]]\n"
                    "ld1b {z9.b}, p0/z, [%[rp], #1, MUL VL]\n"
                    "ldr w14, [%[lp0]]\n"
                    "dup z10.s, w14\n"
                    "ldr w14, [%[lp1]]\n"
                    "dup z11.s, w14\n"
                    "ldr w14, [%[lp2]]\n"
                    "dup z12.s, w14\n"
                    "ldr w14, [%[lp3]]\n"
                    "dup z13.s, w14\n"
                    "sdot z0.s, z8.b, z10.b\n"
                    "sdot z1.s, z9.b, z10.b\n"
                    "sdot z2.s, z8.b, z11.b\n"
                    "sdot z3.s, z9.b, z11.b\n"
                    "sdot z4.s, z8.b, z12.b\n"
                    "sdot z5.s, z9.b, z12.b\n"
                    "sdot z6.s, z8.b, z13.b\n"
                    "sdot z7.s, z9.b, z13.b\n"
                    "add %[lp0], %[lp0], %[ls]\n"
                    "add %[lp1], %[lp1], %[ls]\n"
                    "add %[lp2], %[lp2], %[ls]\n"
                    "add %[lp3], %[lp3], %[ls]\n"
                    "add %[rp], %[rp], %[rs]\n"
                    "subs %[cnt], %[cnt], #1\n"
                    "b.ne 0b\n"
                    "1:\n"
                    // Requantize and store: load scale once, apply to all 4 rows.
                    "mov x9, %[sp]\n"
                    "ld1w {z8.s}, p0/z, [x9]\n"
                    "ld1w {z9.s}, p0/z, [x9, #1, MUL VL]\n"
                    "whilelt p1.s, xzr, %[nclo]\n"
                    // row 0 (z0, z1)
                    "scvtf z0.s, p0/m, z0.s\n"
                    "scvtf z1.s, p0/m, z1.s\n"
                    "fmul z0.s, p0/m, z0.s, z8.s\n"
                    "fmul z1.s, p0/m, z1.s, z9.s\n"
                    "frinta z0.s, p0/m, z0.s\n"
                    "frinta z1.s, p0/m, z1.s\n"
                    "fcvtzs z0.s, p0/m, z0.s\n"
                    "fcvtzs z1.s, p0/m, z1.s\n"
                    "dup z10.s, %w[ozp]\n"
                    "add z0.s, z0.s, z10.s\n"
                    "add z1.s, z1.s, z10.s\n"
                    "dup z10.s, %w[mn]\n"
                    "smax z0.s, p0/m, z0.s, z10.s\n"
                    "smax z1.s, p0/m, z1.s, z10.s\n"
                    "dup z10.s, %w[mx]\n"
                    "smin z0.s, p0/m, z0.s, z10.s\n"
                    "smin z1.s, p0/m, z1.s, z10.s\n"
                    "st1b {z0.s}, p1, [%[dp0]]\n"
                    "cbz %[nchi], 10f\n"
                    "whilelt p2.s, xzr, %[nchi]\n"
                    "add x9, %[dp0], %[vl32]\n"
                    "st1b {z1.s}, p2, [x9]\n"
                    "10:\n"
                    // row 1 (z2, z3) — scale z8/z9 already loaded
                    "scvtf z2.s, p0/m, z2.s\n"
                    "scvtf z3.s, p0/m, z3.s\n"
                    "fmul z2.s, p0/m, z2.s, z8.s\n"
                    "fmul z3.s, p0/m, z3.s, z9.s\n"
                    "frinta z2.s, p0/m, z2.s\n"
                    "frinta z3.s, p0/m, z3.s\n"
                    "fcvtzs z2.s, p0/m, z2.s\n"
                    "fcvtzs z3.s, p0/m, z3.s\n"
                    "dup z10.s, %w[ozp]\n"
                    "add z2.s, z2.s, z10.s\n"
                    "add z3.s, z3.s, z10.s\n"
                    "dup z10.s, %w[mn]\n"
                    "smax z2.s, p0/m, z2.s, z10.s\n"
                    "smax z3.s, p0/m, z3.s, z10.s\n"
                    "dup z10.s, %w[mx]\n"
                    "smin z2.s, p0/m, z2.s, z10.s\n"
                    "smin z3.s, p0/m, z3.s, z10.s\n"
                    "st1b {z2.s}, p1, [%[dp1]]\n"
                    "cbz %[nchi], 11f\n"
                    "add x9, %[dp1], %[vl32]\n"
                    "st1b {z3.s}, p2, [x9]\n"
                    "11:\n"
                    // row 2 (z4, z5)
                    "scvtf z4.s, p0/m, z4.s\n"
                    "scvtf z5.s, p0/m, z5.s\n"
                    "fmul z4.s, p0/m, z4.s, z8.s\n"
                    "fmul z5.s, p0/m, z5.s, z9.s\n"
                    "frinta z4.s, p0/m, z4.s\n"
                    "frinta z5.s, p0/m, z5.s\n"
                    "fcvtzs z4.s, p0/m, z4.s\n"
                    "fcvtzs z5.s, p0/m, z5.s\n"
                    "dup z10.s, %w[ozp]\n"
                    "add z4.s, z4.s, z10.s\n"
                    "add z5.s, z5.s, z10.s\n"
                    "dup z10.s, %w[mn]\n"
                    "smax z4.s, p0/m, z4.s, z10.s\n"
                    "smax z5.s, p0/m, z5.s, z10.s\n"
                    "dup z10.s, %w[mx]\n"
                    "smin z4.s, p0/m, z4.s, z10.s\n"
                    "smin z5.s, p0/m, z5.s, z10.s\n"
                    "st1b {z4.s}, p1, [%[dp2]]\n"
                    "cbz %[nchi], 12f\n"
                    "add x9, %[dp2], %[vl32]\n"
                    "st1b {z5.s}, p2, [x9]\n"
                    "12:\n"
                    // row 3 (z6, z7)
                    "scvtf z6.s, p0/m, z6.s\n"
                    "scvtf z7.s, p0/m, z7.s\n"
                    "fmul z6.s, p0/m, z6.s, z8.s\n"
                    "fmul z7.s, p0/m, z7.s, z9.s\n"
                    "frinta z6.s, p0/m, z6.s\n"
                    "frinta z7.s, p0/m, z7.s\n"
                    "fcvtzs z6.s, p0/m, z6.s\n"
                    "fcvtzs z7.s, p0/m, z7.s\n"
                    "dup z10.s, %w[ozp]\n"
                    "add z6.s, z6.s, z10.s\n"
                    "add z7.s, z7.s, z10.s\n"
                    "dup z10.s, %w[mn]\n"
                    "smax z6.s, p0/m, z6.s, z10.s\n"
                    "smax z7.s, p0/m, z7.s, z10.s\n"
                    "dup z10.s, %w[mx]\n"
                    "smin z6.s, p0/m, z6.s, z10.s\n"
                    "smin z7.s, p0/m, z7.s, z10.s\n"
                    "st1b {z6.s}, p1, [%[dp3]]\n"
                    "cbz %[nchi], 13f\n"
                    "add x9, %[dp3], %[vl32]\n"
                    "st1b {z7.s}, p2, [x9]\n"
                    "13:\n"
                    : [lp0] "+r"(lp0), [lp1] "+r"(lp1), [lp2] "+r"(lp2), [lp3] "+r"(lp3),
                      [rp]  "+r"(rp),  [cnt] "+r"(cnt)
                    : [bias] "r"(rhs_bias),
                      [sp]   "r"(rhs_scale),
                      [dp0]  "r"(dp0),  [dp1]  "r"(dp1),  [dp2]  "r"(dp2),  [dp3]  "r"(dp3),
                      [ls]   "r"(lhs_kgrp_stride),
                      [rs]   "r"(rhs_kgrp_stride),
                      [ozp]  "r"(ozp), [mn] "r"(clamp_min), [mx] "r"(clamp_max),
                      [nclo] "r"(nc_lo), [nchi] "r"(nc_hi), [vl32] "r"(vl32)
                    : "p0", "p1", "p2",
                      "z0", "z1", "z2", "z3", "z4", "z5", "z6", "z7",
                      "z8", "z9", "z10", "z11", "z12", "z13",
                      "x9", "w14", "memory", "cc"
                );
            }
        }
    }

    __asm__ volatile(".inst 0xd503467f\n" ::: "memory");  // smstop
}

#endif  // Architectural features check.
