//
// SPDX-FileCopyrightText: Copyright 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//
// EXP-3: Single SMSTART/SMSTOP for entire function invocation.
// SDOT + scale/clamp/store all run inside one streaming region.
// Exp2 paid SMSTART/SMSTOP once per (n_blk × row); exp3 pays it once total.
// On M4 with n=256/n_step=32: eliminates 7 of 8 SMSTART/SMSTOP pairs (~630 cycles).
// Additional change: [rp, #1, MUL VL] immediate offset for RHS hi load (no x15 scratch).
// Scale/clamp via streaming SVE: scvtf → fmul → frinta → fcvtzs → smax/smin → st1b.

#if (!defined(__aarch64__) || !defined(__ARM_FEATURE_SVE2)) && !defined(_M_ARM64)
#error This file must be compiled for AArch64, FEAT_SVE2.
#else  // Architectural features check.

#include "kai_imatmul_clamp_qai8_qai8p1x4_qsi8cxpsb2vlx4_1x16vl_sme2_dot.h"

#include <stddef.h>
#include <stdint.h>

#include "kai/kai_common.h"

static const size_t kai_mr = 2;
static const size_t kai_nr = 2;
static const size_t kai_kr = 4;

size_t kai_get_m_step_imatmul_clamp_qai8_qai8p1x4_qsi8cxpsb2vlx4_1x16vl_sme2_dot(void) {
    return kai_mr * kai_get_sme_vector_length_u8() / kai_kr;
}

size_t kai_get_n_step_imatmul_clamp_qai8_qai8p1x4_qsi8cxpsb2vlx4_1x16vl_sme2_dot(void) {
    return kai_nr * kai_get_sme_vector_length_u8() / kai_kr;
}

size_t kai_get_lhs_packed_offset_imatmul_clamp_qai8_qai8p1x4_qsi8cxpsb2vlx4_1x16vl_sme2_dot(
    size_t m_idx, size_t k_chunk_count, size_t k_chunk_length) {
    KAI_ASSUME(m_idx % kai_get_m_step_imatmul_clamp_qai8_qai8p1x4_qsi8cxpsb2vlx4_1x16vl_sme2_dot() == 0);
    return m_idx * k_chunk_count * kai_roundup(k_chunk_length, kai_kr) * sizeof(int8_t);
}

static size_t kai_get_rhs_packed_stride(size_t n_step_val, size_t k_chunk_count, size_t k_chunk_length) {
    return n_step_val *
        (sizeof(int32_t) + k_chunk_count * kai_roundup(k_chunk_length, kai_kr) * sizeof(int8_t) + sizeof(float));
}

size_t kai_get_rhs_packed_offset_imatmul_clamp_qai8_qai8p1x4_qsi8cxpsb2vlx4_1x16vl_sme2_dot(
    size_t n_idx, size_t k_chunk_count, size_t k_chunk_length) {
    const size_t n_step_val =
        kai_get_n_step_imatmul_clamp_qai8_qai8p1x4_qsi8cxpsb2vlx4_1x16vl_sme2_dot();
    KAI_ASSUME(n_idx % n_step_val == 0);
    return (n_idx / n_step_val) * kai_get_rhs_packed_stride(n_step_val, k_chunk_count, k_chunk_length);
}

size_t kai_get_dst_offset_imatmul_clamp_qai8_qai8p1x4_qsi8cxpsb2vlx4_1x16vl_sme2_dot(
    size_t m_idx, size_t n_idx, size_t dst_stride_row) {
    KAI_ASSUME(m_idx % kai_get_m_step_imatmul_clamp_qai8_qai8p1x4_qsi8cxpsb2vlx4_1x16vl_sme2_dot() == 0);
    KAI_ASSUME(n_idx % kai_get_n_step_imatmul_clamp_qai8_qai8p1x4_qsi8cxpsb2vlx4_1x16vl_sme2_dot() == 0);
    return m_idx * dst_stride_row + n_idx * sizeof(int8_t);
}

size_t kai_get_dst_size_imatmul_clamp_qai8_qai8p1x4_qsi8cxpsb2vlx4_1x16vl_sme2_dot(size_t m, size_t n) {
    return m * n * sizeof(int8_t);
}

void kai_run_imatmul_clamp_qai8_qai8p1x4_qsi8cxpsb2vlx4_1x16vl_sme2_dot(
    size_t m, size_t n, size_t k_chunk_count, size_t k_chunk_length, const void* lhs_packed, const void* rhs_packed,
    void* dst, size_t dst_stride_row, const struct kai_matmul_requantize32_params* params) {

    const size_t m_step_val =
        kai_get_m_step_imatmul_clamp_qai8_qai8p1x4_qsi8cxpsb2vlx4_1x16vl_sme2_dot();
    const size_t n_step_val =
        kai_get_n_step_imatmul_clamp_qai8_qai8p1x4_qsi8cxpsb2vlx4_1x16vl_sme2_dot();
    const size_t kl_rounded = kai_roundup(k_chunk_length, kai_kr);
    const size_t vl32 = kai_get_sme_vector_length_u8() / sizeof(int32_t);
    const uint64_t k_groups_total = (uint64_t)(k_chunk_count * (kl_rounded / kai_kr));
    const uint64_t lhs_kgrp_stride = (uint64_t)(m_step_val * kai_kr);
    const uint64_t rhs_kgrp_stride = (uint64_t)(n_step_val * kai_kr);

    kai_commit_za();

    // Enter streaming mode once for the entire function.
    __asm__ volatile(".inst 0xd503477f\n" ::: "memory");  // smstart

    const int32_t ozp       = params->output_zero_point;
    const int32_t clamp_min = (int32_t)params->min_value;
    const int32_t clamp_max = (int32_t)params->max_value;

    for (size_t m_blk = 0; m_blk < m; m_blk += m_step_val) {
        const size_t m_count = (m_blk + m_step_val <= m) ? m_step_val : (m - m_blk);
        const int8_t* lhs_blk = (const int8_t*)lhs_packed + m_blk * k_chunk_count * kl_rounded;

        for (size_t n_blk = 0; n_blk < n; n_blk += n_step_val) {
            const size_t n_count = (n_blk + n_step_val <= n) ? n_step_val : (n - n_blk);
            const int8_t* rhs_blk = (const int8_t*)rhs_packed +
                (n_blk / n_step_val) * kai_get_rhs_packed_stride(n_step_val, k_chunk_count, k_chunk_length);
            const int32_t* rhs_bias  = (const int32_t*)rhs_blk;
            const int8_t*  rhs_data  = (const int8_t*)(rhs_bias + n_step_val);
            const float*   rhs_scale = (const float*)(rhs_data + k_chunk_count * n_step_val * kl_rounded);

            // Predicate lane counts for partial last column-block.
            const uint64_t nc_lo = (n_count < vl32) ? n_count : vl32;
            const uint64_t nc_hi = (n_count > vl32) ? (n_count - vl32) : 0;

            for (size_t r = 0; r < m_count; r++) {
                const int8_t* lhs_ptr = lhs_blk + r * kai_kr;
                const int8_t* rhs_ptr = rhs_data;
                uint64_t       cnt     = k_groups_total;
                int8_t*        dst_ptr = (int8_t*)dst + (m_blk + r) * dst_stride_row + n_blk;

                // Streaming-SVE2 SDOT + requantize + store, all inside one streaming region.
                //
                // Registers:
                //   z8.s / z9.s  : int32 accumulators (lo / hi n_step half)
                //   z0.s         : LHS 4-byte broadcast
                //   z1.b / z2.b  : RHS lo / hi VL-byte loads
                //   z10.s/z11.s  : scale vectors (reused for dup of ozp/min/max)
                //   p0           : all-true (ptrue .b)
                //   p1 / p2      : lane masks for lo / hi store
                //   x9 / x11     : SVE base register relays (avoid x16/x17 restriction)
                //   w14          : LHS scalar load scratch
                __asm__ volatile(
                    "ptrue p0.b\n"
                    // Load bias into accumulators.
                    "mov x9, %[bias]\n"
                    "ld1w {z8.s}, p0/z, [x9]\n"
                    "ld1w {z9.s}, p0/z, [x9, #1, MUL VL]\n"
                    // SDOT countdown loop.
                    "cbz %[cnt], 1f\n"
                    "0:\n"
                    "ldr w14, [%[lp]]\n"
                    "dup z0.s, w14\n"
                    "mov x11, %[rp]\n"
                    "ld1b {z1.b}, p0/z, [x11]\n"
                    "ld1b {z2.b}, p0/z, [x11, #1, MUL VL]\n"
                    "sdot z8.s, z1.b, z0.b\n"
                    "sdot z9.s, z2.b, z0.b\n"
                    "add %[lp], %[lp], %[ls]\n"
                    "add %[rp], %[rp], %[rs]\n"
                    "subs %[cnt], %[cnt], #1\n"
                    "b.ne 0b\n"
                    "1:\n"
                    // Convert accumulators int32 → float.
                    "scvtf z8.s, p0/m, z8.s\n"
                    "scvtf z9.s, p0/m, z9.s\n"
                    // Multiply by per-column scale.
                    "mov x9, %[sp]\n"
                    "ld1w {z10.s}, p0/z, [x9]\n"
                    "ld1w {z11.s}, p0/z, [x9, #1, MUL VL]\n"
                    "fmul z8.s, p0/m, z8.s, z10.s\n"
                    "fmul z9.s, p0/m, z9.s, z11.s\n"
                    // Round half-away-from-zero (matches roundf).
                    "frinta z8.s, p0/m, z8.s\n"
                    "frinta z9.s, p0/m, z9.s\n"
                    // Convert back to int32 (frinta already produced integer values).
                    "fcvtzs z8.s, p0/m, z8.s\n"
                    "fcvtzs z9.s, p0/m, z9.s\n"
                    // Add output zero point.
                    "dup z10.s, %w[ozp]\n"
                    "add z8.s, z8.s, z10.s\n"
                    "add z9.s, z9.s, z10.s\n"
                    // Clamp to [min, max].
                    "dup z10.s, %w[mn]\n"
                    "smax z8.s, p0/m, z8.s, z10.s\n"
                    "smax z9.s, p0/m, z9.s, z10.s\n"
                    "dup z10.s, %w[mx]\n"
                    "smin z8.s, p0/m, z8.s, z10.s\n"
                    "smin z9.s, p0/m, z9.s, z10.s\n"
                    // Store lo block: low byte of each active S-lane → int8.
                    "whilelt p1.s, xzr, %[nclo]\n"
                    "mov x9, %[dp]\n"
                    "st1b {z8.s}, p1, [x9]\n"
                    // Store hi block if n_count > vl32.
                    "cbz %[nchi], 2f\n"
                    "whilelt p2.s, xzr, %[nchi]\n"
                    "add x9, x9, %[vl32]\n"
                    "st1b {z9.s}, p2, [x9]\n"
                    "2:\n"
                    : [lp] "+r"(lhs_ptr), [rp] "+r"(rhs_ptr), [cnt] "+r"(cnt)
                    : [bias] "r"(rhs_bias),
                      [sp]   "r"(rhs_scale),
                      [dp]   "r"(dst_ptr),
                      [ls]   "r"(lhs_kgrp_stride),
                      [rs]   "r"(rhs_kgrp_stride),
                      [ozp]  "r"(ozp),
                      [mn]   "r"(clamp_min),
                      [mx]   "r"(clamp_max),
                      [nclo] "r"(nc_lo),
                      [nchi] "r"(nc_hi),
                      [vl32] "r"(vl32)
                    : "p0", "p1", "p2",
                      "z0", "z1", "z2", "z8", "z9", "z10", "z11",
                      "x9", "x11", "x14", "w14", "memory", "cc"
                );
            }
        }
    }

    // Exit streaming mode.
    __asm__ volatile(".inst 0xd503467f\n" ::: "memory");  // smstop
}

#endif  // Architectural features check.
