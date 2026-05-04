//
// SPDX-FileCopyrightText: Copyright 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//
// EXP-1: Pure C reference implementation.
// m_step=32 matches x8p2vlx4 packer. n_step=32 matches qsi8cxpsb2vlx4 imatmul packer.
// Both LHS and RHS de-interleaving done in scalar loops — correct but slow.
// Goal: correctness gate before SME2 ASM (exp 2+).

#if (!defined(__aarch64__) || !defined(__ARM_FEATURE_SVE2)) && !defined(_M_ARM64)
#error This file must be compiled for AArch64, FEAT_SVE2.
#else  // Architectural features check.

#include "kai_imatmul_clamp_qai8_qai8p1x4_qsi8cxpsb2vlx4_1x16vl_sme2_dot.h"

#include <math.h>
#include <stddef.h>
#include <stdint.h>

#include "kai/kai_common.h"

// Tile constants
// mr=2, nr=2, kr=4 match the x8p2vlx4 LHS packer and qsi8cxpsb2vlx4 RHS packer.
// m_step = mr * sme_vl_u8 / kr; n_step = nr * sme_vl_u8 / kr.
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
    const size_t block_idx = n_idx / n_step_val;
    return block_idx * kai_get_rhs_packed_stride(n_step_val, k_chunk_count, k_chunk_length);
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

// Pure C reference: correct but O(m*n*k), not optimised.
// LHS format: x8p2vlx4 — element (row r, k pos k within chunk) at byte:
//   (k/kr)*m_step*kr + r*kr + (k%kr)
// RHS format: qsi8cxpsb2vlx4 imatmul — per n_step block:
//   [n_step int32 bias] | [kcc * n_step * roundup(kl,kr) int8 data] | [n_step float scale]
//   element (N-col c, k pos k within chunk) in data section:
//   chunk*n_step*roundup(kl,kr) + (k/kr)*n_step*kr + c*kr + (k%kr)
void kai_run_imatmul_clamp_qai8_qai8p1x4_qsi8cxpsb2vlx4_1x16vl_sme2_dot(
    size_t m, size_t n, size_t k_chunk_count, size_t k_chunk_length, const void* lhs_packed, const void* rhs_packed,
    void* dst, size_t dst_stride_row, const struct kai_matmul_requantize32_params* params) {

    const size_t m_step_val =
        kai_get_m_step_imatmul_clamp_qai8_qai8p1x4_qsi8cxpsb2vlx4_1x16vl_sme2_dot();
    const size_t n_step_val =
        kai_get_n_step_imatmul_clamp_qai8_qai8p1x4_qsi8cxpsb2vlx4_1x16vl_sme2_dot();
    const size_t kl_rounded = kai_roundup(k_chunk_length, kai_kr);

    for (size_t m_blk = 0; m_blk < m; m_blk += m_step_val) {
        const size_t m_count = (m_blk + m_step_val <= m) ? m_step_val : (m - m_blk);
        // LHS block base for this m-tile (already offset formula matches MOPA / x8p2vlx4)
        const int8_t* lhs_blk = (const int8_t*)lhs_packed +
            m_blk * k_chunk_count * kl_rounded;

        for (size_t n_blk = 0; n_blk < n; n_blk += n_step_val) {
            const size_t n_count = (n_blk + n_step_val <= n) ? n_step_val : (n - n_blk);
            // RHS n_step block
            const int8_t* rhs_blk = (const int8_t*)rhs_packed +
                (n_blk / n_step_val) * kai_get_rhs_packed_stride(n_step_val, k_chunk_count, k_chunk_length);
            const int32_t* rhs_bias = (const int32_t*)rhs_blk;
            const int8_t* rhs_data = (const int8_t*)(rhs_bias + n_step_val);
            const float* rhs_scale =
                (const float*)(rhs_data + k_chunk_count * n_step_val * kl_rounded);

            for (size_t r = 0; r < m_count; r++) {
                for (size_t c = 0; c < n_count; c++) {
                    int32_t acc = rhs_bias[c];

                    for (size_t chunk = 0; chunk < k_chunk_count; chunk++) {
                        const int8_t* lhs_chunk = lhs_blk + chunk * m_step_val * kl_rounded;
                        const int8_t* rhs_chunk = rhs_data + chunk * n_step_val * kl_rounded;

                        for (size_t k = 0; k < k_chunk_length; k++) {
                            // x8p2vlx4 LHS de-interleave
                            const size_t lhs_byte =
                                (k / kai_kr) * m_step_val * kai_kr + r * kai_kr + (k % kai_kr);
                            // qsi8cxpsb2vlx4 RHS layout
                            const size_t rhs_byte =
                                (k / kai_kr) * n_step_val * kai_kr + c * kai_kr + (k % kai_kr);
                            acc += (int32_t)lhs_chunk[lhs_byte] * (int32_t)rhs_chunk[rhs_byte];
                        }
                    }

                    float output_f = (float)acc * rhs_scale[c];
                    int32_t output_i32 = (int32_t)roundf(output_f) + params->output_zero_point;
                    if (output_i32 < (int32_t)params->min_value) output_i32 = (int32_t)params->min_value;
                    if (output_i32 > (int32_t)params->max_value) output_i32 = (int32_t)params->max_value;

                    ((int8_t*)dst)[(m_blk + r) * dst_stride_row + (n_blk + c)] = (int8_t)output_i32;
                }
            }
        }
    }
}

#endif  // Architectural features check.
