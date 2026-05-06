//
// SPDX-FileCopyrightText: Copyright 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//
// EXP-1: Pure C reference. m_step=32 matches x8p2vlx4 packer, n_step=32 matches qsi8cxpsb2vlx4 packer.
// Both LHS and RHS de-interleaving done in scalar loops — correct but slow.

#if (!defined(__aarch64__) || !defined(__ARM_FEATURE_SVE2)) && !defined(_M_ARM64)
#error This file must be compiled for AArch64, FEAT_SVE2.
#else  // Architectural features check.

#include "kai_matmul_clamp_qai8_qai8p2vlx4_qsi8cxpsb2vlx4_1x16vl_sme2_dot.h"

#include <math.h>
#include <stddef.h>
#include <stdint.h>

#include "kai/kai_common.h"

static const size_t kai_mr = 2;
static const size_t kai_nr = 2;
static const size_t kai_kr = 4;
static const size_t kai_sr = 1;

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

// Pure C reference: LHS x8p2vlx4 de-interleave + qsi8cxpsb2vlx4 RHS layout.
// LHS element (row r, k pos k): (k/4)*m_step*4 + r*4 + (k%4)
// RHS element (col c, k pos k): (k/4)*n_step*4 + c*4 + (k%4)
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

    for (size_t m_blk = 0; m_blk < m; m_blk += m_step_val) {
        const size_t m_count = (m_blk + m_step_val <= m) ? m_step_val : (m - m_blk);
        const int8_t* lhs_blk = (const int8_t*)lhs_packed + m_blk * k_rounded;

        for (size_t n_blk = 0; n_blk < n; n_blk += n_step_val) {
            const size_t n_count = (n_blk + n_step_val <= n) ? n_step_val : (n - n_blk);
            const int8_t* rhs_blk = (const int8_t*)rhs_packed +
                (n_blk / n_step_val) * kai_get_rhs_packed_stride(n_step_val, k);
            const int32_t* rhs_bias = (const int32_t*)rhs_blk;
            const int8_t* rhs_data = (const int8_t*)(rhs_bias + n_step_val);
            const float* rhs_scale = (const float*)(rhs_data + n_step_val * k_rounded);

            for (size_t r = 0; r < m_count; r++) {
                for (size_t c = 0; c < n_count; c++) {
                    int32_t acc = rhs_bias[c];

                    for (size_t ki = 0; ki < k; ki++) {
                        const size_t lhs_byte =
                            (ki / kai_kr) * m_step_val * kai_kr + r * kai_kr + (ki % kai_kr);
                        const size_t rhs_byte =
                            (ki / kai_kr) * n_step_val * kai_kr + c * kai_kr + (ki % kai_kr);
                        acc += (int32_t)lhs_blk[lhs_byte] * (int32_t)rhs_data[rhs_byte];
                    }

                    float out_f = (float)acc * rhs_scale[c];
                    int32_t out_i = (int32_t)roundf(out_f) + params->output_zero_point;
                    if (out_i < (int32_t)params->min_value) out_i = (int32_t)params->min_value;
                    if (out_i > (int32_t)params->max_value) out_i = (int32_t)params->max_value;
                    ((int8_t*)dst)[(m_blk + r) * dst_stride_row + (n_blk + c)] = (int8_t)out_i;
                }
            }
        }
    }
}

#endif  // Architectural features check.
