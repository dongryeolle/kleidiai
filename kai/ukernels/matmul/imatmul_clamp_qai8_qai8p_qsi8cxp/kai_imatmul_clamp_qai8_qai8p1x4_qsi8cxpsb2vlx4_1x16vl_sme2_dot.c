//
// SPDX-FileCopyrightText: Copyright 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//
// EXP-0 SKELETON: stub implementation, no computation.
// Registers the 1x16vl_sme2_dot imatmul variant so build + benchmark infrastructure
// can be validated before the actual kernel is written.

#if (!defined(__aarch64__) || !defined(__ARM_FEATURE_SVE2)) && !defined(_M_ARM64)
#error This file must be compiled for AArch64, FEAT_SVE2.
#else  // Architectural features check.

#include "kai_imatmul_clamp_qai8_qai8p1x4_qsi8cxpsb2vlx4_1x16vl_sme2_dot.h"

#include <stddef.h>
#include <stdint.h>

#include "kai/kai_common.h"

// Tile constants
// m_step=1: GEMV — one output row per call (targets LLM decode M=1)
// n_step=16*vl: wide N tile using SME2 ZA accumulator across 16 SVE-length columns
// kr=4: K reduction factor (4 INT8 elements per dot product)
static const size_t kai_m_step = 1;
static const size_t kai_n_step = 16;
static const size_t kai_kr = 4;

size_t kai_get_m_step_imatmul_clamp_qai8_qai8p1x4_qsi8cxpsb2vlx4_1x16vl_sme2_dot(void) {
    return kai_m_step;
}

size_t kai_get_n_step_imatmul_clamp_qai8_qai8p1x4_qsi8cxpsb2vlx4_1x16vl_sme2_dot(void) {
    return kai_n_step * kai_get_sme_vector_length_u8() / kai_kr;
}

size_t kai_get_lhs_packed_offset_imatmul_clamp_qai8_qai8p1x4_qsi8cxpsb2vlx4_1x16vl_sme2_dot(
    size_t m_idx, size_t k_chunk_count, size_t k_chunk_length) {
    KAI_ASSUME(m_idx % kai_m_step == 0);
    return m_idx * k_chunk_count * kai_roundup(k_chunk_length, kai_kr) * sizeof(int8_t);
}

static size_t kai_get_rhs_packed_stride_imatmul_clamp_qai8_qai8p1x4_qsi8cxpsb2vlx4_1x16vl_sme2_dot(
    size_t k_chunk_count, size_t k_chunk_length) {
    return kai_get_n_step_imatmul_clamp_qai8_qai8p1x4_qsi8cxpsb2vlx4_1x16vl_sme2_dot() *
        (sizeof(int32_t) + k_chunk_count * kai_roundup(k_chunk_length, kai_kr) * sizeof(int8_t) + sizeof(float));
}

size_t kai_get_rhs_packed_offset_imatmul_clamp_qai8_qai8p1x4_qsi8cxpsb2vlx4_1x16vl_sme2_dot(
    size_t n_idx, size_t k_chunk_count, size_t k_chunk_length) {
    KAI_ASSUME(n_idx % kai_get_n_step_imatmul_clamp_qai8_qai8p1x4_qsi8cxpsb2vlx4_1x16vl_sme2_dot() == 0);
    const size_t block_idx =
        n_idx / kai_get_n_step_imatmul_clamp_qai8_qai8p1x4_qsi8cxpsb2vlx4_1x16vl_sme2_dot();
    return block_idx *
        kai_get_rhs_packed_stride_imatmul_clamp_qai8_qai8p1x4_qsi8cxpsb2vlx4_1x16vl_sme2_dot(
            k_chunk_count, k_chunk_length);
}

size_t kai_get_dst_offset_imatmul_clamp_qai8_qai8p1x4_qsi8cxpsb2vlx4_1x16vl_sme2_dot(
    size_t m_idx, size_t n_idx, size_t dst_stride_row) {
    KAI_ASSUME(m_idx % kai_m_step == 0);
    KAI_ASSUME(n_idx % kai_get_n_step_imatmul_clamp_qai8_qai8p1x4_qsi8cxpsb2vlx4_1x16vl_sme2_dot() == 0);
    return m_idx * dst_stride_row + n_idx * sizeof(int8_t);
}

size_t kai_get_dst_size_imatmul_clamp_qai8_qai8p1x4_qsi8cxpsb2vlx4_1x16vl_sme2_dot(size_t m, size_t n) {
    return m * n * sizeof(int8_t);
}

// EXP-0: no-op stub. Output buffer is left uninitialized.
// Correctness tests are expected to FAIL against this stub.
void kai_run_imatmul_clamp_qai8_qai8p1x4_qsi8cxpsb2vlx4_1x16vl_sme2_dot(
    size_t m, size_t n, size_t k_chunk_count, size_t k_chunk_length, const void* lhs_packed, const void* rhs_packed,
    void* dst, size_t dst_stride_row, const struct kai_matmul_requantize32_params* params) {
    KAI_UNUSED(m);
    KAI_UNUSED(n);
    KAI_UNUSED(k_chunk_count);
    KAI_UNUSED(k_chunk_length);
    KAI_UNUSED(lhs_packed);
    KAI_UNUSED(rhs_packed);
    KAI_UNUSED(dst);
    KAI_UNUSED(dst_stride_row);
    KAI_UNUSED(params);

    kai_commit_za();
}

#endif  // Architectural features check.
