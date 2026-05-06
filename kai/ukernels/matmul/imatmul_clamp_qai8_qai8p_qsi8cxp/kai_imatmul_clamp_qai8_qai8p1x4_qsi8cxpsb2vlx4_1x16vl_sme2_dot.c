//
// SPDX-FileCopyrightText: Copyright 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//
// EXP-2: Streaming-SVE2 SDOT via inline assembly (smstart/smstop per output row).
// Apple M4 only has SVE2 in streaming mode — C intrinsics outside streaming cause SIGILL.
// Pattern follows KleidiAI convention: kai_commit_za() + inline asm with .inst smstart/smstop.
//
// Key derivation: both LHS and RHS k-group strides are constant across chunk boundaries:
//   LHS stride = m_step * kr = 128 bytes (x8p2vlx4 layout)
//   RHS stride = n_step * kr = 128 bytes (qsi8cxpsb2vlx4 layout)
// Total k-group iterations = k_chunk_count * roundup(k_chunk_length, kr) / kr.
// Simple countdown loop advances both pointers by 128 bytes per iteration.

#if (!defined(__aarch64__) || !defined(__ARM_FEATURE_SVE2)) && !defined(_M_ARM64)
#error This file must be compiled for AArch64, FEAT_SVE2.
#else  // Architectural features check.

#include "kai_imatmul_clamp_qai8_qai8p1x4_qsi8cxpsb2vlx4_1x16vl_sme2_dot.h"

#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

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

// Streaming-SVE2 SDOT kernel:
//   - One output row at a time; 2 SVE int32 accumulators (z8, z9) for n_step columns.
//   - smstart / SDOT countdown loop / smstop per output row.
//   - Scale + clamp output in non-streaming C after smstop.
void kai_run_imatmul_clamp_qai8_qai8p1x4_qsi8cxpsb2vlx4_1x16vl_sme2_dot(
    size_t m, size_t n, size_t k_chunk_count, size_t k_chunk_length, const void* lhs_packed, const void* rhs_packed,
    void* dst, size_t dst_stride_row, const struct kai_matmul_requantize32_params* params) {

    const size_t m_step_val =
        kai_get_m_step_imatmul_clamp_qai8_qai8p1x4_qsi8cxpsb2vlx4_1x16vl_sme2_dot();
    const size_t n_step_val =
        kai_get_n_step_imatmul_clamp_qai8_qai8p1x4_qsi8cxpsb2vlx4_1x16vl_sme2_dot();
    const size_t kl_rounded = kai_roundup(k_chunk_length, kai_kr);
    // Number of int32 lanes per SVE register (= svcntw()).
    const size_t vl32 = kai_get_sme_vector_length_u8() / sizeof(int32_t);
    // SVE vector length in bytes (= svcntb()).
    const size_t sme_vl = kai_get_sme_vector_length_u8();
    // Total k-group iterations (across all chunks): proven constant stride of m_step*kr bytes.
    const uint64_t k_groups_total = (uint64_t)(k_chunk_count * (kl_rounded / kai_kr));
    // Stride between consecutive k-groups in LHS (= RHS, both 128 bytes on M4).
    const uint64_t lhs_kgrp_stride = (uint64_t)(m_step_val * kai_kr);
    const uint64_t rhs_kgrp_stride = (uint64_t)(n_step_val * kai_kr);

    kai_commit_za();

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

            for (size_t r = 0; r < m_count; r++) {
                // Accumulator buffers: 2 × vl32 int32 values (max vl32=16 on M4/512-bit).
                int32_t acc0[16], acc1[16];
                memcpy(acc0, rhs_bias, vl32 * sizeof(int32_t));
                memcpy(acc1, rhs_bias + vl32, vl32 * sizeof(int32_t));

                if (k_groups_total > 0) {
                    // LHS row pointer: k-group 0, row r.
                    // x8p2vlx4: byte (row r, k-group g) is at lhs_blk + g*m_step*kr + r*kr.
                    const int8_t* lhs_ptr = lhs_blk + r * kai_kr;
                    const int8_t* rhs_ptr = rhs_data;
                    uint64_t cnt = k_groups_total;

                    // Streaming-SVE2 SDOT countdown loop.
                    // z8.s / z9.s: accumulators (n_step=32 int32 outputs, 2 Z registers).
                    // z0.s: LHS 4-byte broadcast.
                    // z1.b / z2.b: RHS lo/hi 64-byte loads.
                    // w14: scratch for scalar LHS load.
                    // x15: scratch for RHS+VL address.
                    // Move acc pointers to x9/x10 to avoid Apple assembler's
                    // restriction on using x16/x17 as SVE base registers.
                    __asm__ volatile(
                        ".inst 0xd503477f\n"           // smstart
                        "ptrue p0.b\n"
                        "mov x9, %[a0]\n"
                        "mov x10, %[a1]\n"
                        "ld1w {z8.s}, p0/z, [x9]\n"
                        "ld1w {z9.s}, p0/z, [x10]\n"
                        "0:\n"
                        "ldr w14, [%[lp]]\n"           // load 4 LHS bytes for row r
                        "dup z0.s, w14\n"              // broadcast int32 to all S-lanes
                        "ld1b {z1.b}, p0/z, [%[rp]]\n"     // RHS lo: VL bytes
                        "add x15, %[rp], %[svl]\n"         // rhs + sme_vl
                        "ld1b {z2.b}, p0/z, [x15]\n"       // RHS hi: next VL bytes
                        "sdot z8.s, z1.b, z0.b\n"
                        "sdot z9.s, z2.b, z0.b\n"
                        "add %[lp], %[lp], %[ls]\n"    // advance LHS pointer
                        "add %[rp], %[rp], %[rs]\n"    // advance RHS pointer
                        "subs %[cnt], %[cnt], #1\n"
                        "b.ne 0b\n"
                        "st1w {z8.s}, p0, [x9]\n"
                        "st1w {z9.s}, p0, [x10]\n"
                        ".inst 0xd503467f\n"           // smstop
                        : [lp] "+r"(lhs_ptr), [rp] "+r"(rhs_ptr), [cnt] "+r"(cnt)
                        : [a0] "r"(acc0), [a1] "r"(acc1),
                          [ls] "r"(lhs_kgrp_stride), [rs] "r"(rhs_kgrp_stride),
                          [svl] "r"(sme_vl)
                        : "p0", "z0", "z1", "z2", "z8", "z9",
                          "x9", "x10", "x14", "x15", "w14", "memory", "cc"
                    );
                }

                // Scale, clamp and store output (non-streaming C).
                int8_t* dst_row = (int8_t*)dst + (m_blk + r) * dst_stride_row + n_blk;
                for (size_t c = 0; c < n_count; c++) {
                    const int32_t v = (c < vl32) ? acc0[c] : acc1[c - vl32];
                    float out_f = (float)v * rhs_scale[c];
                    int32_t out_i = (int32_t)roundf(out_f) + params->output_zero_point;
                    if (out_i < (int32_t)params->min_value) out_i = (int32_t)params->min_value;
                    if (out_i > (int32_t)params->max_value) out_i = (int32_t)params->max_value;
                    dst_row[c] = (int8_t)out_i;
                }
            }
        }
    }
}

#endif  // Architectural features check.
