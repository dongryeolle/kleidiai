//
// SPDX-FileCopyrightText: Copyright 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <stddef.h>

#include "kai/kai_common.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

/// Micro-kernel dependencies
/// -# kai_lhs_pack_x8p2vlx4_x8_sme to pack the LHS matrix.
/// -# kai_rhs_pack_kxn_qsi8cxp2vlx4sb_qs8cx_f32_i32_sme to pack the RHS matrix.

size_t kai_get_m_step_matmul_clamp_qai8_qai8p2vlx4_qsi8cxpsb2vlx4_1x16vl_sme2_dot(void);
size_t kai_get_n_step_matmul_clamp_qai8_qai8p2vlx4_qsi8cxpsb2vlx4_1x16vl_sme2_dot(void);
size_t kai_get_mr_matmul_clamp_qai8_qai8p2vlx4_qsi8cxpsb2vlx4_1x16vl_sme2_dot(void);
size_t kai_get_nr_matmul_clamp_qai8_qai8p2vlx4_qsi8cxpsb2vlx4_1x16vl_sme2_dot(void);
size_t kai_get_kr_matmul_clamp_qai8_qai8p2vlx4_qsi8cxpsb2vlx4_1x16vl_sme2_dot(void);
size_t kai_get_sr_matmul_clamp_qai8_qai8p2vlx4_qsi8cxpsb2vlx4_1x16vl_sme2_dot(void);

size_t kai_get_lhs_packed_offset_matmul_clamp_qai8_qai8p2vlx4_qsi8cxpsb2vlx4_1x16vl_sme2_dot(
    size_t m_idx, size_t k);

size_t kai_get_rhs_packed_offset_matmul_clamp_qai8_qai8p2vlx4_qsi8cxpsb2vlx4_1x16vl_sme2_dot(
    size_t n_idx, size_t k);

size_t kai_get_dst_offset_matmul_clamp_qai8_qai8p2vlx4_qsi8cxpsb2vlx4_1x16vl_sme2_dot(
    size_t m_idx, size_t n_idx, size_t dst_stride_row);

size_t kai_get_dst_size_matmul_clamp_qai8_qai8p2vlx4_qsi8cxpsb2vlx4_1x16vl_sme2_dot(size_t m, size_t n);

void kai_run_matmul_clamp_qai8_qai8p2vlx4_qsi8cxpsb2vlx4_1x16vl_sme2_dot(
    size_t m, size_t n, size_t k, const void* lhs_packed, const void* rhs_packed, void* dst,
    size_t dst_stride_row, size_t dst_stride_col,
    const struct kai_matmul_requantize32_params* params);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus
