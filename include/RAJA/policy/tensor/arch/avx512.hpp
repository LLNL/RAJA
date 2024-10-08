/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Header file containing SIMD abstractions for AVX512
 *
 ******************************************************************************
 */

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-24, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

// Check if the base AVX512 instructions are present
#ifdef __AVX512F__

#include<RAJA/policy/tensor/arch/avx512/traits.hpp>
#include<RAJA/policy/tensor/arch/avx512/avx512_int32.hpp>
#include<RAJA/policy/tensor/arch/avx512/avx512_int64.hpp>
#include<RAJA/policy/tensor/arch/avx512/avx512_float.hpp>
#include<RAJA/policy/tensor/arch/avx512/avx512_double.hpp>


#endif // __AVX512F__
