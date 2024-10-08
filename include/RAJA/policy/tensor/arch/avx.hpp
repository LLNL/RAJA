/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Header file containing SIMD abstractions for AVX
 *
 ******************************************************************************
 */

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-24, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifdef __AVX__

#include<RAJA/policy/tensor/arch/avx/traits.hpp>
#include<RAJA/policy/tensor/arch/avx/avx_int64.hpp>
#include<RAJA/policy/tensor/arch/avx/avx_int32.hpp>
#include<RAJA/policy/tensor/arch/avx/avx_float.hpp>
#include<RAJA/policy/tensor/arch/avx/avx_double.hpp>


#endif // __AVX__
