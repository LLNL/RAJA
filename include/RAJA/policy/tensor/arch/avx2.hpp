/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Header file containing SIMD abstractions for AVX2
 *
 ******************************************************************************
 */

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-23, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifdef __AVX2__

#include<RAJA/policy/tensor/arch/avx2/traits.hpp>
#include<RAJA/policy/tensor/arch/avx2/avx2_int32.hpp>
#include<RAJA/policy/tensor/arch/avx2/avx2_int64.hpp>
#include<RAJA/policy/tensor/arch/avx2/avx2_float.hpp>
#include<RAJA/policy/tensor/arch/avx2/avx2_double.hpp>


#endif // __AVX2__
