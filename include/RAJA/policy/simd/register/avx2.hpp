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
// Copyright (c) 2016-19, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifdef __AVX2__

#ifndef RAJA_policy_simd_register_avx2_HPP
#define RAJA_policy_simd_register_avx2_HPP

namespace RAJA {
  struct simd_avx2_register {};
}


#endif

#include<RAJA/policy/simd/register/avx2_double2.hpp>
#include<RAJA/policy/simd/register/avx2_double3.hpp>
#include<RAJA/policy/simd/register/avx2_double4.hpp>


#endif // __AVX2__
