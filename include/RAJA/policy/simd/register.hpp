/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Header file containing RAJA simd policy definitions.
 *
 ******************************************************************************
 */

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-19, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef RAJA_policy_simd_register_HPP
#define RAJA_policy_simd_register_HPP

#include<RAJA/pattern/register.hpp>
#include<RAJA/policy/simd/policy.hpp>

#ifdef __AVX__
#include<RAJA/policy/simd/register/avx.hpp>
#ifndef RAJA_SIMD_REGISTER_TYPE
#define RAJA_SIMD_REGISTER_TYPE simd_avx_register
#define RAJA_SIMD_REGISTER_WIDTH 256
#endif
#endif


namespace RAJA
{
  struct simd_scalar_register {};
}


#ifndef RAJA_SIMD_REGISTER_TYPE
#define RAJA_SIMD_REGISTER_TYPE RAJA::simd_scalar_register
#define RAJA_SIMD_REGISTER_WIDTH 0
#endif


namespace RAJA
{
namespace policy
{
  namespace simd
  {

    // This sets the default SIMD register that will be used
    // Individual registers can
    using simd_register = RAJA_SIMD_REGISTER_TYPE;
  }
}



  using policy::simd::simd_register;

}

#endif
