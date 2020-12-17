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

#ifndef RAJA_policy_register_arch_HPP
#define RAJA_policy_register_arch_HPP

#include "RAJA/config.hpp"
#include "RAJA/pattern/tensor/TensorRegister.hpp"

//
//////////////////////////////////////////////////////////////////////
//
// SIMD register types and policies
//
//////////////////////////////////////////////////////////////////////
//

#ifdef __AVX512F__
#include<RAJA/policy/simd_register/arch/avx512.hpp>
#ifndef RAJA_VECTOR_REGISTER_TYPE
#define RAJA_VECTOR_REGISTER_TYPE RAJA::avx512_register
#endif
#endif


#ifdef __AVX2__
#include<RAJA/policy/simd_register/arch/avx2.hpp>
#ifndef RAJA_VECTOR_REGISTER_TYPE
#define RAJA_VECTOR_REGISTER_TYPE RAJA::avx2_register
#endif
#endif


#ifdef __AVX__
#include<RAJA/policy/simd_register/arch/avx.hpp>
#ifndef RAJA_VECTOR_REGISTER_TYPE
#define RAJA_VECTOR_REGISTER_TYPE RAJA::avx_register
#endif
#endif



//#ifdef RAJA_ALTIVEC
//#include<RAJA/policy/vector/register/altivec.hpp>
//#ifndef RAJA_VECTOR_REGISTER_TYPE
//#define RAJA_VECTOR_REGISTER_TYPE RAJA::altivec_register
//#endif
//#endif


// The scalar register is always supported (doesn't require any SIMD/SIMT)
#include<RAJA/policy/simd_register/arch/scalar/scalar.hpp>
#ifndef RAJA_VECTOR_REGISTER_TYPE
#define RAJA_VECTOR_REGISTER_TYPE RAJA::scalar_register
#endif


namespace RAJA
{
namespace policy
{
    // This sets the default SIMD register that will be used
    using register_default = RAJA_VECTOR_REGISTER_TYPE;

} // namespace policy
} // namespace RAJA


#endif
