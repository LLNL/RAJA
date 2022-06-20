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
// Copyright (c) 2016-22, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef RAJA_policy_tensor_arch_impl_HPP
#define RAJA_policy_tensor_arch_impl_HPP

#include "RAJA/config.hpp"
#include "RAJA/policy/tensor/arch.hpp"



//
//////////////////////////////////////////////////////////////////////
//
// SIMD register types and policies
//
//////////////////////////////////////////////////////////////////////
//

#ifdef __AVX512F__
#include<RAJA/policy/tensor/arch/avx512.hpp>
#endif


#ifdef __AVX2__
#include<RAJA/policy/tensor/arch/avx2.hpp>
#endif


#ifdef __AVX__
#include<RAJA/policy/tensor/arch/avx.hpp>
#endif

#ifdef RAJA_CUDA_ACTIVE
#include<RAJA/policy/tensor/arch/cuda.hpp>
#endif

#ifdef RAJA_HIP_ACTIVE
#include<RAJA/policy/tensor/arch/hip.hpp>
#endif

// The scalar register is always supported (doesn't require any SIMD/SIMT)
#include<RAJA/policy/tensor/arch/scalar.hpp>



#endif
