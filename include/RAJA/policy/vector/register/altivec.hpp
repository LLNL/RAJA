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
// Copyright (c) 2016-19, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef RAJA_policy_vector_register_altivec_HPP
#define RAJA_policy_vector_register_altivec_HPP

#include "RAJA/config.hpp"
#ifdef RAJA_ALTIVEC

#include<RAJA/pattern/vector.hpp>

namespace RAJA {
  struct altivec_register {};
}

#include<RAJA/policy/vector/register/altivec/altivec_int32.hpp>
#include<RAJA/policy/vector/register/altivec/altivec_int64.hpp>
#include<RAJA/policy/vector/register/altivec/altivec_float.hpp>
#include<RAJA/policy/vector/register/altivec/altivec_double.hpp>


#endif // RAJA_ALTIVEC
#endif // guard
