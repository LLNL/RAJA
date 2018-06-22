/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Header file for a portable SIMD vector register abstraction
 *
 *
 ******************************************************************************
 */

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-18, Lawrence Livermore National Security, LLC.
//
// Produced at the Lawrence Livermore National Laboratory
//
// LLNL-CODE-689114
//
// All rights reserved.
//
// This file is part of RAJA.
//
// For details about use and distribution, please read RAJA/LICENSE.
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef RAJA_policy_vec_vectorregister_HPP
#define RAJA_policy_vec_vectorregister_HPP

#include <iterator>
#include <type_traits>

#include "RAJA/config.hpp"

#include "RAJA/util/types.hpp"

namespace RAJA
{

namespace vec
{

namespace internal {


#if defined(RAJA_COMPILER_CLANG) 

template<typename T, size_t N>
using VectorRegister = T __attribute__((ext_vector_type(N)));

#endif


#if defined(RAJA_COMPILER_GNU)

template<typename T, size_t N>
using VectorRegister = T __attribute__((vector_size(N*sizeof(T))));

#endif




#if defined(RAJA_COMPILER_INTEL)

template<typename T, size_t N>
using VectorRegister = T __attribute__((vector_size(N*sizeof(T))));

#endif





} // namespace internal

}  // closing brace for vec namespace

}  // closing brace for RAJA namespace

#endif  // closing endif for header file include guard
