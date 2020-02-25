//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef __TEST_FORALL_HPP__
#define __TEST_FOARLL_HPP__

#include "RAJA/RAJA.hpp"
#include "camp/resource.hpp"

using camp::list;
using camp::cartesian_product;


// Unroll types for gtest testing::Types
template<class T>
struct Test;

template<class ...T>
struct Test<list<T...>>{
  using Types = ::testing::Types<T...>;
};


// Forall Functional Test Class
template<typename T>
class ForallFunctionalTest: public ::testing::Test {};


// Define Index Types
using IdxTypes = list<RAJA::Index_type,
                      short,
                      unsigned short,
                      int, 
                      unsigned int,
                      long,
                      unsigned long,
                      long int,
                      unsigned long int,
                      long long,
                      unsigned long long>;


// Generate Sequential Type List
using SequentialTypes = list< RAJA::seq_exec, 
                              RAJA::loop_exec,
                              RAJA::simd_exec
                            >;

using ListHost = list< camp::resources::Host >;
using SequentialForallTypes = Test<cartesian_product< IdxTypes, ListHost, SequentialTypes >>::Types;



// Generate Cuda Type List
#if defined(RAJA_ENABLE_CUDA)
using CudaTypes = list< RAJA::cuda_exec<128>
                      >;

using ListCuda = list < camp::resources::Cuda >;
using CudaForallTypes = Test<cartesian_product< IdxTypes, ListCuda, CudaTypes >>::Types;
#endif

#endif //__TEST_FORALL_TYPES_HPP__
