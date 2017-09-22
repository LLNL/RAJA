//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-17, Lawrence Livermore National Security, LLC.
//
// Produced at the Lawrence Livermore National Laboratory
//
// LLNL-CODE-689114
//
// All rights reserved.
//
// This file is part of RAJA.
//
// For additional details, please also read RAJA/README.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// * Redistributions of source code must retain the above copyright notice,
//   this list of conditions and the disclaimer below.
//
// * Redistributions in binary form must reproduce the above copyright notice,
//   this list of conditions and the disclaimer (as noted below) in the
//   documentation and/or other materials provided with the distribution.
//
// * Neither the name of the LLNS/LLNL nor the names of its contributors may
//   be used to endorse or promote products derived from this software without
//   specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL LAWRENCE LIVERMORE NATIONAL SECURITY,
// LLC, THE U.S. DEPARTMENT OF ENERGY OR CONTRIBUTORS BE LIABLE FOR ANY
// DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
// OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
// HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
// STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
// IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

///
/// Source file containing tests for CHAI with basic RAJA constructs
///

#include "chai/ManagedArray.hpp"

#include "RAJA/RAJA.hpp"
#include "RAJA_gtest.hpp"

#include <iostream>

CUDA_TEST(ChaiTest, Simple) {
  chai::ManagedArray<float> v1(10);
  chai::ManagedArray<float> v2(10);

  RAJA::forall<RAJA::seq_exec>(0, 10, [=] (int i) {
      v1[i] = static_cast<float>(i * 1.0f);
  });

  std::cout << "end of loop 1" << std::endl;


#if defined(RAJA_ENABLE_CUDA)
  RAJA::forall<RAJA::cuda_exec<16> >(0, 10, [=] __device__ (int i) {
      v2[i] = v1[i]*2.0f;
  });
#else
  RAJA::forall<RAJA::omp_for_exec >(0, 10, [=] (int i) {
      v2[i] = v1[i]*2.0f;
  });
#endif

  std::cout << "end of loop 2" << std::endl;

  RAJA::forall<RAJA::seq_exec>(0, 10, [=] (int i) {
      ASSERT_FLOAT_EQ(v2[i], i*2.0f);
  });


#if defined(RAJA_ENABLE_CUDA)
  RAJA::forall<RAJA::cuda_exec<16> >(0, 10, [=] __device__ (int i) {
      v2[i] *= 2.0f;
  });
#else
  RAJA::forall<RAJA::omp_for_exec >(0, 10, [=] (int i) {
      v2[i] *= 2.0f;
  });
#endif

  float * raw_v2 = v2;
  for (int i = 0; i < 10; i++ ) {
      ASSERT_FLOAT_EQ(raw_v2[i], i*2.0f*2.0f);;
  }
}

CUDA_TEST(ChaiTest, Views) {
  chai::ManagedArray<float> v1_array(10);
  chai::ManagedArray<float> v2_array(10);

  typedef RAJA::ManagedArrayView<float, RAJA::Layout<1> > view;

  view v1(v1_array, 10);
  view v2(v2_array, 10);

  RAJA::forall<RAJA::seq_exec>(0, 10, [=] (int i) {
      v1(i) = static_cast<float>(i * 1.0f);
  });

#if defined(RAJA_ENABLE_CUDA)
  RAJA::forall<RAJA::cuda_exec<16> >(0, 10, [=] __device__ (int i) {
      v2(i) = v1(i)*2.0f;
  });
#else
  RAJA::forall<RAJA::omp_for_exec >(0, 10, [=](int i) {
      v2(i) = v1(i)*2.0f;
  });
#endif

  RAJA::forall<RAJA::seq_exec>(0, 10, [=] (int i) {
      ASSERT_FLOAT_EQ(v2(i), i*2.0f);
  });


#if defined(RAJA_ENABLE_CUDA)
  RAJA::forall<RAJA::cuda_exec<16> >(0, 10, [=] __device__ (int i) {
      v2(i) *= 2.0f;
  });
#else
  RAJA::forall<RAJA::omp_for_exec >(0, 10, [=](int i) {
      v2(i) *= 2.0f;
  });
#endif

  float * raw_v2 = v2.data;
  for (int i = 0; i < 10; i++ ) {
      ASSERT_FLOAT_EQ(raw_v2[i], i*1.0f*2.0f*2.0f);;
  }
}
