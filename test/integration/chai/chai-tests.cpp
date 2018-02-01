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
