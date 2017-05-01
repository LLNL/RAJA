#include "gtest/gtest.h"
#include "chai/ManagedArray.hpp"

#include "RAJA/RAJA.hpp"

#include <iostream>

#define CUDA_TEST(X, Y) \
  static void cuda_test_ ## X ## Y();\
  TEST(X,Y) { cuda_test_ ## X ## Y();} \
  static void cuda_test_ ## X ## Y()

CUDA_TEST(ChaiTest, Simple) {
  chai::ManagedArray<float> v1(10);
  chai::ManagedArray<float> v2(10);

  RAJA::forall<RAJA::seq_exec>(0, 10, [=] (int i) {
      v1[i] = static_cast<float>(i * 1.0f);
  });

  RAJA::forall<RAJA::cuda_exec<16> >(0, 10, [=] __device__ (int i) {
      v2[i] = v1[i]*2.0f;
  });

  RAJA::forall<RAJA::seq_exec>(0, 10, [=] (int i) {
      ASSERT_FLOAT_EQ(v2[i], i*2.0f);
  });

  RAJA::forall<RAJA::cuda_exec<16> >(0, 10, [=] __device__ (int i) {
      v2[i] *= 2.0f;
  });

  float * raw_v2 = v2;
  for (int i = 0; i < 10; i++ ) {
      ASSERT_FLOAT_EQ(raw_v2[i], i*1.0f*2.0f*2.0f);;
  }
}

CUDA_TEST(ChaiTest, Views) {
  chai::ManagedArray<float> v1_array(10);
  chai::ManagedArray<float> v2_array(10);

  typedef RAJA::ManagedArrayView<float, RAJA::Layout<int, RAJA::PERM_I, int> > view;

  view v1(v1_array, 10);
  view v2(v2_array, 10);

  RAJA::forall<RAJA::seq_exec>(0, 10, [=] (int i) {
      v1(i) = static_cast<float>(i * 1.0f);
  });

  RAJA::forall<RAJA::cuda_exec<16> >(0, 10, [=] __device__ (int i) {
      v2(i) = v1(i)*2.0f;
  });

  RAJA::forall<RAJA::seq_exec>(0, 10, [=] (int i) {
      ASSERT_FLOAT_EQ(v2(i), i*2.0f);
  });

  RAJA::forall<RAJA::cuda_exec<16> >(0, 10, [=] __device__ (int i) {
      v2(i) *= 2.0f;
  });

  float * raw_v2 = v2.array;
  for (int i = 0; i < 10; i++ ) {
      ASSERT_FLOAT_EQ(raw_v2[i], i*1.0f*2.0f*2.0f);;
  }
}
