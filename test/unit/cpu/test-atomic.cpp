//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016, Lawrence Livermore National Security, LLC.
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
/// Source file containing tests for atomic operations
///

#include <RAJA/RAJA.hpp>
#include "RAJA_gtest.hpp"

TEST(Atomic, basic_add_seq)
{

  static const RAJA::Index_type N = 100000;

  RAJA::RangeSegment seg(0, N);

  // initialize an array
  double *vec_double = new double[N];
  double expected = 0.0;
  double *expected_ptr = &expected;
  RAJA::forall<RAJA::seq_exec>(seg, [=](RAJA::Index_type i){
    vec_double[i] = (double)i;
    *expected_ptr += (double)i;
  });

  // use atomic add to reduce the array
  double sum = 0.0;
  double *sum_ptr = &sum;
  RAJA::forall<RAJA::seq_exec>(seg, [=](RAJA::Index_type i){
    RAJA::atomicAdd<RAJA::auto_atomic>(sum_ptr, (double)i);
  });

  EXPECT_EQ(expected, sum);


  delete[] vec_double;
}

TEST(Atomic, basic_sub_seq)
{

  static const RAJA::Index_type N = 100000;

  RAJA::RangeSegment seg(0, N);

  // initialize an array
  double *vec_double = new double[N];
  double expected = 0.0;
  double *expected_ptr = &expected;
  RAJA::forall<RAJA::seq_exec>(seg, [=](RAJA::Index_type i){
    vec_double[i] = (double)i;
    *expected_ptr -= (double)i;
  });

  // use atomic add to reduce the array
  double sum = 0.0;
  double *sum_ptr = &sum;
  RAJA::forall<RAJA::seq_exec>(seg, [=](RAJA::Index_type i){
    RAJA::atomicSub<RAJA::auto_atomic>(sum_ptr, (double)i);
  });

  EXPECT_EQ(expected, sum);


  delete[] vec_double;
}





#ifdef RAJA_ENABLE_OPENMP
TEST(Atomic, basic_add_omp)
{

  static const RAJA::Index_type N = 100000;

  RAJA::RangeSegment seg(0, N);

  // initialize an array
  double *vec_double = new double[N];
  double expected = 0.0;
  double *expected_ptr = &expected;
  RAJA::forall<RAJA::seq_exec>(seg, [=](RAJA::Index_type i){
    vec_double[i] = (double)i;
    *expected_ptr += (double)i;
  });

  // use atomic add to reduce the array
  double sum = 0.0;
  double *sum_ptr = &sum;
  RAJA::forall<RAJA::omp_for_exec>(seg, [=](RAJA::Index_type i){
    RAJA::atomicAdd<RAJA::auto_atomic>(sum_ptr, (double)i);
  });

  EXPECT_EQ(expected, sum);

  delete[] vec_double;
}
#endif



#ifdef RAJA_ENABLE_OPENMP
TEST(Atomic, basic_sub_omp)
{

  static const RAJA::Index_type N = 100000;

  RAJA::RangeSegment seg(0, N);

  // initialize an array
  double *vec_double = new double[N];
  double expected = 0.0;
  double *expected_ptr = &expected;
  RAJA::forall<RAJA::seq_exec>(seg, [=](RAJA::Index_type i){
    vec_double[i] = (double)i;
    *expected_ptr -= (double)i;
  });

  // use atomic add to reduce the array
  double sum = 0.0;
  double *sum_ptr = &sum;
  RAJA::forall<RAJA::omp_for_exec>(seg, [=](RAJA::Index_type i){
    RAJA::atomicSub<RAJA::auto_atomic>(sum_ptr, (double)i);
  });

  EXPECT_EQ(expected, sum);

  delete[] vec_double;
}
#endif




TEST(Atomic, AtomicRef_add_seq)
{

  static const RAJA::Index_type N = 100000;

  RAJA::RangeSegment seg(0, N);

  // initialize an array
  double *vec_double = new double[N];
  double expected = 0.0;
  double *expected_ptr = &expected;
  RAJA::forall<RAJA::seq_exec>(seg, [=](RAJA::Index_type i){
    vec_double[i] = (double)i;
    *expected_ptr += (double)i;
  });

  // use atomic add to reduce the array
  double sum = 0.0;
  RAJA::AtomicRef<double> sum_ptr(&sum);
  RAJA::forall<RAJA::seq_exec>(seg, [=](RAJA::Index_type i){
    sum_ptr += (double) i;
  });

  EXPECT_EQ(expected, sum);


  delete[] vec_double;
}

TEST(Atomic, AtomicRef_inc_seq)
{

  static const RAJA::Index_type N = 100000;

  RAJA::RangeSegment seg(0, N);

  double expected = N;

  // use atomic add to reduce the array
  double sum1 = 0.0, sum2 = 0.0, sum3 = 0.0;
  RAJA::AtomicRef<double> sum_ptr1(&sum1);
  RAJA::AtomicRef<double> sum_ptr2(&sum2);
  RAJA::AtomicRef<double> sum_ptr3(&sum3);
  RAJA::forall<RAJA::seq_exec>(seg, [=](RAJA::Index_type i){
    sum_ptr1 ++;
    ++ sum_ptr2;
    sum_ptr3 += 1;
  });

  EXPECT_EQ(expected, sum1);
  EXPECT_EQ(expected, sum2);
  EXPECT_EQ(expected, sum3);

}




template<typename T, RAJA::Index_type N>
void testAtomicViewBasicOpenMP(){
  RAJA::RangeSegment seg(0, N);
  RAJA::RangeSegment seg_half(0, N/2);

  // initialize an array
  T *vec_double = new T[N];
  T *sum_ptr = new T[N/2];


  RAJA::forall<RAJA::seq_exec>(seg, [=](RAJA::Index_type i){
    vec_double[i] = (T)1;
  });

  RAJA::forall<RAJA::seq_exec>(seg_half, [=](RAJA::Index_type i){
    sum_ptr[i] = (T)0;
  });

  // use atomic add to reduce the array
  RAJA::View<T, RAJA::Layout<1>> vec_view(vec_double, N);

  RAJA::View<T, RAJA::Layout<1>> sum_view(sum_ptr, N);
  auto sum_atomic_view = RAJA::make_atomic_view(sum_view);

  RAJA::forall<RAJA::omp_for_exec>(seg,
    [=] (RAJA::Index_type i){
      sum_atomic_view(i/2) += vec_view(i);
    }
  );

  for(RAJA::Index_type i = 0;i < N/2;++ i){
    EXPECT_EQ((T)2, sum_ptr[i]);
  }


  delete[] vec_double;
  delete[] sum_ptr;
}



CUDA_TEST(Atomic, basic_OpenMP_AtomicView_int)
{
  testAtomicViewBasicOpenMP<int, 100000>();
}

CUDA_TEST(Atomic, basic_OpenMP_AtomicView_unsigned)
{
  testAtomicViewBasicOpenMP<unsigned, 100000>();
}

CUDA_TEST(Atomic, basic_OpenMP_AtomicView_float)
{
  testAtomicViewBasicOpenMP<float, 100000>();
}

CUDA_TEST(Atomic, basic_OpenMP_AtomicView_double)
{
  testAtomicViewBasicOpenMP<int, 100000>();
}
