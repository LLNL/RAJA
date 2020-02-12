//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-19, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

///
/// Header file containing basic functional tests for atomic operations with forall and views.
///

#include <RAJA/RAJA.hpp>
#include "RAJA_gtest.hpp"
#include "RAJA_value_params.hpp"


template <typename ExecPolicy,
          typename AtomicPolicy,
          typename T,
          RAJA::Index_type N>
void testAtomicViewBasic()
{
  RAJA::RangeSegment seg(0, N);
  RAJA::RangeSegment seg_half(0, N / 2);

// initialize an array
#if defined(RAJA_ENABLE_CUDA)
  T *source = nullptr;
  cudaErrchk(cudaMallocManaged((void **)&source, sizeof(T) * N));

  T *dest = nullptr;
  cudaErrchk(cudaMallocManaged((void **)&dest, sizeof(T) * N / 2));

  cudaErrchk(cudaDeviceSynchronize());
#else
  T *source = new T[N];
  T *dest = new T[N / 2];
#endif

  RAJA::forall<RAJA::seq_exec>(seg,
                               [=](RAJA::Index_type i) { source[i] = (T)1; });

  // use atomic add to reduce the array
  RAJA::View<T, RAJA::Layout<1>> vec_view(source, N);

  RAJA::View<T, RAJA::Layout<1>> sum_view(dest, N);
  auto sum_atomic_view = RAJA::make_atomic_view<AtomicPolicy>(sum_view);


  // Zero out dest using atomic view
  RAJA::forall<ExecPolicy>(seg_half, [=] RAJA_HOST_DEVICE(RAJA::Index_type i) {
    sum_atomic_view(i) = (T)0;
  });

  // Assign values to dest using atomic view
  RAJA::forall<ExecPolicy>(seg, [=] RAJA_HOST_DEVICE(RAJA::Index_type i) {
    sum_atomic_view(i / 2) += vec_view(i);
  });

#if defined(RAJA_ENABLE_CUDA)
  cudaErrchk(cudaDeviceSynchronize());
#endif

  for (RAJA::Index_type i = 0; i < N / 2; ++i) {
    EXPECT_EQ((T)2, dest[i]);
  }

#if defined(RAJA_ENABLE_CUDA)
  cudaErrchk(cudaFree(source));
  cudaErrchk(cudaFree(dest));
#else
  delete[] source;
  delete[] dest;
#endif
}


