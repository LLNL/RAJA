//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

///
/// Source file containing tests for atomic operations
///

#include <RAJA/RAJA.hpp>
#include "RAJA_gtest.hpp"


template <typename ExecPolicy,
          typename AtomicPolicy,
          typename T,
          RAJA::Index_type N>
void testAtomicFunctionBasic()
{
  RAJA::RangeSegment seg(0, N);

// initialize an array
#if defined(RAJA_ENABLE_CUDA)
  T *dest = nullptr;
  cudaErrchk(cudaMallocManaged((void **)&dest, sizeof(T) * 8));

  cudaErrchk(cudaDeviceSynchronize());

#else
  T *dest = new T[8];
#endif


  // use atomic add to reduce the array
  dest[0] = (T)0;
  dest[1] = (T)N;
  dest[2] = (T)N;
  dest[3] = (T)0;
  dest[4] = (T)0;
  dest[5] = (T)0;
  dest[6] = (T)N + 1;
  dest[7] = (T)0;


  RAJA::forall<ExecPolicy>(seg, [=] RAJA_HOST_DEVICE(RAJA::Index_type i) {
    RAJA::atomicAdd<AtomicPolicy>(dest + 0, (T)1);
    RAJA::atomicSub<AtomicPolicy>(dest + 1, (T)1);

    RAJA::atomicMin<AtomicPolicy>(dest + 2, (T)i);
    RAJA::atomicMax<AtomicPolicy>(dest + 3, (T)i);
    RAJA::atomicInc<AtomicPolicy>(dest + 4);
    RAJA::atomicInc<AtomicPolicy>(dest + 5, (T)16);
    RAJA::atomicDec<AtomicPolicy>(dest + 6);
    RAJA::atomicDec<AtomicPolicy>(dest + 7, (T)16);
  });

#if defined(RAJA_ENABLE_CUDA)
  cudaErrchk(cudaDeviceSynchronize());
#endif

  EXPECT_EQ((T)N, dest[0]);
  EXPECT_EQ((T)0, dest[1]);
  EXPECT_EQ((T)0, dest[2]);
  EXPECT_EQ((T)N - 1, dest[3]);
  EXPECT_EQ((T)N, dest[4]);
  EXPECT_EQ((T)4, dest[5]);
  EXPECT_EQ((T)1, dest[6]);
  EXPECT_EQ((T)13, dest[7]);


#if defined(RAJA_ENABLE_CUDA)
  cudaErrchk(cudaFree(dest));
#else
  delete[] dest;
#endif
}


template <typename ExecPolicy, typename AtomicPolicy>
void testAtomicFunctionPol()
{
  testAtomicFunctionBasic<ExecPolicy, AtomicPolicy, int, 10000>();
  testAtomicFunctionBasic<ExecPolicy, AtomicPolicy, unsigned, 10000>();
  testAtomicFunctionBasic<ExecPolicy, AtomicPolicy, long long, 10000>();
  testAtomicFunctionBasic<ExecPolicy,
                          AtomicPolicy,
                          unsigned long long,
                          10000>();
  testAtomicFunctionBasic<ExecPolicy, AtomicPolicy, float, 10000>();
  testAtomicFunctionBasic<ExecPolicy, AtomicPolicy, double, 10000>();
}



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


template <typename ExecPolicy, typename AtomicPolicy>
void testAtomicViewPol()
{
  testAtomicViewBasic<ExecPolicy, AtomicPolicy, int, 100000>();
  testAtomicViewBasic<ExecPolicy, AtomicPolicy, unsigned, 100000>();
  testAtomicViewBasic<ExecPolicy, AtomicPolicy, long long, 100000>();
  testAtomicViewBasic<ExecPolicy, AtomicPolicy, unsigned long long, 100000>();
  testAtomicViewBasic<ExecPolicy, AtomicPolicy, float, 100000>();
  testAtomicViewBasic<ExecPolicy, AtomicPolicy, double, 100000>();
}


template <typename ExecPolicy,
          typename AtomicPolicy,
          typename T,
          RAJA::Index_type N>
void testAtomicLogical()
{
  RAJA::RangeSegment seg(0, N * 8);
  RAJA::RangeSegment seg_bytes(0, N);

// initialize an array
#if defined(RAJA_ENABLE_CUDA)
  T *dest_and = nullptr;
  cudaErrchk(cudaMallocManaged((void **)&dest_and, sizeof(T) * N));

  T *dest_or = nullptr;
  cudaErrchk(cudaMallocManaged((void **)&dest_or, sizeof(T) * N));

  T *dest_xor = nullptr;
  cudaErrchk(cudaMallocManaged((void **)&dest_xor, sizeof(T) * N));

  cudaErrchk(cudaDeviceSynchronize());
#else
  T *dest_and = new T[N];
  T *dest_or = new T[N];
  T *dest_xor = new T[N];
#endif

  RAJA::forall<RAJA::seq_exec>(seg_bytes, [=](RAJA::Index_type i) {
    dest_and[i] = (T)0;
    dest_or[i] = (T)0;
    dest_xor[i] = (T)0;
  });


  RAJA::forall<ExecPolicy>(seg, [=] RAJA_HOST_DEVICE(RAJA::Index_type i) {
    RAJA::Index_type offset = i / 8;
    RAJA::Index_type bit = i % 8;
    RAJA::atomicAnd<AtomicPolicy>(dest_and + offset,
                                          (T)(0xFF ^ (1 << bit)));
    RAJA::atomicOr<AtomicPolicy>(dest_or + offset, (T)(1 << bit));
    RAJA::atomicXor<AtomicPolicy>(dest_xor + offset, (T)(1 << bit));
  });

#if defined(RAJA_ENABLE_CUDA)
  cudaErrchk(cudaDeviceSynchronize());
#endif

  for (RAJA::Index_type i = 0; i < N; ++i) {
    EXPECT_EQ((T)0x00, dest_and[i]);
    EXPECT_EQ((T)0xFF, dest_or[i]);
    EXPECT_EQ((T)0xFF, dest_xor[i]);
  }

#if defined(RAJA_ENABLE_CUDA)
  cudaErrchk(cudaFree(dest_and));
  cudaErrchk(cudaFree(dest_or));
  cudaErrchk(cudaFree(dest_xor));
#else
  delete[] dest_and;
  delete[] dest_or;
  delete[] dest_xor;
#endif
}


template <typename ExecPolicy, typename AtomicPolicy>
void testAtomicLogicalPol()
{
  testAtomicLogical<ExecPolicy, AtomicPolicy, int, 100000>();
  testAtomicLogical<ExecPolicy, AtomicPolicy, unsigned, 100000>();
  testAtomicLogical<ExecPolicy, AtomicPolicy, long long, 100000>();
  testAtomicLogical<ExecPolicy, AtomicPolicy, unsigned long long, 100000>();
}

#if defined(RAJA_ENABLE_HIP)

template <typename ExecPolicy,
          typename AtomicPolicy,
          typename T,
          RAJA::Index_type N>
void testAtomicFunctionBasic_gpu()
{
  RAJA::RangeSegment seg(0, N);

  // initialize an array
  T *dest = new T[8];
  T *d_dest = nullptr;
  hipMalloc((void **)&d_dest, sizeof(T) * 8);

  // use atomic add to reduce the array
  dest[0] = (T)0;
  dest[1] = (T)N;
  dest[2] = (T)N;
  dest[3] = (T)0;
  dest[4] = (T)0;
  dest[5] = (T)0;
  dest[6] = (T)N + 1;
  dest[7] = (T)0;

  hipMemcpy(d_dest, dest, 8*sizeof(T), hipMemcpyHostToDevice);

  RAJA::forall<ExecPolicy>(seg, [=] RAJA_HOST_DEVICE(RAJA::Index_type i) {
    RAJA::atomicAdd<AtomicPolicy>(d_dest + 0, (T)1);
    RAJA::atomicSub<AtomicPolicy>(d_dest + 1, (T)1);

    RAJA::atomicMin<AtomicPolicy>(d_dest + 2, (T)i);
    RAJA::atomicMax<AtomicPolicy>(d_dest + 3, (T)i);
    RAJA::atomicInc<AtomicPolicy>(d_dest + 4);
    RAJA::atomicInc<AtomicPolicy>(d_dest + 5, (T)16);
    RAJA::atomicDec<AtomicPolicy>(d_dest + 6);
    RAJA::atomicDec<AtomicPolicy>(d_dest + 7, (T)16);
  });

  hipDeviceSynchronize();

  hipMemcpy(dest, d_dest, 8*sizeof(T), hipMemcpyDeviceToHost);

  EXPECT_EQ((T)N, dest[0]);
  EXPECT_EQ((T)0, dest[1]);
  EXPECT_EQ((T)0, dest[2]);
  EXPECT_EQ((T)N - 1, dest[3]);
  EXPECT_EQ((T)N, dest[4]);
  EXPECT_EQ((T)4, dest[5]);
  EXPECT_EQ((T)1, dest[6]);
  EXPECT_EQ((T)13, dest[7]);


  delete[] dest;
  hipFree(d_dest);
}

template <typename ExecPolicy, typename AtomicPolicy>
void testAtomicFunctionPol_gpu()
{
  testAtomicFunctionBasic_gpu<ExecPolicy, AtomicPolicy, int, 10000>();
  testAtomicFunctionBasic_gpu<ExecPolicy, AtomicPolicy, unsigned, 10000>();
  testAtomicFunctionBasic_gpu<ExecPolicy, AtomicPolicy, long long, 10000>();
  testAtomicFunctionBasic_gpu<ExecPolicy,
                          AtomicPolicy,
                          unsigned long long,
                          10000>();
  testAtomicFunctionBasic_gpu<ExecPolicy, AtomicPolicy, float, 10000>();
  testAtomicFunctionBasic_gpu<ExecPolicy, AtomicPolicy, double, 10000>();
}

template <typename ExecPolicy,
          typename AtomicPolicy,
          typename T,
          RAJA::Index_type N>
void testAtomicViewBasic_gpu()
{
  RAJA::RangeSegment seg(0, N);
  RAJA::RangeSegment seg_half(0, N / 2);

// initialize an array
  T *source = new T[N];
  T *dest = new T[N / 2];
  T *d_source = nullptr;
  T *d_dest = nullptr;
  hipMalloc((void **)&d_source, sizeof(T) * N);
  hipMalloc((void **)&d_dest, sizeof(T) * N / 2);

  RAJA::forall<RAJA::seq_exec>(seg,
                               [=](RAJA::Index_type i) { source[i] = (T)1; });

  hipMemcpy(d_source, source, N*sizeof(T), hipMemcpyHostToDevice);

  // use atomic add to reduce the array
  RAJA::View<T, RAJA::Layout<1>> vec_view(d_source, N);

  RAJA::View<T, RAJA::Layout<1>> sum_view(d_dest, N);
  auto sum_atomic_view = RAJA::make_atomic_view<AtomicPolicy>(sum_view);

  // Zero out dest using atomic view
  RAJA::forall<ExecPolicy>(seg_half, [=] RAJA_HOST_DEVICE (RAJA::Index_type i) {
    sum_atomic_view(i) = (T)0;
  });

  // Assign values to dest using atomic view
  RAJA::forall<ExecPolicy>(seg, [=] RAJA_HOST_DEVICE(RAJA::Index_type i) {
    sum_atomic_view(i / 2) += vec_view(i);
  });

  hipDeviceSynchronize();

  hipMemcpy(dest, d_dest, (N / 2)*sizeof(T), hipMemcpyDeviceToHost);

  for (RAJA::Index_type i = 0; i < N / 2; ++i) {
    EXPECT_EQ((T)2, dest[i]);
  }

  hipFree(d_source);
  hipFree(d_dest);
  delete[] source;
  delete[] dest;
}


template <typename ExecPolicy, typename AtomicPolicy>
void testAtomicViewPol_gpu()
{
  testAtomicViewBasic_gpu<ExecPolicy, AtomicPolicy, int, 100000>();
  testAtomicViewBasic_gpu<ExecPolicy, AtomicPolicy, unsigned, 100000>();
  testAtomicViewBasic_gpu<ExecPolicy, AtomicPolicy, long long, 100000>();
  testAtomicViewBasic_gpu<ExecPolicy, AtomicPolicy, unsigned long long, 100000>();
  testAtomicViewBasic_gpu<ExecPolicy, AtomicPolicy, float, 100000>();
  testAtomicViewBasic_gpu<ExecPolicy, AtomicPolicy, double, 100000>();
}


template <typename ExecPolicy,
          typename AtomicPolicy,
          typename T,
          RAJA::Index_type N>
void testAtomicLogical_gpu()
{
  RAJA::RangeSegment seg(0, N * 8);
  RAJA::RangeSegment seg_bytes(0, N);

// initialize an array
  T *dest_and = new T[N];
  T *dest_or = new T[N];
  T *dest_xor = new T[N];

  T *d_dest_and = nullptr;
  T *d_dest_or = nullptr;
  T *d_dest_xor = nullptr;
  hipMalloc((void **)&d_dest_and, sizeof(T) * N);
  hipMalloc((void **)&d_dest_or, sizeof(T) * N);
  hipMalloc((void **)&d_dest_xor, sizeof(T) * N);

  RAJA::forall<RAJA::seq_exec>(seg_bytes, [=](RAJA::Index_type i) {
    dest_and[i] = (T)0;
    dest_or[i] = (T)0;
    dest_xor[i] = (T)0;
  });

  hipMemcpy(d_dest_and, dest_and, N*sizeof(T), hipMemcpyHostToDevice);
  hipMemcpy(d_dest_or,  dest_or,  N*sizeof(T), hipMemcpyHostToDevice);
  hipMemcpy(d_dest_xor, dest_xor, N*sizeof(T), hipMemcpyHostToDevice);

  RAJA::forall<ExecPolicy>(seg, [=] RAJA_HOST_DEVICE(RAJA::Index_type i) {
    RAJA::Index_type offset = i / 8;
    RAJA::Index_type bit = i % 8;
    RAJA::atomicAnd<AtomicPolicy>(d_dest_and + offset,
                                          (T)(0xFF ^ (1 << bit)));
    RAJA::atomicOr<AtomicPolicy>(d_dest_or + offset, (T)(1 << bit));
    RAJA::atomicXor<AtomicPolicy>(d_dest_xor + offset, (T)(1 << bit));
  });

  hipDeviceSynchronize();

  hipMemcpy(dest_and, d_dest_and, N*sizeof(T), hipMemcpyDeviceToHost);
  hipMemcpy(dest_or,  d_dest_or,  N*sizeof(T), hipMemcpyDeviceToHost);
  hipMemcpy(dest_xor, d_dest_xor, N*sizeof(T), hipMemcpyDeviceToHost);

  for (RAJA::Index_type i = 0; i < N; ++i) {
    EXPECT_EQ((T)0x00, dest_and[i]);
    EXPECT_EQ((T)0xFF, dest_or[i]);
    EXPECT_EQ((T)0xFF, dest_xor[i]);
  }

  hipFree(d_dest_and);
  hipFree(d_dest_or);
  hipFree(d_dest_xor);
  delete[] dest_and;
  delete[] dest_or;
  delete[] dest_xor;
}


template <typename ExecPolicy, typename AtomicPolicy>
void testAtomicLogicalPol_gpu()
{
  testAtomicLogical_gpu<ExecPolicy, AtomicPolicy, int, 100000>();
  testAtomicLogical_gpu<ExecPolicy, AtomicPolicy, unsigned, 100000>();
  testAtomicLogical_gpu<ExecPolicy, AtomicPolicy, long long, 100000>();
  testAtomicLogical_gpu<ExecPolicy, AtomicPolicy, unsigned long long, 100000>();
}

#endif //RAJA_ENABLE_HIP

#if defined(RAJA_ENABLE_OPENMP)

TEST(Atomic, basic_OpenMP_AtomicFunction)
{
  testAtomicFunctionPol<RAJA::omp_for_exec, RAJA::auto_atomic>();
  testAtomicFunctionPol<RAJA::omp_for_exec, RAJA::omp_atomic>();
  testAtomicFunctionPol<RAJA::omp_for_exec, RAJA::builtin_atomic>();
}


TEST(Atomic, basic_OpenMP_AtomicView)
{
  testAtomicViewPol<RAJA::omp_for_exec, RAJA::auto_atomic>();
  testAtomicViewPol<RAJA::omp_for_exec, RAJA::omp_atomic>();
  testAtomicViewPol<RAJA::omp_for_exec, RAJA::builtin_atomic>();
}


TEST(Atomic, basic_OpenMP_Logical)
{
  testAtomicLogicalPol<RAJA::omp_for_exec, RAJA::auto_atomic>();
  testAtomicLogicalPol<RAJA::omp_for_exec, RAJA::omp_atomic>();
  testAtomicLogicalPol<RAJA::omp_for_exec, RAJA::builtin_atomic>();
}

#endif

#if defined(RAJA_ENABLE_CUDA)

GPU_TEST(Atomic, basic_CUDA_AtomicFunction)
{
  testAtomicFunctionPol<RAJA::cuda_exec<256>, RAJA::auto_atomic>();
  testAtomicFunctionPol<RAJA::cuda_exec<256>, RAJA::cuda_atomic>();
}

GPU_TEST(Atomic, basic_CUDA_AtomicView)
{
  testAtomicViewPol<RAJA::cuda_exec<256>, RAJA::auto_atomic>();
  testAtomicViewPol<RAJA::cuda_exec<256>, RAJA::cuda_atomic>();
}


GPU_TEST(Atomic, basic_CUDA_Logical)
{
  testAtomicLogicalPol<RAJA::cuda_exec<256>, RAJA::auto_atomic>();
  testAtomicLogicalPol<RAJA::cuda_exec<256>, RAJA::cuda_atomic>();
}


#endif

#if defined(RAJA_ENABLE_HIP)

GPU_TEST(Atomic, basic_HIP_AtomicFunction)
{
  testAtomicFunctionPol_gpu<RAJA::hip_exec<256>, RAJA::auto_atomic>();
  testAtomicFunctionPol_gpu<RAJA::hip_exec<256>, RAJA::hip_atomic>();
}

GPU_TEST(Atomic, basic_HIP_AtomicView)
{
  testAtomicViewPol_gpu<RAJA::hip_exec<256>, RAJA::auto_atomic>();
  testAtomicViewPol_gpu<RAJA::hip_exec<256>, RAJA::hip_atomic>();
}


GPU_TEST(Atomic, basic_HIP_Logical)
{
  testAtomicLogicalPol_gpu<RAJA::hip_exec<256>, RAJA::auto_atomic>();
  testAtomicLogicalPol_gpu<RAJA::hip_exec<256>, RAJA::hip_atomic>();
}


#endif

TEST(Atomic, basic_seq_AtomicFunction)
{
  testAtomicFunctionPol<RAJA::seq_exec, RAJA::auto_atomic>();
  testAtomicFunctionPol<RAJA::seq_exec, RAJA::seq_atomic>();
  testAtomicFunctionPol<RAJA::seq_exec, RAJA::builtin_atomic>();
}

TEST(Atomic, basic_seq_AtomicView)
{
  testAtomicViewPol<RAJA::seq_exec, RAJA::auto_atomic>();
  testAtomicViewPol<RAJA::seq_exec, RAJA::seq_atomic>();
  testAtomicViewPol<RAJA::seq_exec, RAJA::builtin_atomic>();
}


TEST(Atomic, basic_seq_Logical)
{
  testAtomicLogicalPol<RAJA::seq_exec, RAJA::auto_atomic>();
  testAtomicLogicalPol<RAJA::seq_exec, RAJA::seq_atomic>();
  testAtomicLogicalPol<RAJA::seq_exec, RAJA::builtin_atomic>();
}
