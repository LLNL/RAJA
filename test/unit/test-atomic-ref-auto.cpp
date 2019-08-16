//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-19, Lawrence Livermore National Security, LLC.
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
/// Source file containing tests for atomic operations
///

#include "test-atomic-ref.hpp"

#if defined(RAJA_ENABLE_OPENMP)

TEST(Atomic, basic_OpenMP_AtomicRef)
{
  testAtomicRefPol<RAJA::omp_for_exec, RAJA::auto_atomic>();
}

#endif

#if defined(RAJA_ENABLE_CUDA)

GPU_TEST(Atomic, basic_CUDA_AtomicRef)
{
  testAtomicRefPol<RAJA::cuda_exec<256>, RAJA::auto_atomic>();
}

#endif

#if defined(RAJA_ENABLE_HIP)

GPU_TEST(Atomic, basic_HIP_AtomicRef)
{
  testAtomicRefPol_gpu<RAJA::hip_exec<256>, RAJA::atomic::auto_atomic>();
}

#endif

TEST(Atomic, basic_seq_AtomicRef)
{
  testAtomicRefPol<RAJA::seq_exec, RAJA::auto_atomic>();
}

