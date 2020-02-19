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
/// Source file containing tests for logical, accessor, min/max, and cas atomic operations
///

#include "test-atomic-ref-forall-other.hpp"

// top layer of function templates for tests
template <typename ExecPolicy, typename AtomicPolicy>
void testAtomicRefPol()
{
  testAtomicRefIntegral<ExecPolicy, AtomicPolicy, int, tenk>();
  #if defined(TEST_EXHAUSTIVE)
  testAtomicRefIntegral<ExecPolicy, AtomicPolicy, unsigned, tenk>();
  testAtomicRefIntegral<ExecPolicy, AtomicPolicy, long long, tenk>();
  testAtomicRefIntegral<ExecPolicy, AtomicPolicy, unsigned long long, tenk>();

  testAtomicRefFloating<ExecPolicy, AtomicPolicy, float, tenk>();
  #endif
  testAtomicRefFloating<ExecPolicy, AtomicPolicy, double, tenk>();
}


// test instantiations
#if defined(RAJA_ENABLE_OPENMP)
TEST(Atomic, OpenMP_omp_AtomicRefOtherFunctionalTest)
{
  testAtomicRefPol<RAJA::omp_for_exec, RAJA::omp_atomic>();
}

TEST(Atomic, OpenMP_builtin_AtomicRefOtherFunctionalTest)
{
  testAtomicRefPol<RAJA::omp_for_exec, RAJA::builtin_atomic>();
}
#endif

#if defined(RAJA_ENABLE_CUDA)
GPU_TEST(Atomic, CUDA_cuda_AtomicRefOtherFunctionalTest)
{
  testAtomicRefPol<RAJA::cuda_exec<256>, RAJA::cuda_atomic>();
}
#endif

#if defined(TEST_EXHAUSTIVE) || !defined(RAJA_ENABLE_OPENMP)
TEST(Atomic, basic_seq_AtomicRefOtherFunctionalTest)
{
  testAtomicRefPol<RAJA::seq_exec, RAJA::seq_atomic>();
}

TEST(Atomic, basic_builtin_AtomicRefOtherFunctionalTest)
{
  testAtomicRefPol<RAJA::seq_exec, RAJA::builtin_atomic>();
}
#endif

