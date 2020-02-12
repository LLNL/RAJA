//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-19, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

///
/// Source file containing basic functional tests for atomic operations with forall and views.
///

#include "test-atomic-forall-view.hpp"

// top layer of function templates for tests
template <typename ExecPolicy, typename AtomicPolicy>
void testAtomicViewPol()
{
  testAtomicViewBasic<ExecPolicy, AtomicPolicy, int, hundredk>();
  testAtomicViewBasic<ExecPolicy, AtomicPolicy, unsigned, hundredk>();
  testAtomicViewBasic<ExecPolicy, AtomicPolicy, long long, hundredk>();
  testAtomicViewBasic<ExecPolicy, AtomicPolicy, unsigned long long, hundredk>();
  testAtomicViewBasic<ExecPolicy, AtomicPolicy, float, hundredk>();
  testAtomicViewBasic<ExecPolicy, AtomicPolicy, double, hundredk>();
}

// test instantiations
#if defined(RAJA_ENABLE_OPENMP)

TEST(Atomic, OpenMP_auto_AtomicViewFunctionalTest)
{
  testAtomicViewPol<RAJA::omp_for_exec, RAJA::auto_atomic>();
}
TEST(Atomic, OpenMP_omp_AtomicViewFunctionalTest)
{
  testAtomicViewPol<RAJA::omp_for_exec, RAJA::omp_atomic>();
}
TEST(Atomic, OpenMP_builtin_AtomicViewFunctionalTest)
{
  testAtomicViewPol<RAJA::omp_for_exec, RAJA::builtin_atomic>();
}

#endif

#if defined(RAJA_ENABLE_CUDA)

GPU_TEST(Atomic, CUDA_auto_AtomicViewFunctionalTest)
{
  testAtomicViewPol<RAJA::cuda_exec<256>, RAJA::auto_atomic>();
}
GPU_TEST(Atomic, CUDA_cuda_AtomicViewFunctionalTest)
{
  testAtomicViewPol<RAJA::cuda_exec<256>, RAJA::cuda_atomic>();
}

#endif

TEST(Atomic, basic_auto_AtomicViewFunctionalTest)
{
  testAtomicViewPol<RAJA::seq_exec, RAJA::auto_atomic>();
}
TEST(Atomic, basic_seq_AtomicViewFunctionalTest)
{
  testAtomicViewPol<RAJA::seq_exec, RAJA::seq_atomic>();
}
TEST(Atomic, basic_builtin_AtomicViewFunctionalTest)
{
  testAtomicViewPol<RAJA::seq_exec, RAJA::builtin_atomic>();
}

