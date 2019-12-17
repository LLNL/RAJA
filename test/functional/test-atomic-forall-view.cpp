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

#if defined(RAJA_ENABLE_OPENMP)

TEST(Atomic, OpenMP_auto_AtomicFuncFunctionalTest)
{
  testAtomicFunctionPol<RAJA::omp_for_exec, RAJA::auto_atomic>();
}
TEST(Atomic, OpenMP_omp_AtomicFuncFunctionalTest)
{
  testAtomicFunctionPol<RAJA::omp_for_exec, RAJA::omp_atomic>();
}
TEST(Atomic, OpenMP_builtin_AtomicFuncFunctionalTest)
{
  testAtomicFunctionPol<RAJA::omp_for_exec, RAJA::builtin_atomic>();
}


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


TEST(Atomic, OpenMP_auto_AtomicLogicalFunctionalTest)
{
  testAtomicLogicalPol<RAJA::omp_for_exec, RAJA::auto_atomic>();
}
TEST(Atomic, OpenMP_omp_AtomicLogicalFunctionalTest)
{
  testAtomicLogicalPol<RAJA::omp_for_exec, RAJA::omp_atomic>();
}
TEST(Atomic, OpenMP_builtin_AtomicLogicalFunctionalTest)
{
  testAtomicLogicalPol<RAJA::omp_for_exec, RAJA::builtin_atomic>();
}

#endif

#if defined(RAJA_ENABLE_CUDA)

CUDA_TEST(Atomic, CUDA_auto_AtomicFuncFunctionalTest)
{
  testAtomicFunctionPol<RAJA::cuda_exec<256>, RAJA::auto_atomic>();
}
CUDA_TEST(Atomic, CUDA_cuda_AtomicFuncFunctionalTest)
{
  testAtomicFunctionPol<RAJA::cuda_exec<256>, RAJA::cuda_atomic>();
}

CUDA_TEST(Atomic, CUDA_auto_AtomicViewFunctionalTest)
{
  testAtomicViewPol<RAJA::cuda_exec<256>, RAJA::auto_atomic>();
}
CUDA_TEST(Atomic, CUDA_cuda_AtomicViewFunctionalTest)
{
  testAtomicViewPol<RAJA::cuda_exec<256>, RAJA::cuda_atomic>();
}


CUDA_TEST(Atomic, CUDA_auto_AtomicLogicalFunctionalTest)
{
  testAtomicLogicalPol<RAJA::cuda_exec<256>, RAJA::auto_atomic>();
}
CUDA_TEST(Atomic, CUDA_cuda_AtomicLogicalFunctionalTest)
{
  testAtomicLogicalPol<RAJA::cuda_exec<256>, RAJA::cuda_atomic>();
}


#endif

TEST(Atomic, basic_auto_AtomicFuncFunctionalTest)
{
  testAtomicFunctionPol<RAJA::seq_exec, RAJA::auto_atomic>();
}
TEST(Atomic, basic_seq_AtomicFuncFunctionalTest)
{
  testAtomicFunctionPol<RAJA::seq_exec, RAJA::seq_atomic>();
}
TEST(Atomic, basic_atomic_AtomicFuncFunctionalTest)
{
  testAtomicFunctionPol<RAJA::seq_exec, RAJA::builtin_atomic>();
}

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


TEST(Atomic, basic_auto_AtomicLogicalFunctionalTest)
{
  testAtomicLogicalPol<RAJA::seq_exec, RAJA::auto_atomic>();
}
TEST(Atomic, basic_seq_AtomicLogicalFunctionalTest)
{
  testAtomicLogicalPol<RAJA::seq_exec, RAJA::seq_atomic>();
}
TEST(Atomic, basic_builtin_AtomicLogicalFunctionalTest)
{
  testAtomicLogicalPol<RAJA::seq_exec, RAJA::builtin_atomic>();
}
