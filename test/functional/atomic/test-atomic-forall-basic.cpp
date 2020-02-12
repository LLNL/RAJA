//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-19, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

///
/// Source file containing basic functional tests for atomic operations with forall.
///

#include "RAJA_gtest.hpp"
#include "test-atomic-forall-basic.hpp"


// top layer of function templates for tests
template <typename ExecPolicy, typename AtomicPolicy>
void testAtomicFunctionPol()
{
  testAtomicFunctionBasic<ExecPolicy, AtomicPolicy, int, tenk>();
  testAtomicFunctionBasic<ExecPolicy, AtomicPolicy, unsigned, tenk>();
  testAtomicFunctionBasic<ExecPolicy, AtomicPolicy, long long, tenk>();
  testAtomicFunctionBasic<ExecPolicy,
                          AtomicPolicy,
                          unsigned long long,
                          tenk>();
  testAtomicFunctionBasic<ExecPolicy, AtomicPolicy, float, tenk>();
  testAtomicFunctionBasic<ExecPolicy, AtomicPolicy, double, tenk>();
}

template <typename ExecPolicy, typename AtomicPolicy>
void testAtomicLogicalPol()
{
  testAtomicLogical<ExecPolicy, AtomicPolicy, int, hundredk>();
  testAtomicLogical<ExecPolicy, AtomicPolicy, unsigned, hundredk>();
  testAtomicLogical<ExecPolicy, AtomicPolicy, long long, hundredk>();
  testAtomicLogical<ExecPolicy, AtomicPolicy, unsigned long long, hundredk>();
}

// test instantiations
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

GPU_TEST(Atomic, CUDA_auto_AtomicFuncFunctionalTest)
{
  testAtomicFunctionPol<RAJA::cuda_exec<256>, RAJA::auto_atomic>();
}
GPU_TEST(Atomic, CUDA_cuda_AtomicFuncFunctionalTest)
{
  testAtomicFunctionPol<RAJA::cuda_exec<256>, RAJA::cuda_atomic>();
}


GPU_TEST(Atomic, CUDA_auto_AtomicLogicalFunctionalTest)
{
  testAtomicLogicalPol<RAJA::cuda_exec<256>, RAJA::auto_atomic>();
}
GPU_TEST(Atomic, CUDA_cuda_AtomicLogicalFunctionalTest)
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

// Type parameterized test for experimentation/discussion.
TYPED_TEST_P(AtomicFuncBasicFunctionalTest, auto_basic_AtomicFuncFunctionalTest)
{
  testAtomicFunctionBasicV2<RAJA::seq_exec, RAJA::auto_atomic, TypeParam>( tenk );
}
TYPED_TEST_P(AtomicFuncBasicFunctionalTest, seq_basic_AtomicFuncFunctionalTest)
{
  testAtomicFunctionBasicV2<RAJA::seq_exec, RAJA::seq_atomic, TypeParam>( tenk );
}
TYPED_TEST_P(AtomicFuncBasicFunctionalTest, builtin_basic_AtomicFuncFunctionalTest)
{
  testAtomicFunctionBasicV2<RAJA::seq_exec, RAJA::builtin_atomic, TypeParam>( tenk );
}

REGISTER_TYPED_TEST_CASE_P( AtomicFuncBasicFunctionalTest,
                            auto_basic_AtomicFuncFunctionalTest,
                            seq_basic_AtomicFuncFunctionalTest,
                            builtin_basic_AtomicFuncFunctionalTest
                          );

using seqtypes = ::testing::Types<
                          int,
                          unsigned,
                          long long,
                          unsigned long long,
                          float,
                          double
                 >;

INSTANTIATE_TYPED_TEST_CASE_P( AtomicBasicFunctionalTest, AtomicFuncBasicFunctionalTest, seqtypes );
// END Type parameterized test for experimentation/discussion.
