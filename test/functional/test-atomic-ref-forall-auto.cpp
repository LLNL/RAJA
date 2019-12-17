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

#include "test-atomic-ref-forall.hpp"

#if defined(RAJA_ENABLE_OPENMP)

TEST(Atomic, OpenMP_auto_AtomicRefForallFunctionalTest)
{
  testAtomicRefPol<RAJA::omp_for_exec, RAJA::auto_atomic>();
}

#endif

#if defined(RAJA_ENABLE_CUDA)

CUDA_TEST(Atomic, CUDA_auto_AtomicRefForallFunctionalTest)
{
  testAtomicRefPol<RAJA::cuda_exec<256>, RAJA::auto_atomic>();
}

#endif

TEST(Atomic, basic_auto_AtomicRefForallFunctionalTest)
{
  testAtomicRefPol<RAJA::seq_exec, RAJA::auto_atomic>();
}

