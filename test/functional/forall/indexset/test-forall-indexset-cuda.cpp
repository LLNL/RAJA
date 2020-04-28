//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "tests/test-forall-indexset.hpp"

#if defined(RAJA_ENABLE_CUDA)

#include "../test-forall-indexset-execpol.hpp"

// Cartesian product of types for Cuda tests
using CudaForallIndexSetTypes =
  Test< camp::cartesian_product<IdxTypeList, 
                                CudaResourceList, 
                                CudaForallIndexSetExecPols>>::Types;

INSTANTIATE_TYPED_TEST_SUITE_P(Cuda,
                               ForallIndexSetTest,
                               CudaForallIndexSetTypes);

#endif
