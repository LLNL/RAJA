//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "tests/test-forall-reduce-sanity.hpp"

#if defined(RAJA_ENABLE_CUDA)

#include "../test-forall-execpol.hpp"
#include "../test-reducepol.hpp"

// Cartesian product of types for CUDA tests
using CudaForallReduceSanityTypes =
  Test< camp::cartesian_product<ReduceSanityDataTypeList, 
                                CudaResourceList, 
                                CudaForallExecPols,
                                CudaReducePols>>::Types;

INSTANTIATE_TYPED_TEST_SUITE_P(Cuda,
                               ForallReduceSanityTest,
                               CudaForallReduceSanityTypes);

#endif
