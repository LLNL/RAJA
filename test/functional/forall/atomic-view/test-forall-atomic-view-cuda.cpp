//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-19, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

///
/// Source file containing basic functional tests for atomic operations with forall and views.
///

#include "tests/test-forall-atomic-view.hpp"

#include "../test-forall-atomic-utils.hpp"

#if defined(RAJA_ENABLE_CUDA)
using CudaAtomicForallViewTypes = Test< camp::cartesian_product<
                                                                 AtomicCudaExecs,
                                                                 AtomicCudaPols,
                                                                 CudaResourceList,
                                                                 AtomicTypeList >
                                      >::Types;

INSTANTIATE_TYPED_TEST_SUITE_P( CudaTest,
                                CudaForallAtomicViewFunctionalTest,
                                CudaAtomicForallViewTypes );
#endif
