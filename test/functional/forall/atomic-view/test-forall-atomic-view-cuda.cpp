//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "test-forall-atomic-view.hpp"

#if defined(RAJA_ENABLE_CUDA)
using CudaAtomicForallViewTypes = 
  Test< camp::cartesian_product< CudaForallExecPols,
                                 CudaAtomicPols,
                                 CudaResourceList,
                                 AtomicDataTypeList > >::Types;

INSTANTIATE_TYPED_TEST_SUITE_P( CudaTest,
                                ForallAtomicViewFunctionalTest,
                                CudaAtomicForallViewTypes );
#endif
