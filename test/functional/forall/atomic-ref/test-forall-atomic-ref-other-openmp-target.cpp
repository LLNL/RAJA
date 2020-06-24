//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "test-forall-atomic-ref.hpp"

#if defined(RAJA_ENABLE_TARGET_OPENMP)
using OmpTargetAtomicForallRefOtherTypes = 
  Test< camp::cartesian_product< OpenMPTargetForallExecPols,
                                 OpenMPAtomicPols,
                                 OpenMPTargetResourceList,
                                 AtomicDataTypeList > >::Types;

INSTANTIATE_TYPED_TEST_SUITE_P( OmpTargetTest,
                                ForallAtomicRefOtherFunctionalTest,
                                OmpTargetAtomicForallRefOtherTypes );
#endif
