//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "test-scan-exclusive.hpp"

// Sequential policy types to test
using SequentialExecTypes = camp::list< RAJA::seq_exec, 
                                        RAJA::loop_exec >;

using SequentialExclusiveScanTypes = 
  Test<cartesian_product< SequentialExecTypes, ListHostRes, OpTypes >>::Types;

INSTANTIATE_TYPED_TEST_SUITE_P(Sequential, 
                               ScanFunctionalTest, 
                               SequentialExclusiveScanTypes);
