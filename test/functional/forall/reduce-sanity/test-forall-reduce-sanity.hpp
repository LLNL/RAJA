//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef __TEST_FORALL_REDUCE_SANITY_HPP__
#define __TEST_FORALL_REDUCE_SANITY_HPP__

#include "RAJA_test-base.hpp"
#include "RAJA_test-camp.hpp"
#include "RAJA_test-index-types.hpp"

#include "RAJA_test-forall-data.hpp"
#include "RAJA_test-forall-execpol.hpp"
#include "RAJA_test-reducepol.hpp"

//
// Data types for reduction sanity tests
//
using ReductionDataTypeList = camp::list< int,
                                          float,
                                          double >;

#include "tests/test-forall-reduce-sanity-sum.hpp"
#include "tests/test-forall-reduce-sanity-min.hpp"
#include "tests/test-forall-reduce-sanity-max.hpp"
#include "tests/test-forall-reduce-sanity-minloc.hpp"
#include "tests/test-forall-reduce-sanity-maxloc.hpp"


//
// Cartesian product of types for Sequential tests
//

using SequentialForallReduceSanityTypes =
  Test< camp::cartesian_product<ReductionDataTypeList, 
                                HostResourceList, 
                                SequentialForallReduceExecPols,
                                SequentialReducePols>>::Types;

#if defined(RAJA_ENABLE_OPENMP)
//
// Cartesian product of types for OpenMP tests
//

using OpenMPForallReduceSanityTypes =
  Test< camp::cartesian_product<ReductionDataTypeList, 
                                HostResourceList, 
                                OpenMPForallExecPols,
                                OpenMPReducePols>>::Types;
#endif

#if defined(RAJA_ENABLE_TBB)
//
// Cartesian product of types for OpenMP tests
//

using TBBForallReduceSanityTypes =
  Test< camp::cartesian_product<ReductionDataTypeList, 
                                HostResourceList, 
                                TBBForallExecPols,
                                TBBReducePols>>::Types; 
#endif

#if defined(RAJA_ENABLE_CUDA)
//
// Cartesian product of types for CUDA tests
//
using CudaForallReduceSanityTypes =
  Test< camp::cartesian_product<ReductionDataTypeList,
                                CudaResourceList,
                                CudaForallExecPols,
                                CudaReducePols>>::Types;
#endif

#if defined(RAJA_ENABLE_HIP)
//
// Cartesian product of types for HIP tests
//
using HipForallReduceSanityTypes =
  Test< camp::cartesian_product<ReduceSanityDataTypeList,
                                HipResourceList,
                                HipForallExecPols,
                                HipReducePols>>::Types;
#endif

#if defined(RAJA_ENABLE_TARGET_OPENMP)
//
// Cartesian product of types for OpenMP Target tests
//

using OpenMPTargetForallReduceSanityTypes =
  Test< camp::cartesian_product<ReductionDataTypeList,
                                OpenMPTargetResourceList,
                                OpenMPTargetForallExecPols,
                                OpenMPTargetReducePols>>::Types;
#endif

#endif  // __TEST_FORALL_REDUCE_SANITY_HPP__
