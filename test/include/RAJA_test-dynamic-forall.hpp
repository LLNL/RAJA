//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-24, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

//
// Execution policy lists used throughout dynamic forall tests
//

#ifndef __RAJA_test_dynamic_execpol_HPP__
#define __RAJA_test_dynamic_execpol_HPP__

#include "RAJA/RAJA.hpp"
#include "camp/list.hpp"

using policy_list = camp::list<camp::list<
    RAJA::seq_exec,
    RAJA::simd_exec
#if defined(RAJA_ENABLE_OPENMP)
    ,
    RAJA::omp_parallel_for_exec
#endif
#if defined(RAJA_ENABLE_CUDA)
    ,
    RAJA::cuda_exec<256>,
    RAJA::cuda_exec<512>
#endif
#if defined(RAJA_ENABLE_HIP)
    ,
    RAJA::hip_exec<256>,
    RAJA::hip_exec<512>
#endif
    >>;


#endif  // __RAJA_test_dynamic_execpol_HPP__
