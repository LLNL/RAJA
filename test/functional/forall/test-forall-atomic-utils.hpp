//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef __TEST_FORALL_ATOMIC_UTILS_HPP__
#define __TEST_FORALL_ATOMIC_UTILS_HPP__

#include "RAJA/RAJA.hpp"

#include "test-forall-utils.hpp"

using AtomicSeqExecs = camp::list< RAJA::seq_exec,
                                   RAJA::loop_exec
                                   //RAJA::simd_exec  // fails for atomic-view with clang9
                                 >;

using AtomicSeqPols = camp::list<
#if defined(RAJA_TEST_EXHAUSTIVE)
                                  RAJA::auto_atomic,
                                  RAJA::builtin_atomic,
#endif
                                  RAJA::seq_atomic
                                >;

#if defined(RAJA_ENABLE_OPENMP)
using AtomicOmpExecs = camp::list<
#if defined(RAJA_TEST_EXHAUSTIVE)
                                   RAJA::omp_for_nowait_exec,
                                   RAJA::omp_parallel_for_exec,
#endif
                                   //RAJA::omp_parallel_exec<RAJA::seq_exec>, // fails for atomic-basic and atomic-ref-math
                                   RAJA::omp_for_exec
                                 >;

using AtomicOmpPols = camp::list<
#if defined(RAJA_TEST_EXHAUSTIVE)
                                  RAJA::omp_atomic,
                                  RAJA::builtin_atomic,
#endif
                                  RAJA::auto_atomic
                                >;
#endif  // RAJA_ENABLE_OPENMP

#if defined(RAJA_ENABLE_CUDA)
using AtomicCudaPols = camp::list<
#if defined(RAJA_TEST_EXHAUSTIVE)
                                   RAJA::auto_atomic,
#endif
                                   RAJA::cuda_atomic
                                 >;
#endif  // RAJA_ENABLE_CUDA

#if defined(RAJA_ENABLE_HIP)
using AtomicHipPols = camp::list<
#if defined(RAJA_TEST_EXHAUSTIVE)
                                   RAJA::auto_atomic,
#endif
                                   RAJA::hip_atomic
                                >;
#endif  // RAJA_ENABLE_HIP

//
// Atomic index types for segments
//
using AtomicDataTypeList = camp::list<
                                  RAJA::Index_type,
                                  int,
#if defined(RAJA_TEST_EXHAUSTIVE)
                                  unsigned,
                                  long long,
                                  unsigned long long,
                                  float,
#endif
                                  double
                                 >;

#endif  // __TEST_FORALL_ATOMIC_UTILS_HPP__
