//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef __TEST_ATOMICPOL__
#define __TEST_ATOMICPOL__

#include "RAJA/RAJA.hpp"
#include "camp/list.hpp"

using SequentialAtomicPols =
  camp::list<
#if defined(RAJA_TEST_EXHAUSTIVE)
              RAJA::auto_atomic,
              RAJA::builtin_atomic,
#endif
              RAJA::seq_atomic
            >;

#if defined(RAJA_ENABLE_OPENMP)
using OpenMPAtomicPols =
  camp::list<
#if defined(RAJA_TEST_EXHAUSTIVE)
              RAJA::omp_atomic,
              RAJA::builtin_atomic,
#endif
              RAJA::auto_atomic
            >;
#endif  // RAJA_ENABLE_OPENMP

#if defined(RAJA_ENABLE_CUDA)
using CudaAtomicPols =
  camp::list<
#if defined(RAJA_TEST_EXHAUSTIVE)
              RAJA::auto_atomic,
#endif
              RAJA::cuda_atomic
            >;
#endif  // RAJA_ENABLE_CUDA

#if defined(RAJA_ENABLE_HIP)
using HipAtomicPols =
  camp::list<
#if defined(RAJA_TEST_EXHAUSTIVE)
               RAJA::auto_atomic,
#endif
               RAJA::hip_atomic
            >;
#endif  // RAJA_ENABLE_HIP

#if defined(RAJA_ENABLE_TARGET_OPENMP)
using OpenMPTargetAtomicPols = OpenMPAtomicPols;
#endif

#endif  // __TEST_ATOMICPOL__
