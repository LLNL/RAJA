//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-19, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

///
/// Source file containing tests for CHAI with different RAJA policies
///

#include "gtest/gtest.h"

#include "chai/ExecutionSpaces.hpp"

#include "RAJA/RAJA.hpp"

static_assert(RAJA::detail::get_space<RAJA::seq_exec>::value == chai::CPU, "");
static_assert(RAJA::detail::get_space<
                  RAJA::ExecPolicy<RAJA::seq_segit, RAJA::seq_exec>>::value ==
                  chai::CPU,
              "");

#if defined(RAJA_ENABLE_OPENMP)
static_assert(RAJA::detail::get_space<RAJA::omp_parallel_for_exec>::value ==
                  chai::CPU,
              "");
#endif

#if defined(RAJA_ENABLE_CUDA)
static_assert(RAJA::detail::get_space<RAJA::cuda_exec<128>>::value == chai::GPU,
              "");
#endif

#if defined(RAJA_ENABLE_CUDA)
static_assert(
    RAJA::detail::get_space<
        RAJA::ExecPolicy<RAJA::seq_segit, RAJA::cuda_exec<128>>>::value ==
        chai::GPU,
    "");
#endif

static_assert(RAJA::detail::get_space<RAJA::KernelPolicy<
                      RAJA::statement::For<0, RAJA::seq_exec, RAJA::statement::Lambda<0>>>>::value ==
                  chai::CPU,
              "");

#if defined(RAJA_ENABLE_CUDA)


static_assert(
    RAJA::detail::get_space<
        RAJA::KernelPolicy<RAJA::statement::For<0, RAJA::seq_exec>>>::value ==
        chai::CPU,
    "");
static_assert(
    RAJA::detail::get_space<RAJA::KernelPolicy<RAJA::statement::CudaKernel<
            RAJA::statement::For<0, RAJA::seq_exec>>>>::value == chai::GPU,
    "");
#endif


TEST(ChaiPolicyTest, Default)
{
#if defined(RAJA_ENABLE_CUDA)
  std::cout
      << RAJA::detail::get_space<
             RAJA::ExecPolicy<RAJA::seq_segit, RAJA::cuda_exec<128>>>::value
      << std::endl;
#else
  std::cout << RAJA::detail::get_space<
                   RAJA::ExecPolicy<RAJA::seq_segit, RAJA::simd_exec>>::value
            << std::endl;
#endif

  ASSERT_EQ(true, true);
}
