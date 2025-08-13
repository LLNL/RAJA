/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Header file containing RAJA headers for NVCC CUDA execution.
 *
 *          These methods work only on platforms that support CUDA.
 *
 ******************************************************************************
 */

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-25, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef RAJA_cuda_HPP
#define RAJA_cuda_HPP

#include "RAJA/config.hpp"

#if defined(RAJA_CUDA_ACTIVE)

#include <cuda.h>
#include <cuda_runtime.h>

#if !defined(RAJA_ENABLE_DESUL_ATOMICS)
#include "RAJA/policy/cuda/atomic.hpp"
#endif

#include "RAJA/policy/cuda/forall.hpp"
#include "RAJA/policy/cuda/policy.hpp"
#include "RAJA/policy/cuda/reduce.hpp"
#include "RAJA/policy/cuda/multi_reduce.hpp"
#include "RAJA/policy/cuda/scan.hpp"
#include "RAJA/policy/cuda/sort.hpp"
#include "RAJA/policy/cuda/kernel.hpp"
#include "RAJA/policy/cuda/synchronize.hpp"
#include "RAJA/policy/cuda/launch.hpp"
#include "RAJA/policy/cuda/WorkGroup.hpp"
#include "RAJA/policy/cuda/params/reduce.hpp"
#include "RAJA/policy/cuda/params/kernel_name.hpp"

#endif  // closing endif for if defined(RAJA_ENABLE_CUDA)

#endif  // closing endif for header file include guard
