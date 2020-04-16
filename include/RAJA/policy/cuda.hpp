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
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef RAJA_cuda_HPP
#define RAJA_cuda_HPP

#include "RAJA/config.hpp"

#if (defined(__NVCC__) || (defined(__clang__) && defined(__CUDA__))) && defined(RAJA_ENABLE_CUDA)

#include <cuda.h>
#include <cuda_runtime.h>

#include "RAJA/policy/cuda/atomic.hpp"
#include "RAJA/policy/cuda/forall.hpp"
#include "RAJA/policy/cuda/policy.hpp"
#include "RAJA/policy/cuda/reduce.hpp"
#include "RAJA/policy/cuda/scan.hpp"
#include "RAJA/policy/cuda/kernel.hpp"
#include "RAJA/policy/cuda/synchronize.hpp"

#endif  // closing endif for if defined(RAJA_ENABLE_CUDA)

#endif  // closing endif for header file include guard
