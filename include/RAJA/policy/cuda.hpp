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
// Copyright (c) 2016-18, Lawrence Livermore National Security, LLC.
//
// Produced at the Lawrence Livermore National Laboratory
//
// LLNL-CODE-689114
//
// All rights reserved.
//
// This file is part of RAJA.
//
// For details about use and distribution, please read RAJA/LICENSE.
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef RAJA_cuda_HPP
#define RAJA_cuda_HPP

#include "RAJA/config.hpp"

#if defined(RAJA_ENABLE_CUDA)

#include <cuda.h>
#include <cuda_runtime.h>

#include "RAJA/policy/cuda/atomic.hpp"
#include "RAJA/policy/cuda/forall.hpp"
#include "RAJA/policy/cuda/policy.hpp"
#include "RAJA/policy/cuda/reduce.hpp"
#if defined(__NVCC__)
#include "RAJA/policy/cuda/scan.hpp"
#endif
#include "RAJA/policy/cuda/synchronize.hpp"

#include "RAJA/policy/cuda/forallN.hpp"

#include "RAJA/policy/cuda/shared_memory.hpp"

#include "RAJA/policy/cuda/kernel.hpp"

#endif  // closing endif for if defined(RAJA_ENABLE_CUDA)

#endif  // closing endif for header file include guard
