/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   RAJA header file containing constructs used to run kernel::forall
 *          traversals on GPU with CUDA.
 *
 ******************************************************************************
 */

#ifndef RAJA_policy_cuda_kernel_HPP
#define RAJA_policy_cuda_kernel_HPP

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


#include "RAJA/policy/cuda/kernel/Collapse.hpp"
#include "RAJA/policy/cuda/kernel/Conditional.hpp"
#include "RAJA/policy/cuda/kernel/CudaKernel.hpp"
#include "RAJA/policy/cuda/kernel/For.hpp"
#include "RAJA/policy/cuda/kernel/Hyperplane.hpp"
#include "RAJA/policy/cuda/kernel/Lambda.hpp"
#include "RAJA/policy/cuda/kernel/ShmemWindow.hpp"
#include "RAJA/policy/cuda/kernel/Sync.hpp"
#include "RAJA/policy/cuda/kernel/Thread.hpp"
#include "RAJA/policy/cuda/kernel/Tile.hpp"
#include "RAJA/policy/cuda/kernel/internal.hpp"

#endif  // closing endif for header file include guard
