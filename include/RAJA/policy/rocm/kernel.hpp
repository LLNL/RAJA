/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   RAJA header file containing constructs used to run kernel::forall
 *          traversals on GPU with ROCM.
 *
 ******************************************************************************
 */

#ifndef RAJA_policy_rocm_kernel_HPP
#define RAJA_policy_rocm_kernel_HPP

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


#include "RAJA/policy/rocm/MemUtils_ROCm.hpp"
#include "RAJA/policy/rocm/kernel/Collapse.hpp"
#include "RAJA/policy/rocm/kernel/Conditional.hpp"
#include "RAJA/policy/rocm/kernel/ROCmKernel.hpp"
#include "RAJA/policy/rocm/kernel/For.hpp"
#include "RAJA/policy/rocm/kernel/Hyperplane.hpp"
#include "RAJA/policy/rocm/kernel/Lambda.hpp"
#include "RAJA/policy/rocm/kernel/ShmemWindow.hpp"
#include "RAJA/policy/rocm/kernel/Sync.hpp"
#include "RAJA/policy/rocm/kernel/Thread.hpp"
#include "RAJA/policy/rocm/kernel/Tile.hpp"
#include "RAJA/policy/rocm/kernel/internal.hpp"

#endif  // closing endif for header file include guard
