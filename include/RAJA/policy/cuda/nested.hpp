/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   RAJA header file containing constructs used to run nested::forall
 *          traversals on GPU with CUDA.
 *
 ******************************************************************************
 */

#ifndef RAJA_policy_cuda_nested_HPP
#define RAJA_policy_cuda_nested_HPP

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


#include "RAJA/policy/cuda/nested/internal.hpp"
#include "RAJA/policy/cuda/nested/Collapse.hpp"
#include "RAJA/policy/cuda/nested/Conditional.hpp"
#include "RAJA/policy/cuda/nested/CudaKernel.hpp"
#include "RAJA/policy/cuda/nested/For.hpp"
#include "RAJA/policy/cuda/nested/Lambda.hpp"
#include "RAJA/policy/cuda/nested/Hyperplane.hpp"
#include "RAJA/policy/cuda/nested/ShmemWindow.hpp"
#include "RAJA/policy/cuda/nested/Sync.hpp"
#include "RAJA/policy/cuda/nested/Tile.hpp"
#include "RAJA/policy/cuda/nested/Thread.hpp"

#endif  // closing endif for header file include guard
