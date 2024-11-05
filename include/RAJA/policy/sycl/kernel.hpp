/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   RAJA header file containing constructs used to run kernel::forall
 *          traversals on GPU with SYCL.
 *
 ******************************************************************************
 */

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-24, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef RAJA_policy_sycl_kernel_HPP
#define RAJA_policy_sycl_kernel_HPP

#include "RAJA/policy/sycl/kernel/Conditional.hpp"
#include "RAJA/policy/sycl/kernel/SyclKernel.hpp"
#include "RAJA/policy/sycl/kernel/For.hpp"
#include "RAJA/policy/sycl/kernel/ForICount.hpp"
// #include "RAJA/policy/sycl/kernel/Hyperplane.hpp"
// #include "RAJA/policy/sycl/kernel/InitLocalMem.hpp"
#include "RAJA/policy/sycl/kernel/Lambda.hpp"
// #include "RAJA/policy/sycl/kernel/Reduce.hpp"
// #include "RAJA/policy/sycl/kernel/Sync.hpp"
#include "RAJA/policy/sycl/kernel/Tile.hpp"
#include "RAJA/policy/sycl/kernel/TileTCount.hpp"
#include "RAJA/policy/sycl/kernel/internal.hpp"

#endif  // closing endif for header file include guard
