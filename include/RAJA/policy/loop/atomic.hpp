/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   RAJA header file defining loop atomic operations.
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

#ifndef RAJA_policy_loop_atomic_HPP
#define RAJA_policy_loop_atomic_HPP

#include "RAJA/config.hpp"
#include "RAJA/util/defines.hpp"

#include "RAJA/policy/sequential/atomic.hpp"

namespace RAJA
{
namespace atomic
{

using loop_atomic = seq_atomic;

}  // closing namespace atomic

}  // closing namespace RAJA

#endif  // guard
