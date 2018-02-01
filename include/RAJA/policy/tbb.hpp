/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Header file containing RAJA headers for tbb execution.
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

#ifndef RAJA_tbb_HPP
#define RAJA_tbb_HPP

#include "RAJA/config.hpp"

#if defined(RAJA_ENABLE_TBB)

#include "RAJA/policy/tbb/forall.hpp"
#include "RAJA/policy/tbb/forallN.hpp"
#include "RAJA/policy/tbb/policy.hpp"
#include "RAJA/policy/tbb/reduce.hpp"
#include "RAJA/policy/tbb/scan.hpp"

#endif

#endif  // closing endif for header file include guard
