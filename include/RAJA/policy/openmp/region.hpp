/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Header file containing RAJA omp region
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

#ifndef RAJA_region_openmp_HPP
#define RAJA_region_openmp_HPP

#include "RAJA/config.hpp"

#include "RAJA/util/types.hpp"

#include "RAJA/policy/openmp/policy.hpp"

#include "RAJA/internal/fault_tolerance.hpp"

#include "RAJA/pattern/detail/forall.hpp"

namespace RAJA
{
namespace policy
{
namespace omp
{


//
//////////////////////////////////////////////////////////////////////
//
// Add description here... 
//
//////////////////////////////////////////////////////////////////////
//


template <typename Func>
RAJA_INLINE void Region_impl(const omp_region &, Func &&body)
{

#pragma omp parallel 
  body();
}


}  // closing brace for omp namespace

}  // closing brace for policy namespace

}  // closing brace for RAJA namespace

#endif  // closing endif for header file include guard
