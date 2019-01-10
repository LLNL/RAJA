/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Header file containing the RAJA Region API call
 *
 *             \code region<exec_policy>(loop body ); \endcode
 *
 *
 ******************************************************************************
 */

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-19, Lawrence Livermore National Security, LLC.
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

#ifndef RAJA_region_HPP
#define RAJA_region_HPP

#include "RAJA/policy/sequential/region.hpp"

namespace RAJA
{

template <typename ExecutionPolicy, typename LoopBody>
void region(LoopBody&& loop_body)
{
  region_impl(ExecutionPolicy(), loop_body);
}

template <typename ExecutionPolicy, typename OuterBody, typename InnerBody>
void region(OuterBody&& outer_body, InnerBody&& inner_body)
{
  region_impl(ExecutionPolicy(), outer_body, inner_body);
}

}  // namespace RAJA


#endif  // closing endif for header file include guard
