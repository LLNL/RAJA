/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Header file containing the RAJA Region API call
 *
 *             region<exec_policy>(loop body );
 *
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

#ifndef RAJA_region_HPP
#define RAJA_region_HPP

#include <functional>
#include <iterator>
#include <type_traits>

#include "RAJA/config.hpp"

#include "RAJA/internal/Iterators.hpp"
#include "RAJA/internal/Span.hpp"
#include "RAJA/policy/PolicyBase.hpp"
#include "RAJA/policy/sequential/region.hpp"

#include "RAJA/internal/fault_tolerance.hpp"
#include "RAJA/util/concepts.hpp"
#include "RAJA/util/types.hpp"

#include "RAJA/util/chai_support.hpp"


namespace RAJA
{
  
  template<typename ExecutionPolicy, typename LoopBody>
  void Region(LoopBody&& loop_body)
  {
    Region_impl(ExecutionPolicy(), loop_body);    
  }

}  // closing brace for RAJA namespace


#endif  // closing endif for header file include guard
