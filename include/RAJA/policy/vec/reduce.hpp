/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Header file containing RAJA reduction templates for
 *          explicit SIMD vector execution.
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

#ifndef RAJA_policy_vec_reduce_HPP
#define RAJA_policy_vec_reduce_HPP

#include "RAJA/config.hpp"

#include "RAJA/util/types.hpp"

#include "RAJA/pattern/detail/reduce.hpp"
#include "RAJA/pattern/reduce.hpp"


#include <memory>
#include <vector>

namespace RAJA
{




//! specialization of ReduceSum for cuda_reduce
template <typename VecType, typename Reduce>
class ReduceVec :
public reduce::detail::BaseCombinable<VecType, Reduce, ReduceVec<VecType, Reduce>, typename VecType::scalar_type>
{
public:

  //! enable operator+= for ReduceSum -- alias for combine()
  RAJA_HOST_DEVICE
  const ReduceVec& operator+=(VecType rhs) const
  {
    value += rhs;
    return *this;
  }

private:
  VecType value;
};


}  // closing brace for RAJA namespace


#endif  // closing endif for header file include guard
