
/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Header file for RAJA operator Vector extensions
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

#ifndef RAJA_policy_vec_operators_HPP
#define RAJA_policy_vec_operators_HPP

#include "RAJA/config.hpp"

#include "RAJA/util/Operators.hpp"
#include "RAJA/policy/vec/Vector.hpp"

namespace RAJA
{

namespace operators
{


template<typename T, size_t N, size_t U>
struct limits<RAJA::vec::Vector<T,N,U>> :
  public RAJA::operators::limits<T>
{
};



}  // closing brace for operators namespace



namespace reduce 
{

template <typename T, size_t N, size_t U>
struct min<RAJA::vec::Vector<T, N, U>> : detail::op_adapter<RAJA::vec::Vector<T, N, U>, RAJA::operators::minimum> {
  using value_type = RAJA::vec::Vector<T, N, U>;

  RAJA_HOST_DEVICE RAJA_INLINE void operator()(value_type &val, value_type const &v) const
  { 
    val.min(v);
  }  
};

template <typename T, size_t N, size_t U>
struct max<RAJA::vec::Vector<T, N, U>> : detail::op_adapter<RAJA::vec::Vector<T, N, U>, RAJA::operators::maximum> {
  using value_type = RAJA::vec::Vector<T, N, U>;
  
  RAJA_HOST_DEVICE RAJA_INLINE void operator()(value_type &val, value_type const &v) const
  { 
    val.max(v);
  }  
};



}  // closing brace for reduce namespace


}  // closing brace for RAJA namespace

#endif  // closing endif for header file include guard
