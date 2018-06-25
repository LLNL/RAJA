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

namespace reduce
{
namespace detail
{

// Define the result type for all vector reductions
// as the vector's scalar type
template<typename T, size_t N, size_t U>
struct result_traits<RAJA::vec::Vector<T,N,U> > {
  using type = T;
};


//! specialization of ReduceSum for cuda_reduce
template <typename VecType, typename Reduce>
class ReduceVec :
public reduce::detail::BaseCombinable<VecType, Reduce, ReduceVec<VecType, Reduce>, typename VecType::scalar_type>
{
public:
  using result_type = typename VecType::scalar_type;
  using Base = reduce::detail::BaseCombinable<VecType, Reduce, ReduceVec, result_type>;

  using Base::Base;
  //! prohibit compiler-generated default ctor
  ReduceVec() = delete;

  //! enable operator+= for ReduceSum -- alias for combine()
  RAJA_HOST_DEVICE
  const ReduceVec& operator+=(VecType const &rhs) const
  {
    Base::my_data += rhs;
    return *this;
  }


  //! enable operator+= for ReduceSum -- alias for combine()
  RAJA_HOST_DEVICE
  const ReduceVec& operator+=(result_type const &rhs) const
  {
    Base::my_data += VecType(rhs);
    return *this;
  }

  result_type get() const { return Base::my_data.sum(); }


};

} // closing brace for detail namespace
} // closing brace for reduce namespace


RAJA_DECLARE_ALL_REDUCERS(vec_reduce, reduce::detail::ReduceVec)

}  // closing brace for RAJA namespace


#endif  // closing endif for header file include guard
