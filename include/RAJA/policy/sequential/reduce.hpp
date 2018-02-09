/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Header file containing RAJA reduction templates for
 *          sequential execution.
 *
 *          These methods should work on any platform.
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

#ifndef RAJA_sequential_reduce_HPP
#define RAJA_sequential_reduce_HPP

#include "RAJA/config.hpp"

#include "RAJA/internal/MemUtils_CPU.hpp"
#include "RAJA/pattern/detail/reduce.hpp"
#include "RAJA/pattern/reduce.hpp"
#include "RAJA/policy/sequential/policy.hpp"
#include "RAJA/util/types.hpp"

namespace RAJA
{

namespace detail
{
template <typename T, typename Reduce>
class ReduceSeq
    : public reduce::detail::BaseCombinable<T, Reduce, ReduceSeq<T, Reduce>>
{
  using Base = reduce::detail::BaseCombinable<T, Reduce, ReduceSeq<T, Reduce>>;

public:
  //! prohibit compiler-generated default ctor
  ReduceSeq() = delete;

  using Base::Base;
};


} /* detail */

RAJA_DECLARE_ALL_REDUCERS(seq_reduce, detail::ReduceSeq)

}  // closing brace for RAJA namespace

#endif  // closing endif for header file include guard
