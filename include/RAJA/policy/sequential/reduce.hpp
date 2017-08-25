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

#ifndef RAJA_reduce_sequential_HPP
#define RAJA_reduce_sequential_HPP

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016, Lawrence Livermore National Security, LLC.
//
// Produced at the Lawrence Livermore National Laboratory
//
// LLNL-CODE-689114
//
// All rights reserved.
//
// This file is part of RAJA.
//
// For additional details, please also read RAJA/LICENSE.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// * Redistributions of source code must retain the above copyright notice,
//   this list of conditions and the disclaimer below.
//
// * Redistributions in binary form must reproduce the above copyright notice,
//   this list of conditions and the disclaimer (as noted below) in the
//   documentation and/or other materials provided with the distribution.
//
// * Neither the name of the LLNS/LLNL nor the names of its contributors may
//   be used to endorse or promote products derived from this software without
//   specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL LAWRENCE LIVERMORE NATIONAL SECURITY,
// LLC, THE U.S. DEPARTMENT OF ENERGY OR CONTRIBUTORS BE LIABLE FOR ANY
// DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
// OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
// HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
// STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
// IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "RAJA/config.hpp"

#include "RAJA/internal/MemUtils_CPU.hpp"
#include "RAJA/pattern/reduce.hpp"
#include "RAJA/pattern/detail/reduce.hpp"
#include "RAJA/policy/sequential/policy.hpp"
#include "RAJA/util/types.hpp"

namespace RAJA
{

namespace detail
{
template <typename T, typename Reduce>
class ReduceSeq
{
  ReduceSeq const *parent = nullptr;
  T identity;
  T mutable my_data;

public:
  //! prohibit compiler-generated default ctor
  ReduceSeq() = delete;

  constexpr ReduceSeq(T init_val, T identity_ = T())
      : identity{identity_}, my_data{init_val}
  {
  }
  constexpr ReduceSeq(ReduceSeq const &other)
      : parent{other.parent ? other.parent : &other},
        identity{other.identity},
        my_data{identity}
  {
  }
  ~ReduceSeq()
  {
    if (parent) {
      Reduce()(parent->my_data, my_data);
    }
  }

  void combine(T const &other) { Reduce{}(my_data, other); }

  /*!
   *  \return the calculated reduced value
   */
  T get() const { return my_data; }
};


} /* detail */

/*!
 **************************************************************************
 *
 * \brief  Min reducer class template for use in sequential execution.
 *
 **************************************************************************
 */
template <typename T>
class ReduceMin<seq_reduce, T>
    : public detail::BaseReduceMin<T, detail::ReduceSeq>
{
public:
  using Base = detail::BaseReduceMin<T, detail::ReduceSeq>;
  using Base::Base;
};

template <typename T>
class ReduceMax<seq_reduce, T>
    : public detail::BaseReduceMax<T, detail::ReduceSeq>
{
public:
  using Base = detail::BaseReduceMax<T, detail::ReduceSeq>;
  using Base::Base;
};

template <typename T>
class ReduceSum<seq_reduce, T>
    : public detail::BaseReduceSum<T, detail::ReduceSeq>
{
public:
  using Base = detail::BaseReduceSum<T, detail::ReduceSeq>;
  using Base::Base;
};

template <typename T>
class ReduceMinLoc<seq_reduce, T>
    : public detail::BaseReduceMinLoc<T, detail::ReduceSeq>
{
public:
  using Base = detail::BaseReduceMinLoc<T, detail::ReduceSeq>;
  using Base::Base;
};

template <typename T>
class ReduceMaxLoc<seq_reduce, T>
    : public detail::BaseReduceMaxLoc<T, detail::ReduceSeq>
{
public:
  using Base = detail::BaseReduceMaxLoc<T, detail::ReduceSeq>;
  using Base::Base;
};

}  // closing brace for RAJA namespace

#endif  // closing endif for header file include guard
