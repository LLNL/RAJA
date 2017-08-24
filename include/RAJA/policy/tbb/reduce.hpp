/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Header file containing RAJA reduction templates for
 *          TBB execution.
 *
 *          These methods should work on any platform that supports TBB.
 *
 ******************************************************************************
 */

#ifndef RAJA_tbb_reduce_HPP
#define RAJA_tbb_reduce_HPP

#include "RAJA/config.hpp"

#if defined(RAJA_ENABLE_TBB)

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

#include "RAJA/internal/MemUtils_CPU.hpp"
#include "RAJA/pattern/detail/reduce.hpp"
#include "RAJA/pattern/reduce.hpp"
#include "RAJA/policy/tbb/policy.hpp"
#include "RAJA/util/types.hpp"

#include <tbb/tbb.h>
#include <memory>
#include <tuple>

namespace RAJA
{

namespace detail
{
template <typename T, typename Reduce>
class ReduceTBB
{
public:
  struct reduce_adapter {
    T operator()(T const &l, T const &r)
    {
      T tmp = l;
      Reduce{}(tmp, r);
      return tmp;
    }
  };
  using value_type = T;

  //! TBB native per-thread container
  std::shared_ptr<tbb::combinable<T>> data;

  //! prohibit compiler-generated default ctor
  ReduceTBB() = delete;

  //! prohibit compiler-generated copy assignment
  ReduceTBB &operator=(const ReduceTBB &) = delete;

  //! compiler-generated copy constructor
  ReduceTBB(const ReduceTBB &) = default;

  //! compiler-generated move constructor
  ReduceTBB(ReduceTBB &&) = default;

  //! compiler-generated move assignment
  ReduceTBB &operator=(ReduceTBB &&) = default;

  //! constructor requires a default value for the reducer
  explicit ReduceTBB(T init_val, T initializer = T())
      : data(
            std::make_shared<tbb::combinable<T>>([=]() { return initializer; }))
  {
    data->local() = init_val;
  }

  /*!
   *  \return the calculated reduced value
   */
  operator T() const { return data->combine(reduce_adapter{}); }

  /*!
   *  \return the calculated reduced value
   */
  T get() const { return operator T(); }

protected:
  /*!
   *  \return update the local value
   */
  void combine(const T &other) { Reduce{}(data->local(), other); }
  /*!
   *  \return update the local value
   */
  void combine(const T &other) const { Reduce{}(data->local(), other); }
};

}

/*!
 **************************************************************************
 *
 * \brief  Min reducer class template for use in tbb execution.
 *
 **************************************************************************
 */
template <typename T>
class ReduceMin<tbb_reduce, T>
    : public detail::BaseReduceMin<T, detail::ReduceTBB>
{
public:
  using Base = detail::BaseReduceMin<T, detail::ReduceTBB>;
  using Base::Base;
};

template <typename T>
class ReduceMax<tbb_reduce, T>
    : public detail::BaseReduceMax<T, detail::ReduceTBB>
{
public:
  using Base = detail::BaseReduceMax<T, detail::ReduceTBB>;
  using Base::Base;
};

template <typename T>
class ReduceSum<tbb_reduce, T>
    : public detail::BaseReduceSum<T, detail::ReduceTBB>
{
public:
  using Base = detail::BaseReduceSum<T, detail::ReduceTBB>;
  using Base::Base;
};

template <typename T>
class ReduceMinLoc<tbb_reduce, T>
    : public detail::BaseReduceMinLoc<T, detail::ReduceTBB>
{
public:
  using Base = detail::BaseReduceMinLoc<T, detail::ReduceTBB>;
  using Base::Base;
};

template <typename T>
class ReduceMaxLoc<tbb_reduce, T>
    : public detail::BaseReduceMaxLoc<T, detail::ReduceTBB>
{
public:
  using Base = detail::BaseReduceMaxLoc<T, detail::ReduceTBB>;
  using Base::Base;
};

}  // closing brace for RAJA namespace

#endif  // closing endif for RAJA_ENABLE_TBB guard

#endif  // closing endif for header file include guard
