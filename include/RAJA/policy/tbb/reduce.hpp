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

#if defined(ENABLE_TBB)

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
template <typename T, typename Op>
class ReduceTBB
{
public:
  using Reduce = Op;
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
  operator T() const { return data->combine(Op{}); }

  /*!
   *  \return the calculated reduced value
   */
  T get() const { return operator T(); }

protected:
  /*!
   *  \return update the local value
   */
  void combine(const T &other) { data->local() = Op{}(data->local(), other); }
  /*!
   *  \return update the local value
   */
  void combine(const T &other) const
  {
    data->local() = Op{}(data->local(), other);
  }
};


template <typename T, bool doing_min = true>
struct ValueLoc {
  T val = doing_min ? operators::limits<T>::max() : operators::limits<T>::min();
  Index_type loc = -1;
  constexpr ValueLoc() = default;
  constexpr ValueLoc(ValueLoc const &) = default;
  ValueLoc &operator=(ValueLoc const &) = default;
  constexpr ValueLoc(T const &val) : val{val}, loc{-1} {}
  constexpr ValueLoc(T const &val, Index_type const &loc) : val{val}, loc{loc}
  {
  }
  operator T() const { return val; }
  bool operator<(ValueLoc const &rhs) const { return val < rhs.val; }
};


template <typename T>
using MinReduce = ReduceTBB<T, RAJA::operators::minimum<T>>;

template <typename T>
using MaxReduce = ReduceTBB<T, RAJA::operators::maximum<T>>;

template <typename T>
using SumReduce = ReduceTBB<T, RAJA::operators::plus<T>>;

template <typename T>
using MinLocReduce =
    ReduceTBB<ValueLoc<T>, RAJA::operators::minimum<ValueLoc<T>>>;

template <typename T>
using MaxLocReduce =
    ReduceTBB<ValueLoc<T, false>, RAJA::operators::maximum<ValueLoc<T, false>>>;

}

namespace operators
{

template <typename T, bool B>
struct limits<::RAJA::detail::ValueLoc<T, B>> : limits<T> {
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
class ReduceMin<tbb_reduce, T> : public detail::MinReduce<T>
{

public:
  using Base = detail::MinReduce<T>;

  //! prohibit compiler-generated default ctor
  ReduceMin() = delete;

  //! prohibit compiler-generated copy assignment
  ReduceMin &operator=(const ReduceMin &) = delete;

  //! compiler-generated copy constructor
  ReduceMin(const ReduceMin &) = default;

  //! compiler-generated move constructor
  ReduceMin(ReduceMin &&) = default;

  //! compiler-generated move assignment
  ReduceMin &operator=(ReduceMin &&) = default;

  explicit ReduceMin(T init_val, T initializer = operators::limits<T>::max())
      : Base(init_val, initializer)
  {
  }
  //! reducer function; updates the current instance's state
  /*!
   * Assumes each thread has its own copy of the object.
   */
  const ReduceMin &min(T rhs) const
  {
    this->combine(rhs);
    return *this;
  }

  //! reducer function; updates the current instance's state
  /*!
   * Assumes each thread has its own copy of the object.
   */
  ReduceMin &min(T rhs)
  {
    this->combine(rhs);
    return *this;
  }
};

/*!
 **************************************************************************
 *
 * \brief  MinLoc reducer class template for use in tbb execution.
 *
 **************************************************************************
 */
template <typename T>
class ReduceMinLoc<tbb_reduce, T> : public detail::MinLocReduce<T>
{
public:
  using Base = detail::MinLocReduce<T>;
  //! prohibit compiler-generated default ctor
  ReduceMinLoc() = delete;

  //! prohibit compiler-generated copy assignment
  ReduceMinLoc &operator=(const ReduceMinLoc &) = delete;

  //! compiler-generated copy constructor
  ReduceMinLoc(const ReduceMinLoc &) = default;

  //! compiler-generated move constructor
  ReduceMinLoc(ReduceMinLoc &&) = default;

  //! compiler-generated move assignment
  ReduceMinLoc &operator=(ReduceMinLoc &&) = default;

  //! constructor requires a default value for the reducer
  explicit ReduceMinLoc(T init_val, Index_type init_idx)
      : Base(typename Base::value_type(init_val, init_idx))
  {
  }
  //! reducer function; updates the current instance's state
  /*!
   * Assumes each thread has its own copy of the object.
   */
  const ReduceMinLoc &minloc(T rhs, Index_type loc) const
  {
    this->combine(typename Base::value_type(rhs, loc));
    return *this;
  }

  //! reducer function; updates the current instance's state
  /*!
   * Assumes each thread has its own copy of the object.
   */
  ReduceMinLoc &minloc(T rhs, Index_type loc)
  {
    this->combine(typename Base::value_type(rhs, loc));
    return *this;
  }

  Index_type getLoc() { return Base::get().loc; }
  operator T() const { return Base::get(); }
};

/*!
 **************************************************************************
 *
 * \brief  Max reducer class template for use in tbb execution.
 *
 **************************************************************************
 */
template <typename T>
class ReduceMax<tbb_reduce, T> : public detail::MaxReduce<T>
{
public:
  using Base = detail::MaxReduce<T>;
  using Base::Base;
  //! reducer function; updates the current instance's state
  /*!
   * Assumes each thread has its own copy of the object.
   */
  const ReduceMax &max(T rhs) const
  {
    this->combine(rhs);
    return *this;
  }

  //! reducer function; updates the current instance's state
  /*!
   * Assumes each thread has its own copy of the object.
   */
  ReduceMax &max(T rhs)
  {
    this->combine(rhs);
    return *this;
  }
};

/*!
 **************************************************************************
 *
 * \brief  Sum reducer class template for use in tbb execution.
 *
 **************************************************************************
 */
template <typename T>
class ReduceSum<tbb_reduce, T> : public detail::SumReduce<T>
{
public:
  using Base = detail::SumReduce<T>;
  using Base::Base;
  //! reducer function; updates the current instance's state
  /*!
   * Assumes each thread has its own copy of the object.
   */
  const ReduceSum &operator+=(T rhs) const
  {
    this->combine(rhs);
    return *this;
  }

  //! reducer function; updates the current instance's state
  /*!
   * Assumes each thread has its own copy of the object.
   */
  ReduceSum &operator+=(T rhs)
  {
    this->combine(rhs);
    return *this;
  }
};

/*!
 **************************************************************************
 *
 * \brief  MaxLoc reducer class template for use in tbb execution.
 *
 **************************************************************************
 */
template <typename T>
class ReduceMaxLoc<tbb_reduce, T> : public detail::MaxLocReduce<T>
{
public:
  using Base = detail::MaxLocReduce<T>;
  //! prohibit compiler-generated default ctor
  ReduceMaxLoc() = delete;

  //! prohibit compiler-generated copy assignment
  ReduceMaxLoc &operator=(const ReduceMaxLoc &) = delete;

  //! compiler-generated copy constructor
  ReduceMaxLoc(const ReduceMaxLoc &) = default;

  //! compiler-generated move constructor
  ReduceMaxLoc(ReduceMaxLoc &&) = default;

  //! compiler-generated move assignment
  ReduceMaxLoc &operator=(ReduceMaxLoc &&) = default;

  //! constructor requires a default value for the reducer
  explicit ReduceMaxLoc(T init_val, Index_type init_idx)
      : Base(typename Base::value_type(init_val, init_idx))
  {
  }
  //! reducer function; updates the current instance's state
  /*!
   * Assumes each thread has its own copy of the object.
   */
  const ReduceMaxLoc &maxloc(T rhs, Index_type loc) const
  {
    this->combine(typename Base::value_type(rhs, loc));
    return *this;
  }

  //! reducer function; updates the current instance's state
  /*!
   * Assumes each thread has its own copy of the object.
   */
  ReduceMaxLoc &maxloc(T rhs, Index_type loc)
  {
    this->combine(typename Base::value_type(rhs, loc));
    return *this;
  }

  Index_type getLoc() { return Base::get().loc; }

  operator T() const { return Base::get(); }
};

}  // closing brace for RAJA namespace

#endif  // closing endif for ENABLE_TBB guard

#endif  // closing endif for header file include guard
