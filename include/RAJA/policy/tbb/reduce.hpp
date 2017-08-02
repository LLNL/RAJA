/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Header file containing RAJA reduction templates for
 *          tbb execution.
 *
 *          These methods should work on any platform.
 *
 ******************************************************************************
 */

#ifndef RAJA_reduce_tbb_HPP
#define RAJA_reduce_tbb_HPP

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
// DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONtbb
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
#include "RAJA/policy/tbb/policy.hpp"
#include "RAJA/util/types.hpp"

#include <tbb/tbb.h>
#include <memory>

namespace RAJA
{

namespace detail
{
template <typename T, typename Reduce>
class ReduceTBB
{
  std::shared_ptr<tbb::combinable<T>> data;

public:
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
  RAJA_HOST_DEVICE explicit ReduceTBB(T init_val, T initializer = T())
      : data(
            std::make_shared<tbb::combinable<T>>([=]() { return initializer; }))
  {
    data->local() = init_val;
  }

  //! return the reduced min value.
  /*!
   *  \return the calculated reduced value
   */
  RAJA_HOST_DEVICE operator T()
  {
    return data->combine([](const T &l, const T &r) {
      T res = l;
      Reduce{}(res, r);
      return res;
    });
  }

  //! return the reduced min value.
  /*!
   *  \return the calculated reduced value
   */
  RAJA_HOST_DEVICE T get() { return operator T(); }

  //! reducer function; updates the current instance's state
  /*!
   * Assumes each thread has its own copy of the object.
   */
  RAJA_HOST_DEVICE const ReduceTBB &operator+=(T rhs) const
  {
    Reduce()(data->local(), rhs);
    return *this;
  }

  //! reducer function; updates the current instance's state
  /*!
   * Assumes each thread has its own copy of the object.
   */
  RAJA_HOST_DEVICE ReduceTBB &operator+=(T rhs)
  {
    Reduce()(data->local(), rhs);
    return *this;
  }

private:
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
    : public detail::ReduceTBB<T, RAJA::reduce::min<T>>
{
  using detail::ReduceTBB<T, RAJA::reduce::min<T>>::ReduceTBB;
};

/*!
 **************************************************************************
 *
 * \brief  MinLoc reducer class template for use in tbb execution.
 *
 **************************************************************************
 */
template <typename T>
class ReduceMinLoc<tbb_reduce, T>
    : public detail::ReduceTBB<T, RAJA::reduce::minloc<T, Index_type>>
{
  using detail::ReduceTBB<T, RAJA::reduce::minloc<T, Index_type>>::ReduceTBB;
};

/*!
 **************************************************************************
 *
 * \brief  Max reducer class template for use in tbb execution.
 *
 **************************************************************************
 */
template <typename T>
class ReduceMax<tbb_reduce, T>
    : public detail::ReduceTBB<T, RAJA::reduce::max<T>>
{
  using detail::ReduceTBB<T, RAJA::reduce::max<T>>::ReduceTBB;
};

/*!
 **************************************************************************
 *
 * \brief  Sum reducer class template for use in tbb execution.
 *
 **************************************************************************
 */
template <typename T>
class ReduceSum<tbb_reduce, T>
    : public detail::ReduceTBB<T, RAJA::reduce::sum<T>>
{
  using detail::ReduceTBB<T, RAJA::reduce::sum<T>>::ReduceTBB;
};

/*!
 **************************************************************************
 *
 * \brief  MaxLoc reducer class template for use in tbb execution.
 *
 **************************************************************************
 */
template <typename T>
class ReduceMaxLoc<tbb_reduce, T>
    : public detail::ReduceTBB<T, RAJA::reduce::maxloc<T, Index_type>>
{
  using detail::ReduceTBB<T, RAJA::reduce::maxloc<T, Index_type>>::ReduceTBB;
};

}  // closing brace for RAJA namespace

#endif  // closing endif for header file include guard
