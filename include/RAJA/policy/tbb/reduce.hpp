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

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-23, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef RAJA_tbb_reduce_HPP
#define RAJA_tbb_reduce_HPP

#include "RAJA/config.hpp"

#if defined(RAJA_ENABLE_TBB)

#include <memory>
#include <tuple>

#include <tbb/tbb.h>

#include "RAJA/internal/MemUtils_CPU.hpp"

#include "RAJA/pattern/detail/reduce.hpp"
#include "RAJA/pattern/reduce.hpp"

#include "RAJA/policy/tbb/policy.hpp"

#include "RAJA/util/types.hpp"

namespace RAJA
{

namespace detail
{
template <typename T, typename Reduce>
class ReduceTBB
{
  //! TBB native per-thread container
  std::shared_ptr<tbb::combinable<T>> data;

public:
  //! default constructor calls the reset method
  ReduceTBB() { reset(T(), T()); }

  //! constructor requires a default value for the reducer
  explicit ReduceTBB(T init_val, T initializer)
  {
    reset(init_val, initializer);
  }

  void reset(T init_val, T initializer)
  {
    data = std::shared_ptr<tbb::combinable<T>>(
        std::make_shared<tbb::combinable<T>>([=]() { return initializer; }));
    data->local() = init_val;
  }

  /*!
   *  \return the calculated reduced value
   */
  T get() const { return data->combine(typename Reduce::operator_type{}); }

  /*!
   *  \return update the local value
   */
  void combine(const T& other) { Reduce{}(this->local(), other); }

  /*!
   *  \return reference to the local value
   */
  T& local() { return data->local(); }
};
}  // namespace detail

RAJA_DECLARE_ALL_REDUCERS(tbb_reduce, detail::ReduceTBB)

}  // namespace RAJA

#endif  // closing endif for RAJA_ENABLE_TBB guard

#endif  // closing endif for header file include guard
