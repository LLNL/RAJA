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
// Copyright (c) 2016-17, Lawrence Livermore National Security, LLC.
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

#ifndef RAJA_tbb_reduce_HPP
#define RAJA_tbb_reduce_HPP

#include "RAJA/config.hpp"

#if defined(RAJA_ENABLE_TBB)

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
  //! TBB native per-thread container
  std::shared_ptr<tbb::combinable<T>> data;

public:
  //! prohibit compiler-generated default ctor
  ReduceTBB() = delete;

  //! constructor requires a default value for the reducer
  explicit ReduceTBB(T init_val, T initializer)
      : data(
            std::make_shared<tbb::combinable<T>>([=]() { return initializer; }))
  {
    data->local() = init_val;
  }

  /*!
   *  \return the calculated reduced value
   */
  T get() const { return data->combine(typename Reduce::operator_type{}); }

  /*!
   *  \return update the local value
   */
  void combine(const T &other) { Reduce{}(this->local(), other); }

  /*!
   *  \return reference to the local value
   */
  T& local() { return data->local(); }
};
}

RAJA_DECLARE_ALL_REDUCERS(tbb_reduce, detail::ReduceTBB)

}  // closing brace for RAJA namespace

#endif  // closing endif for RAJA_ENABLE_TBB guard

#endif  // closing endif for header file include guard
