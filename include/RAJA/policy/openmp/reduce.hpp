/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Header file containing RAJA reduction templates for
 *          OpenMP execution.
 *
 *          These methods should work on any platform that supports OpenMP.
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

#ifndef RAJA_omp_reduce_HPP
#define RAJA_omp_reduce_HPP

#include "RAJA/config.hpp"

#if defined(RAJA_ENABLE_OPENMP)

#include "RAJA/util/types.hpp"

#include "RAJA/pattern/detail/reduce.hpp"
#include "RAJA/pattern/reduce.hpp"
#include "RAJA/policy/openmp/policy.hpp"
#include "RAJA/policy/openmp/target_reduce.hpp"

#include <omp.h>

#include <memory>
#include <vector>

namespace RAJA
{

namespace detail
{
template <typename T, typename Reduce>
class ReduceOMP
    : public reduce::detail::BaseCombinable<T, Reduce, ReduceOMP<T, Reduce>>
{
  using Base = reduce::detail::BaseCombinable<T, Reduce, ReduceOMP>;

public:
  //! prohibit compiler-generated default ctor
  ReduceOMP() = delete;

  using Base::Base;

  ~ReduceOMP()
  {
    if (Base::parent) {
#pragma omp critical(ompReduceCritical)
      Reduce()(Base::parent->local(), Base::my_data);
      Base::my_data = Base::identity;
    }
  }
};

} /* detail */

RAJA_DECLARE_ALL_REDUCERS(omp_reduce, detail::ReduceOMP)

///////////////////////////////////////////////////////////////////////////////
//
// Old ordered reductions are included below.
//
///////////////////////////////////////////////////////////////////////////////

namespace detail
{
template <typename T, typename Reduce>
class ReduceOMPOrdered
    : public reduce::detail::
          BaseCombinable<T, Reduce, ReduceOMPOrdered<T, Reduce>>
{
  using Base = reduce::detail::BaseCombinable<T, Reduce, ReduceOMPOrdered>;
  std::shared_ptr<std::vector<T>> data;

public:
  //! prohibit compiler-generated default ctor
  ReduceOMPOrdered() = delete;

  //! constructor requires a default value for the reducer
  explicit ReduceOMPOrdered(T init_val, T identity_)
      : Base(init_val, identity_),
        data(std::make_shared<std::vector<T>>(omp_get_max_threads(), identity_))
  {
  }

  ~ReduceOMPOrdered()
  {
    Reduce{}((*data)[omp_get_thread_num()], Base::my_data);
    Base::my_data = Base::identity;
  }

  T get_combined() const
  {
    if (Base::my_data != Base::identity) {
      Reduce{}((*data)[omp_get_thread_num()], Base::my_data);
      Base::my_data = Base::identity;
    }

    T res = Base::identity;
    for (size_t i = 0; i < data->size(); ++i) {
      Reduce{}(res, (*data)[i]);
    }
    return res;
  }
};

} /* detail */

RAJA_DECLARE_ALL_REDUCERS(omp_reduce_ordered, detail::ReduceOMPOrdered)

}  // closing brace for RAJA namespace

#endif  // closing endif for RAJA_ENABLE_OPENMP guard

#endif  // closing endif for header file include guard
