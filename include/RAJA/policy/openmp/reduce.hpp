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

#ifndef RAJA_omp_reduce_HPP
#define RAJA_omp_reduce_HPP

#include "RAJA/config.hpp"
#include "RAJA/pattern/detail/reduce.hpp"

#include <memory>
#include <vector>

#if defined(RAJA_ENABLE_OPENMP)

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

#include "RAJA/util/types.hpp"

#include "RAJA/internal/MemUtils_CPU.hpp"
#include "RAJA/pattern/reduce.hpp"
#include "RAJA/policy/openmp/policy.hpp"
#include "RAJA/policy/openmp/target_reduce.hpp"

#include <omp.h>

#include <unordered_map>

namespace RAJA
{

namespace detail
{
template <typename T, typename Reduce>
class ReduceOMP
{
  ReduceOMP const *parent = nullptr;
  T identity;
  T mutable my_data;

public:
  //! prohibit compiler-generated default ctor
  ReduceOMP() = delete;

  constexpr ReduceOMP(T init_val, T identity_)
      : identity{identity_}, my_data{init_val}
  {
  }
  constexpr ReduceOMP(ReduceOMP const &other)
      : parent{other.parent ? other.parent : &other},
        identity{other.identity},
        my_data{identity}
  {
  }
  ~ReduceOMP()
  {
    if (parent != this) {
#pragma omp critical(ompReduceCritical)
      Reduce()(parent->my_data, my_data);
    }
  }

  /*!
   *  \return the calculated reduced value
   */
  T get() const { return my_data; }

  /*!
   *  \return update the local value
   */
  void combine(T const &other) { Reduce{}(my_data, other); }
};

template <typename T, typename Reduce>
class ReduceOMPOrdered
{
  ReduceOMPOrdered *parent = nullptr;
  std::shared_ptr<std::vector<T>> data;
  T identity;
  T mutable my_data;

public:
  //! prohibit compiler-generated default ctor
  ReduceOMPOrdered() = delete;

  constexpr ReduceOMPOrdered(T init_val, T identity_)
      : parent{this},
        data{
            std::make_shared<std::vector<T>>(omp_get_max_threads(), identity_)},
        identity{identity_},
        my_data{init_val}
  {
  }
  constexpr ReduceOMPOrdered(ReduceOMPOrdered const &other)
      : parent{other.parent},
        data{other.data},
        identity{other.identity},
        my_data{identity}
  {
  }
  ~ReduceOMPOrdered() { Reduce{}((*data)[omp_get_thread_num()], my_data); }

  void combine(T const &other) { Reduce{}(my_data, other); }

  /*!
   *  \return the calculated reduced value
   */
  T get() const
  {
    if (my_data != identity) {
      Reduce{}((*data)[omp_get_thread_num()], my_data);
      my_data = identity;
    }

    T res = identity;
    for (size_t i = 0; i < data->size(); ++i) {
      Reduce{}(res, (*data)[i]);
    }
    return res;
  }
};

} /* detail */

template <typename T>
class ReduceMin<omp_reduce, T>
    : public detail::BaseReduceMin<T, detail::ReduceOMP>
{
public:
  using Base = detail::BaseReduceMin<T, detail::ReduceOMP>;
  using Base::Base;
};

template <typename T>
class ReduceMax<omp_reduce, T>
    : public detail::BaseReduceMax<T, detail::ReduceOMP>
{
public:
  using Base = detail::BaseReduceMax<T, detail::ReduceOMP>;
  using Base::Base;
};

template <typename T>
class ReduceSum<omp_reduce, T>
    : public detail::BaseReduceSum<T, detail::ReduceOMP>
{
public:
  using Base = detail::BaseReduceSum<T, detail::ReduceOMP>;
  using Base::Base;
};

template <typename T>
class ReduceMinLoc<omp_reduce, T>
    : public detail::BaseReduceMinLoc<T, detail::ReduceOMP>
{
public:
  using Base = detail::BaseReduceMinLoc<T, detail::ReduceOMP>;
  using Base::Base;
};

template <typename T>
class ReduceMaxLoc<omp_reduce, T>
    : public detail::BaseReduceMaxLoc<T, detail::ReduceOMP>
{
public:
  using Base = detail::BaseReduceMaxLoc<T, detail::ReduceOMP>;
  using Base::Base;
};

///////////////////////////////////////////////////////////////////////////////
//
// Old ordered reductions are included below.
//
///////////////////////////////////////////////////////////////////////////////

template <typename T>
class ReduceMin<omp_reduce_ordered, T>
    : public detail::BaseReduceMin<T, detail::ReduceOMPOrdered>
{
public:
  using Base = detail::BaseReduceMin<T, detail::ReduceOMPOrdered>;
  using Base::Base;
};

template <typename T>
class ReduceMax<omp_reduce_ordered, T>
    : public detail::BaseReduceMax<T, detail::ReduceOMPOrdered>
{
public:
  using Base = detail::BaseReduceMax<T, detail::ReduceOMPOrdered>;
  using Base::Base;
};

template <typename T>
class ReduceSum<omp_reduce_ordered, T>
    : public detail::BaseReduceSum<T, detail::ReduceOMPOrdered>
{
public:
  using Base = detail::BaseReduceSum<T, detail::ReduceOMPOrdered>;
  using Base::Base;
};

template <typename T>
class ReduceMinLoc<omp_reduce_ordered, T>
    : public detail::BaseReduceMinLoc<T, detail::ReduceOMPOrdered>
{
public:
  using Base = detail::BaseReduceMinLoc<T, detail::ReduceOMPOrdered>;
  using Base::Base;
};

template <typename T>
class ReduceMaxLoc<omp_reduce_ordered, T>
    : public detail::BaseReduceMaxLoc<T, detail::ReduceOMPOrdered>
{
public:
  using Base = detail::BaseReduceMaxLoc<T, detail::ReduceOMPOrdered>;
  using Base::Base;
};

}  // closing brace for RAJA namespace

#endif  // closing endif for RAJA_ENABLE_OPENMP guard

#endif  // closing endif for header file include guard
