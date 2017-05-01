/*!
******************************************************************************
*
* \file
*
* \brief   Header file providing RAJA scan declarations.
*
******************************************************************************
*/

#ifndef RAJA_scan_openmp_HXX
#define RAJA_scan_openmp_HXX

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

#include "RAJA/policy/sequential/scan.hpp"
#include "RAJA/policy/openmp/policy.hpp"

#include <omp.h>

#include <algorithm>
#include <functional>
#include <iterator>
#include <type_traits>
#include <vector>

namespace RAJA
{
namespace impl
{
namespace scan
{

RAJA_INLINE
int firstIndex(int n, int p, int pid)
{
  return (static_cast<size_t>(n) * pid) / p;
}

/*!
        \brief explicit inclusive inplace scan given range, function, and
   initial value
*/
template <typename Iter, typename BinFn>
void inclusive_inplace(const ::RAJA::omp_parallel_for_exec&,
                       Iter begin,
                       Iter end,
                       BinFn f)
{
  using Value = typename ::std::iterator_traits<Iter>::value_type;
  const int n = end - begin;
  const int p = std::min(n, omp_get_max_threads());
  ::std::vector<Value> sums(p, Value());
#pragma omp parallel num_threads(p)
  {
    const int pid = omp_get_thread_num();
    const int i0 = firstIndex(n, p, pid);
    const int i1 = firstIndex(n, p, pid + 1);
    inclusive_inplace(::RAJA::seq_exec{}, begin + i0, begin + i1, f);
    sums[pid] = *(begin + i1 - 1);
#pragma omp barrier
#pragma omp single
    exclusive_inplace(
        ::RAJA::seq_exec{}, sums.data(), sums.data() + p, f, BinFn::identity);
    for (int i = i0; i < i1; ++i) {
      *(begin + i) = f(*(begin + i), sums[pid]);
    }
  }
}

/*!
        \brief explicit exclusive inplace scan given range, function, and
   initial value
*/
template <typename Iter, typename BinFn, typename ValueT>
void exclusive_inplace(const ::RAJA::omp_parallel_for_exec&,
                       Iter begin,
                       Iter end,
                       BinFn f,
                       ValueT v)
{
  using Value = typename ::std::iterator_traits<Iter>::value_type;
  const int n = end - begin;
  const int p = std::min(n, omp_get_max_threads());
  ::std::vector<Value> sums(p, v);
#pragma omp parallel num_threads(p)
  {
    const int pid = omp_get_thread_num();
    const int i0 = firstIndex(n, p, pid);
    const int i1 = firstIndex(n, p, pid + 1);
    const Value init = ((pid == 0) ? v : *(begin + i0 - 1));
#pragma omp barrier
    exclusive_inplace(seq_exec{}, begin + i0, begin + i1, f, init);
    sums[pid] = *(begin + i1 - 1);
#pragma omp barrier
#pragma omp single
    exclusive_inplace(
        seq_exec{}, sums.data(), sums.data() + p, f, BinFn::identity);
    for (int i = i0; i < i1; ++i) {
      *(begin + i) = f(*(begin + i), sums[pid]);
    }
  }
}

/*!
        \brief explicit inclusive scan given input range, output, function, and
   initial value
*/
template <typename Iter, typename OutIter, typename BinFn>
void inclusive(const ::RAJA::omp_parallel_for_exec& exec,
               Iter begin,
               Iter end,
               OutIter out,
               BinFn f)
{
  ::std::copy(begin, end, out);
  inclusive_inplace(exec, out, out + (end - begin), f);
}

/*!
        \brief explicit exclusive scan given input range, output, function, and
   initial value
*/
template <typename Iter, typename OutIter, typename BinFn, typename ValueT>
void exclusive(const ::RAJA::omp_parallel_for_exec& exec,
               Iter begin,
               Iter end,
               OutIter out,
               BinFn f,
               ValueT v)
{
  ::std::copy(begin, end, out);
  exclusive_inplace(exec, out, out + (end - begin), f, v);
}

}  // namespace scan

}  // namespace impl

}  // namespace RAJA

#endif
