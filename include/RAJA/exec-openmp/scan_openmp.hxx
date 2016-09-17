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
// For additional details, please also read raja/README-license.txt.
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

#include "RAJA/config.hxx"

#include "RAJA/exec-sequential/scan_sequential.hxx"

#include <omp.h>

#include <algorithm>
#include <utility>
#include <vector>

#include <sstream>

namespace RAJA
{

RAJA_INLINE
int firstIndex(int n, int p, int pid)
{
  return static_cast<size_t>(n * pid) / p;
}

template <typename Iter, typename BinFn, typename Value>
void inclusive_scan_inplace(omp_parallel_for_exec,
                            Iter begin,
                            Iter end,
                            BinFn f,
                            Value v)
{
  const int n = end - begin;
  const int p = omp_get_max_threads();
  std::vector<Value> sums(p, v);
#pragma omp parallel
  {
    const int pid = omp_get_thread_num();
    const int i0 = firstIndex(n, p, pid);
    const int i1 = firstIndex(n, p, pid + 1);
    inclusive_scan_inplace(seq_exec{}, begin + i0, begin + i1, f, v);
    sums[pid] = *(begin + i1 - 1);
#pragma omp barrier
#pragma omp single
    exclusive_scan_inplace(seq_exec{}, sums.data(), sums.data() + p, f, v);
    for (int i = i0; i < i1; ++i) {
      *(begin + i) = f(*(begin + i), sums[pid]);
    }
  }
}

template <typename Iter, typename BinFn, typename Value>
void exclusive_scan_inplace(omp_parallel_for_exec,
                            Iter begin,
                            Iter end,
                            BinFn f,
                            Value v)
{
  const int n = end - begin;
  const int p = omp_get_max_threads();
  std::vector<Value> sums(p, v);
#pragma omp parallel
  {
    const int pid = omp_get_thread_num();
    const int i0 = firstIndex(n, p, pid);
    const int i1 = firstIndex(n, p, pid + 1);
    const Value init = ((pid == 0) ? v : *(begin + i0 - 1));
#pragma omp barrier
    exclusive_scan_inplace(seq_exec{}, begin + i0, begin + i1, f, init);
    sums[pid] = *(begin + i1 - 1);
#pragma omp barrier
#pragma omp single
    exclusive_scan_inplace(seq_exec{}, sums.data(), sums.data() + p, f, v);
    for (int i = i0; i < i1; ++i) {
      *(begin + i) = f(*(begin + i), sums[pid]);
    }
  }
}

template <typename Iter, typename OutIter, typename BinFn, typename Value>
void inclusive_scan(omp_parallel_for_exec exec,
                    Iter begin,
                    Iter end,
                    OutIter out,
                    BinFn f,
                    Value v)
{
  std::copy(begin, end, out);
  inclusive_scan_inplace(exec, out, out + (end - begin), f, v);
}

template <typename Iter, typename OutIter, typename BinFn, typename Value>
void exclusive_scan(omp_parallel_for_exec exec,
                    Iter begin,
                    Iter end,
                    OutIter out,
                    BinFn f,
                    Value v)
{
  std::copy(begin, end, out);
  exclusive_scan_inplace(exec, out, out + (end - begin), f, v);
}

}  // namespace RAJA

#endif
