/*!
******************************************************************************
*
* \file
*
* \brief   Header file providing RAJA scan declarations.
*
******************************************************************************
*/

#ifndef RAJA_scan_sequential_HXX
#define RAJA_scan_sequential_HXX

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

#include <algorithm>
#include <functional>
#include <iterator>
#include <type_traits>

namespace RAJA
{
namespace detail
{
namespace scan
{
namespace iterators
{

/*!
        \brief explicit inclusive inplace scan given range, function, and
   initial value
*/
template <typename Iter, typename BinFn, typename T>
void inclusive_inplace(const ::RAJA::seq_exec&,
                       Iter begin,
                       Iter end,
                       BinFn f,
                       T v)
{
  using Value = typename ::std::iterator_traits<Iter>::value_type;
  Value agg = *begin;
  while (++begin != end) {
    agg = f(*begin, agg);
    *begin = agg;
  }
}

/*!
        \brief explicit exclusive inplace scan given range, function, and
   initial value
*/
template <typename Iter, typename BinFn, typename T>
void exclusive_inplace(const ::RAJA::seq_exec&,
                       Iter begin,
                       Iter end,
                       BinFn f,
                       T v)
{
  using Value = typename ::std::iterator_traits<Iter>::value_type;
  const int n = end - begin;
  Value agg = v;
  for (int i = 0; i < n; ++i) {
    Value t = *(begin + i);
    *(begin + i) = agg;
    agg = f(agg, t);
  }
}

/*!
        \brief explicit inclusive scan given range, input, output, function, and
   initial value
*/
template <typename Iter, typename OutIter, typename BinFn, typename T>
void inclusive(const ::RAJA::seq_exec&,
               Iter begin,
               Iter end,
               OutIter out,
               BinFn f,
               T v)
{
  using Value = typename ::std::iterator_traits<Iter>::value_type;
  Value agg = *begin;
  *out++ = agg;
  for (Iter i = begin + 1; i != end; ++i) {
    agg = f(agg, *i);
    *out++ = agg;
  }
}

/*!
        \brief explicit exclusive scan given input range, output, function, and
   initial value
*/
template <typename Iter, typename OutIter, typename BinFn, typename T>
void exclusive(const ::RAJA::seq_exec&,
               Iter begin,
               Iter end,
               OutIter out,
               BinFn f,
               T v)
{
  using Value = typename ::std::iterator_traits<Iter>::value_type;
  Value agg = v;
  OutIter o = out;
  *o++ = v;
  for (Iter i = begin; i != end - 1; ++i, ++o) {
    agg = f(*i, agg);
    *o = agg;
  }
}

}  // namespace iterators

namespace iterable
{

/*!
        \brief explicit inclusive inplace scan given range, pointer, function,
   and
   initial value
*/
template <typename Iterable, typename BinFn, typename T>
void inclusive_inplace(const ::RAJA::seq_exec&,
                       Iterable range,
                       T* in,
                       BinFn f,
                       T v)
{
  auto begin = range.begin();
  auto end = range.end();
  T agg = in[*begin];
  while (++begin != end) {
    agg = f(in[*begin], agg);
    in[*begin] = agg;
  }
}

/*!
        \brief explicit exclusive inplace scan given range, pointer, function,
   and
   initial value
*/
template <typename Iterable, typename BinFn, typename T>
void exclusive_inplace(const ::RAJA::seq_exec&,
                       Iterable range,
                       T* in,
                       BinFn f,
                       T v)
{
  auto begin = range.begin();
  auto end = range.end();
  T agg = v;
  while (begin != end) {
    T t = in[*begin];
    in[*begin] = agg;
    agg = f(agg, t);
    ++begin;
  }
}

/*!
        \brief explicit inclusive scan given range, input, output, function, and
   initial value
*/
template <typename Iterable, typename TIn, typename TOut, typename BinFn>
void inclusive(const ::RAJA::seq_exec&,
               Iterable range,
               TIn* in,
               TOut* out,
               BinFn f,
               TOut v)
{
  auto begin = range.begin();
  auto end = range.end();
  TOut agg = in[*begin];
  out[*begin] = agg;
  while (++begin != end) {
    agg = f(agg, in[*begin]);
    out[*begin] = agg;
  }
}

/*!
        \brief explicit exclusive scan given range, input, output, function, and
   initial value
*/
template <typename Iterable, typename TIn, typename TOut, typename BinFn>
void exclusive(const ::RAJA::seq_exec&,
               Iterable range,
               TIn* in,
               TOut* out,
               BinFn f,
               TOut v)
{
  auto begin = range.begin();
  auto end = range.end();
  TOut agg = v;
  out[*begin] = agg;
  while (++begin != end) {
    agg = f(agg, in[*(begin - 1)]);
    out[*begin] = agg;
  }
}

}  // namespace iterable

namespace container
{
/*!
        \brief explicit in-place inclusive scan given container, function, and
   initial value
*/
template <typename Container, typename BinFn, typename T>
void inclusive_inplace(const ::RAJA::seq_exec& exec, Container& c, BinFn f, T v)
{
  ::RAJA::detail::scan::iterators::inclusive_inplace(
      exec, c.begin(), c.end(), f, v);
}

/*!
        \brief explicit in-place exclusive scan given container, function, and
   initial value
*/
template <typename Container, typename BinFn, typename T>
void exclusive_inplace(const ::RAJA::seq_exec& exec, Container& c, BinFn f, T v)
{
  ::RAJA::detail::scan::iterators::exclusive_inplace(
      exec, c.begin(), c.end(), f, v);
}

/*!
\brief explicit inclusive scan given input container, output container,
function, and
initial value
*/
template <typename ContainerIn,
          typename ContainerOut,
          typename BinFn,
          typename T>
void inclusive(const ::RAJA::seq_exec& e,
               const ContainerIn& in,
               ContainerOut& out,
               BinFn f,
               T v)
{
  ::RAJA::detail::scan::iterators::inclusive(
      e, in.begin(), in.end(), out.begin(), f, v);
}

/*!
\brief explicit exclusive scan given input container, output container,
function, and
initial value
*/
template <typename ContainerIn,
          typename ContainerOut,
          typename BinFn,
          typename T>
void exclusive(const ::RAJA::seq_exec& e,
               const ContainerIn& in,
               ContainerOut& out,
               BinFn f,
               T v)
{
  ::RAJA::detail::scan::iterators::exclusive(
      e, in.begin(), in.end(), out.begin(), f, v);
}

}  // namespace container

}  // namespace scan

}  // namespace detail

}  // namespace RAJA

#endif
