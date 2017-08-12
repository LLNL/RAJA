/*!
******************************************************************************
*
* \file
*
* \brief   Header file providing RAJA scan declarations.
*
******************************************************************************
*/

#ifndef RAJA_scan_tbb_HPP
#define RAJA_scan_tbb_HPP

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

#include "RAJA/util/defines.hpp"

#include "RAJA/util/concepts.hpp"

#include "RAJA/policy/sequential/policy.hpp"

#include <tbb/tbb.h>

#include <algorithm>
#include <functional>
#include <iostream>
#include <iterator>

namespace RAJA
{
namespace impl
{
namespace scan
{

namespace detail
{
template <typename T, typename InIter, typename OutIter, typename Fn>
struct scan_adapter {
  T agg;
  InIter const& in;
  OutIter out;
  Fn fn;

  scan_adapter(InIter const& in_, OutIter out_, T const& identity, Fn fn_)
      : agg(identity), out(out_), in(in_), fn(fn_)
  {
  }
  T get_agg() const { return agg; }

  template <typename Tag>
  void operator()(const tbb::blocked_range<Index_type>& r, Tag)
  {
    T temp = agg;
    for (int i = r.begin(); i < r.end(); ++i) {
      temp = fn(temp, in[i]);
      if (Tag::is_final_scan()) out[i] = temp;
    }
    agg = temp;
  }
  scan_adapter(scan_adapter& b, tbb::split)
      : in(b.in), out(b.out), agg(Fn::identity), fn(b.fn)
  {
  }
  void reverse_join(const scan_adapter& a) { agg = fn(a.agg, agg); }
  void assign(const scan_adapter& b) { agg = b.agg; }
};

template <typename T, typename InIter, typename OutIter, typename Fn>
struct scan_adapter_inclusive : scan_adapter<T, InIter, OutIter, Fn> {

  using Base = scan_adapter<T, InIter, OutIter, Fn>;
  using Base::Base;
  template <typename Tag>
  void operator()(const tbb::blocked_range<Index_type>& r, Tag)
  {
    T temp = this->agg;
    for (int i = r.begin(); i < r.end(); ++i) {
      temp = this->fn(temp, this->in[i]);
      if (Tag::is_final_scan()) this->out[i] = temp;
    }
    this->agg = temp;
  }
};

template <typename T, typename InIter, typename OutIter, typename Fn>
struct scan_adapter_exclusive : scan_adapter<T, InIter, OutIter, Fn> {

  using Base = scan_adapter<T, InIter, OutIter, Fn>;
  using Base::Base;
  template <typename Tag>
  void operator()(const tbb::blocked_range<Index_type>& r, Tag)
  {
    T temp = this->agg;
    for (int i = r.begin(); i < r.end(); ++i) {
      auto t = this->in[i];
      if (Tag::is_final_scan()) this->out[i] = temp;
      temp = this->fn(temp, t);
    }
    this->agg = temp;
  }
};
}

/*!
        \brief explicit inclusive inplace scan given range, function, and
   initial value
*/
template <typename ExecPolicy, typename Iter, typename BinFn>
concepts::enable_if<type_traits::is_tbb_policy<ExecPolicy>> inclusive_inplace(
    const ExecPolicy&,
    Iter begin,
    Iter end,
    BinFn f)
{
  auto adapter = detail::scan_adapter_inclusive<
      typename std::remove_reference<decltype(*begin)>::type,
      Iter,
      Iter,
      BinFn>{begin, begin, BinFn::identity, f};
  tbb::parallel_scan(tbb::blocked_range<Index_type>{0,
                                                    std::distance(begin, end)},
                     adapter);
}

/*!
        \brief explicit exclusive inplace scan given range, function, and
   initial value
*/
template <typename ExecPolicy, typename Iter, typename BinFn, typename T>
concepts::enable_if<type_traits::is_tbb_policy<ExecPolicy>> exclusive_inplace(
    const ExecPolicy&,
    Iter begin,
    Iter end,
    BinFn f,
    T v)
{
  auto adapter = detail::scan_adapter_exclusive<
      typename std::remove_reference<decltype(*begin)>::type,
      Iter,
      Iter,
      BinFn>{begin, begin, v, f};
  tbb::parallel_scan(tbb::blocked_range<Index_type>{0,
                                                    std::distance(begin, end)},
                     adapter);
}

/*!
        \brief explicit inclusive scan given input range, output, function, and
   initial value
*/
template <typename ExecPolicy, typename Iter, typename OutIter, typename BinFn>
concepts::enable_if<type_traits::is_tbb_policy<ExecPolicy>> inclusive(
    const ExecPolicy&,
    const Iter begin,
    const Iter end,
    OutIter out,
    BinFn f)
{
  auto adapter = detail::scan_adapter_inclusive<
      typename std::remove_reference<decltype(*begin)>::type,
      Iter,
      OutIter,
      BinFn>{begin, out, BinFn::identity, f};
  tbb::parallel_scan(tbb::blocked_range<Index_type>{0,
                                                    std::distance(begin, end)},
                     adapter);
}

/*!
        \brief explicit exclusive scan given input range, output, function, and
   initial value
*/
template <typename ExecPolicy,
          typename Iter,
          typename OutIter,
          typename BinFn,
          typename T>
concepts::enable_if<type_traits::is_tbb_policy<ExecPolicy>> exclusive(
    const ExecPolicy&,
    const Iter begin,
    const Iter end,
    OutIter out,
    BinFn f,
    T v)
{
  auto adapter = detail::scan_adapter_exclusive<
      typename std::remove_reference<decltype(*begin)>::type,
      Iter,
      OutIter,
      BinFn>{begin, out, v, f};
  tbb::parallel_scan(tbb::blocked_range<Index_type>{0,
                                                    std::distance(begin, end)},
                     adapter);
}

}  // namespace scan

}  // namespace impl

}  // namespace RAJA

#endif
