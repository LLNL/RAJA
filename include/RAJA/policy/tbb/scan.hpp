/*!
******************************************************************************
*
* \file
*
* \brief   Header file providing RAJA scan declarations.
*
******************************************************************************
*/

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-18, Lawrence Livermore National Security, LLC.
//
// Produced at the Lawrence Livermore National Laboratory
//
// LLNL-CODE-689114
//
// All rights reserved.
//
// This file is part of RAJA.
//
// For details about use and distribution, please read RAJA/LICENSE.
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef RAJA_scan_tbb_HPP
#define RAJA_scan_tbb_HPP

#include "RAJA/config.hpp"

#include "RAJA/util/defines.hpp"

#include "RAJA/util/concepts.hpp"

#include "RAJA/policy/sequential/policy.hpp"

#include <tbb/tbb.h>

#include <algorithm>
#include <functional>
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
  T const init;

  constexpr scan_adapter(InIter const& in_,
                         OutIter out_,
                         Fn fn_,
                         T const& init_)
      : agg(init_), in(in_), out(out_), fn(fn_), init(init_)
  {
  }
  T get_agg() const { return agg; }

  scan_adapter(scan_adapter& b, tbb::split)
      : agg(Fn::identity()), in(b.in), out(b.out), fn(b.fn), init(b.init)
  {
  }
  void reverse_join(const scan_adapter& a) { agg = fn(agg, a.agg); }
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
    for (Index_type i = r.begin(); i < r.end(); ++i) {
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
    if (r.begin() == 0) this->agg = this->init;
    for (Index_type i = r.begin(); i < r.end(); ++i) {
      auto t = this->in[i];
      if (Tag::is_final_scan()) this->out[i] = this->agg;
      this->agg = this->fn(this->agg, t);
    }
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
      BinFn>{begin, begin, f, BinFn::identity()};
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
      BinFn>{begin, begin, f, v};
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
      BinFn>{begin, out, f, BinFn::identity()};
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
      BinFn>{begin, out, f, v};
  tbb::parallel_scan(tbb::blocked_range<Index_type>{0,
                                                    std::distance(begin, end)},
                     adapter);
}

}  // namespace scan

}  // namespace impl

}  // namespace RAJA

#endif
