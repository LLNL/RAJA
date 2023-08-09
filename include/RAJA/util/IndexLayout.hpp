/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   RAJA header file defining the IndexLayout class and IndexList classes.
 *
 ******************************************************************************
 */

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-23, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef RAJA_INDEXLAYOUT_HPP
#define RAJA_INDEXLAYOUT_HPP

#include "RAJA/util/Layout.hpp"

namespace RAJA 
{

template<typename IdxLin = Index_type>
struct ConditionalIndexList {

  RAJA_INLINE constexpr ConditionalIndexList(IdxLin* index_list_in) :
    index_list(index_list_in)
  {
  }

  IdxLin RAJA_INLINE RAJA_HOST_DEVICE constexpr get(const IdxLin idx) const
  {
    if (index_list) return index_list[idx];
    else return idx;
  }

  IdxLin* index_list;
};  

template<typename IdxLin = Index_type>
struct IndexList {

  RAJA_INLINE constexpr IndexList(IdxLin* index_list_in) :
    index_list(index_list_in){}

  IdxLin RAJA_INLINE RAJA_HOST_DEVICE constexpr get(const IdxLin idx) const
  {
    return index_list[idx];
  }

  IdxLin* index_list;
};

template<typename IdxLin = Index_type>
struct DirectIndex {

  RAJA_INLINE constexpr DirectIndex()
  {
  }

  IdxLin RAJA_INLINE RAJA_HOST_DEVICE constexpr get(const IdxLin idx) const
  {
    return idx;
  }

};

namespace internal
{

template<typename Range, typename IdxLin, typename... IndexTypes>
struct IndexLayout_impl;

template <camp::idx_t... RangeInts, typename IdxLin, typename... IndexTypes>
struct IndexLayout_impl<camp::idx_seq<RangeInts...>, IdxLin, IndexTypes...> {
  using IndexRange = camp::idx_seq<RangeInts...>;
  using IndexLinear = IdxLin;
  using Base = RAJA::detail::LayoutBase_impl<IndexRange, IdxLin>;
  Base base_;

  static constexpr size_t n_dims = sizeof...(RangeInts);

  template <typename... Types>
  constexpr RAJA_INLINE IndexLayout_impl(
      camp::tuple<IndexTypes...> index_tuple_in,
      Types... ns)
      : base_{(ns)...},
        tuple(index_tuple_in)
  {
  }

  template <typename... Indices>
  RAJA_INLINE RAJA_HOST_DEVICE constexpr IdxLin operator()(
      Indices... indices) const
  {
    return sum<IdxLin>(
      (base_.strides[RangeInts] * camp::get<RangeInts>(tuple).get(indices))...);
  }

  camp::tuple<IndexTypes...> tuple;

};

} // namespace internal


template <size_t n_dims = 1, typename IdxLin = Index_type, typename... IndexTypes>
struct IndexLayout
    : public internal::IndexLayout_impl<camp::make_idx_seq_t<n_dims>, IdxLin, IndexTypes...> {
  using Base =
      internal::IndexLayout_impl<camp::make_idx_seq_t<n_dims>, IdxLin, IndexTypes...>;

  using internal::IndexLayout_impl<camp::make_idx_seq_t<n_dims>,
                                    IdxLin, IndexTypes...>::IndexLayout_impl;

  constexpr RAJA_INLINE RAJA_HOST_DEVICE IndexLayout(
      const internal::IndexLayout_impl<camp::make_idx_seq_t<n_dims>, IdxLin, IndexTypes...>&
          rhs)
      : Base{rhs}
  {
  }

};

template <typename... IndexTypes>
auto make_index_tuple(IndexTypes... it) -> camp::tuple<IndexTypes...>
{
    return camp::tuple<IndexTypes...>(it...);
}

template <typename IdxLin = Index_type, typename... Types, typename... IndexTypes>
auto make_index_layout(
  camp::tuple<IndexTypes...> index_tuple_in,
  Types... ns) -> IndexLayout<sizeof...(Types), IdxLin, IndexTypes...>
{
    static_assert(sizeof...(Types) == sizeof...(IndexTypes), "");
    return IndexLayout<sizeof...(Types), IdxLin, IndexTypes...>(index_tuple_in, ns...);
}

}

#endif
