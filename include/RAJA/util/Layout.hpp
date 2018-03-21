/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   RAJA header file defining layout operations for forallN templates.
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

#ifndef RAJA_LAYOUT_HPP
#define RAJA_LAYOUT_HPP

#include <iostream>
#include <limits>
#include "RAJA/config.hpp"
#include "RAJA/index/IndexValue.hpp"
#include "RAJA/internal/LegacyCompatibility.hpp"
#include "RAJA/util/Operators.hpp"
#include "RAJA/util/Permutations.hpp"

namespace RAJA
{

namespace detail
{

/*!
 * Functor that returns a*b[i] for all i!=exlcude_i,
 * and returns a for i==exclude_i.
 *
 * This allows Layout to more efficiently compute layouts where a single
 * dimension is stride-1.  Also, it allows compilers to better reason about
 * loop optimizations.
 */
template <ptrdiff_t i, ptrdiff_t exclude_i>
struct ConditionalMultiply {

  template <typename A, typename B, size_t n_dims>
  static RAJA_INLINE RAJA_HOST_DEVICE constexpr A multiply(A a,
                                                           B const (&b)[n_dims])
  {
    // regular product term
    return a * b[i];
  }
};

template <ptrdiff_t i>
struct ConditionalMultiply<i, i> {
  template <typename A, typename B, size_t n_dims>
  static RAJA_INLINE RAJA_HOST_DEVICE constexpr A multiply(A a,
                                                           B const (&)[n_dims])
  {
    // assume b[i]==1
    return a;
  }
};


template <typename Range,
          typename IdxLin = Index_type,
          ptrdiff_t StrideOneDim = -1>
struct LayoutBase_impl;

/*!
 * Helper function to compute the strides
 */

template <size_t j, size_t n_dims, typename IdxLin = Index_type>
struct stride_calculator {
  RAJA_INLINE RAJA_HOST_DEVICE constexpr IdxLin operator()(
      IdxLin cur_stride,
      IdxLin const (&sizes)[n_dims]) const
  {
    return stride_calculator<j + 1, n_dims, IdxLin>{}(
        cur_stride * (sizes[j] ? sizes[j] : 1), sizes);
  }
};
template <size_t n_dims, typename IdxLin>
struct stride_calculator<n_dims, n_dims, IdxLin> {
  RAJA_INLINE RAJA_HOST_DEVICE constexpr IdxLin operator()(
      IdxLin cur_stride,
      IdxLin const (&)[n_dims]) const
  {
    return cur_stride;
  }
};

template <camp::idx_t... RangeInts, typename IdxLin, ptrdiff_t StrideOneDim>
struct LayoutBase_impl<camp::idx_seq<RangeInts...>, IdxLin, StrideOneDim> {
public:
  typedef IdxLin IndexLinear;
  typedef camp::make_idx_seq_t<sizeof...(RangeInts)> IndexRange;

  static constexpr size_t n_dims = sizeof...(RangeInts);
  static constexpr size_t limit = RAJA::operators::limits<IdxLin>::max();
  static constexpr ptrdiff_t stride1_dim = StrideOneDim;

  // const char *index_types[sizeof...(RangeInts)];

  IdxLin sizes[n_dims];
  IdxLin strides[n_dims];
  IdxLin inv_strides[n_dims];
  IdxLin inv_mods[n_dims];


  /*!
   * Default constructor with zero sizes and strides.
   */
  RAJA_INLINE RAJA_HOST_DEVICE constexpr LayoutBase_impl()
      : sizes{0}, strides{0}, inv_strides{0}, inv_mods{0}
  {
  }

  /*!
   * Construct a layout given the size of each dimension.
   */
  template <typename... Types>
  RAJA_INLINE RAJA_HOST_DEVICE constexpr LayoutBase_impl(Types... ns)
      : sizes{convertIndex<IdxLin>(ns)...},
        strides{(detail::stride_calculator<RangeInts + 1, n_dims, IdxLin>{}(
            sizes[RangeInts] ? 1 : 0,
            sizes))...},
        inv_strides{(strides[RangeInts] ? strides[RangeInts] : 1)...},
        inv_mods{(sizes[RangeInts] ? sizes[RangeInts] : 1)...}
  {
    static_assert(n_dims == sizeof...(Types),
                  "number of dimensions must "
                  "match");
  }

  /*!
   *  Copy ctor.
   */
  template <typename CIdxLin, ptrdiff_t CStrideOneDim>
  constexpr RAJA_INLINE RAJA_HOST_DEVICE LayoutBase_impl(
      const LayoutBase_impl<camp::idx_seq<RangeInts...>, CIdxLin, CStrideOneDim>
          &rhs)
      : sizes{rhs.sizes[RangeInts]...},
        strides{rhs.strides[RangeInts]...},
        inv_strides{rhs.inv_strides[RangeInts]...},
        inv_mods{rhs.inv_mods[RangeInts]...}
  {
  }


  /*!
   *  Construct a Layout given the size and stride of each dimension
   */
  template <typename... Types>
  RAJA_INLINE constexpr LayoutBase_impl(
      const std::array<IdxLin, n_dims> &sizes_in,
      const std::array<IdxLin, n_dims> &strides_in)
      : sizes{sizes_in[RangeInts]...},
        strides{strides_in[RangeInts]...},
        inv_strides{(strides[RangeInts] ? strides[RangeInts] : 1)...},
        inv_mods{(sizes[RangeInts] ? sizes[RangeInts] : 1)...}
  {
  }


  /*!
   * Computes a linear space index from specified indices.
   * This is formed by the dot product of the indices and the layout strides.
   *
   * @param indices  Indices in the n-dimensional space of this layout
   * @return Linear space index.
   */
  template <typename... Indices>
  RAJA_INLINE RAJA_HOST_DEVICE constexpr IdxLin operator()(
      Indices... indices) const
  {
    // dot product of strides and indices
    return VarOps::sum<IdxLin>(
        ((IdxLin)detail::ConditionalMultiply<RangeInts, stride1_dim>::multiply(
            indices, strides))...);
  }


  /*!
   * Given a linear-space index, compute the n-dimensional indices defined
   * by this layout.
   *
   * Note that this operation requires 2n integer divide instructions
   *
   * @param linear_index  Linear space index to be converted to indices.
   * @param indices  Variadic list of indices to be assigned, number must match
   *                 dimensionality of this layout.
   */
  template <typename... Indices>
  RAJA_INLINE RAJA_HOST_DEVICE void toIndices(IdxLin linear_index,
                                              Indices &&... indices) const
  {
    VarOps::ignore_args((indices = (linear_index / inv_strides[RangeInts])
                                   % inv_mods[RangeInts])...);
  }

  /*!
   * Computes a total size of the layout's space.
   * This is the produce of each dimensions size.
   *
   * @return Total size spanned by indices
   */
  RAJA_INLINE RAJA_HOST_DEVICE constexpr IdxLin size() const
  {
    // Multiply together all of the sizes,
    // replacing 1 for any zero-sized dimensions
    return VarOps::foldl(RAJA::operators::multiplies<IdxLin>(),
                         (sizes[RangeInts] == 0 ? 1 : sizes[RangeInts])...);
  }
};

template <camp::idx_t... RangeInts, typename IdxLin, ptrdiff_t StrideOneDim>
constexpr size_t
    LayoutBase_impl<camp::idx_seq<RangeInts...>, IdxLin, StrideOneDim>::n_dims;
template <camp::idx_t... RangeInts, typename IdxLin, ptrdiff_t StrideOneDim>
constexpr size_t
    LayoutBase_impl<camp::idx_seq<RangeInts...>, IdxLin, StrideOneDim>::limit;
}

/*!
 * @brief A mapping of n-dimensional index space to a linear index space.
 *
 * This is particularly useful for creating multi-dimensional arrays with the
 * RAJA::View class, and other tasks such as mapping logical i,j,k zone
 * indices to a flattened zone id.
 *
 * For example:
 *
 *     // Create a layout object
 *     Layout<3> layout(5,7,11);
 *
 *     // Map from 3d index space to linear
 *     int lin = layout(2,3,1);   // lin=198
 *
 *     // Map from linear space to 3d indices
 *     int i, j, k;
 *     layout.toIndices(lin, i, j, k); // i,j,k = {2, 3, 1}
 *
 *
 * The above example creates a 3-d layout object with dimension sizes 5, 7,
 * and 11.  So the total index space covers 5*7*11=385 unique indices.
 * The operator() provides a mapping from 3-d indices to linear, and the
 * toIndices provides the inverse.
 *
 * The default striding has the first index (left-most)as the longest stride,
 * and the last (right-most) index with stride-1.
 *
 * To achieve other striding, see the RAJA::make_permuted_layout function
 * which can a permutation to the default striding.
 *
 * Layout supports projections, 0 or more dimensions may be of size zero.
 * In this case, the linear index space is invariant for those dimensions,
 * and toIndicies(...) will always produce a zero for that dimensions index.
 *
 * An example of a "projected" Layout:
 *
 *     // Create a layout with a degenerate dimensions
 *     Layout<3> layout(3, 0, 5);
 *
 *     // The second (J) index is projected out
 *     int lin1 = layout(0, 10, 0);   // lin1 = 0
 *     int lin2 = layout(0, 5, 1);    // lin2 = 1
 *
 *     // The inverse mapping always produces a 0 for J
 *     int i,j,k;
 *     layout.toIndices(lin2, i, j, k); // i,j,k = {0, 0, 1}
 *
 */
template <size_t n_dims, typename IdxLin = Index_type, ptrdiff_t StrideOne = -1>
using Layout =
    detail::LayoutBase_impl<camp::make_idx_seq_t<n_dims>, IdxLin, StrideOne>;

template <typename IdxLin, typename DimTuple, ptrdiff_t StrideOne = -1>
struct TypedLayout;

template <typename IdxLin, typename... DimTypes, ptrdiff_t StrideOne>
struct TypedLayout<IdxLin, camp::tuple<DimTypes...>, StrideOne>
    : public Layout<sizeof...(DimTypes), Index_type, StrideOne> {
  using Self = TypedLayout<IdxLin, camp::tuple<DimTypes...>, StrideOne>;
  using Base = Layout<sizeof...(DimTypes), Index_type, StrideOne>;
  using DimArr = std::array<Index_type, sizeof...(DimTypes)>;

  // Pull in base constructors
  using Base::Base;


  /*!
   * Computes a linear space index from specified indices.
   * This is formed by the dot product of the indices and the layout strides.
   *
   * @param indices  Indices in the n-dimensional space of this layout
   * @return Linear space index.
   */
  template <typename... Indices>
  RAJA_INLINE RAJA_HOST_DEVICE constexpr IdxLin operator()(
      Indices... indices) const
  {
    return convertIndex<IdxLin>(
        Base::operator()(convertIndex<Index_type>(indices)...));
  }


  /*!
   * Given a linear-space index, compute the n-dimensional indices defined
   * by this layout.
   *
   * Note that this operation requires 2n integer divide instructions
   *
   * @param linear_index  Linear space index to be converted to indices.
   * @param indices  Variadic list of indices to be assigned, number must match
   *                 dimensionality of this layout.
   */
  template <typename... Indices>
  RAJA_INLINE RAJA_HOST_DEVICE void toIndices(IdxLin linear_index,
                                              Indices &... indices) const
  {
    toIndicesHelper(camp::make_idx_seq_t<sizeof...(DimTypes)>{},
                    std::forward<IdxLin>(linear_index),
                    std::forward<Indices &>(indices)...);
  }

private:
  /*!
   * @internal
   *
   * Helper that uses the non-typed toIndices() function, and converts the
   * result to typed indices
   *
   */
  template <typename... Indices, camp::idx_t... RangeInts>
  RAJA_INLINE RAJA_HOST_DEVICE void toIndicesHelper(camp::idx_seq<RangeInts...>,
                                                    IdxLin linear_index,
                                                    Indices &... indices) const
  {
    Index_type locals[sizeof...(DimTypes)];
    Base::toIndices(convertIndex<Index_type>(linear_index),
                    locals[RangeInts]...);
    VarOps::ignore_args((indices = Indices{locals[RangeInts]})...);
  }
};


/*!
 * Convert a non-stride-one Layout to a stride-1 Layout
 *
 */
template <ptrdiff_t s1_dim, size_t n_dims, typename IdxLin>
RAJA_INLINE Layout<n_dims, IdxLin, s1_dim> make_stride_one(
    Layout<n_dims, IdxLin> const &l)
{
  return Layout<n_dims, IdxLin, s1_dim>(l);
}


/*!
 * Convert a non-stride-one TypedLayout to a stride-1 TypedLayout
 *
 */
template <ptrdiff_t s1_dim, typename IdxLin, typename IdxTuple>
RAJA_INLINE TypedLayout<IdxLin, IdxTuple, s1_dim> make_stride_one(
    TypedLayout<IdxLin, IdxTuple> const &l)
{
  // strip l to it's base-class type
  using Base = typename TypedLayout<IdxLin, IdxTuple>::Base;
  Base const &b = (Base const &)l;

  // Use non-typed layout to initialize new typed layout
  return TypedLayout<IdxLin, IdxTuple, s1_dim>(b);
}


}  // namespace RAJA

#endif
