/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   RAJA header file defining Layout, a N-dimensional index calculator
 *
 ******************************************************************************
 */

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-21, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef RAJA_LAYOUT_HPP
#define RAJA_LAYOUT_HPP

#include "RAJA/config.hpp"

#include <iostream>
#include <limits>
#include <cassert>

#include "RAJA/index/IndexValue.hpp"

#include "RAJA/internal/foldl.hpp"

#include "RAJA/util/concepts.hpp"
#include "RAJA/util/Operators.hpp"
#include "RAJA/util/Permutations.hpp"

namespace RAJA
{

namespace detail
{

/*!
 * Helper function to compute the strides
 */
template <size_t j, size_t n_dims, typename IdxLin>
struct stride_calculator {
  RAJA_INLINE RAJA_HOST_DEVICE constexpr IdxLin operator()(
      IdxLin cur_stride,
      IdxLin const (&sizes)[n_dims]) const
  {
    return stride_calculator<j + 1, n_dims, IdxLin>{}(
        cur_stride * sizes[j], sizes);
  }
};
///
template <size_t n_dims, typename IdxLin>
struct stride_calculator<n_dims, n_dims, IdxLin> {
  RAJA_INLINE RAJA_HOST_DEVICE constexpr IdxLin operator()(
      IdxLin cur_stride,
      IdxLin const (&)[n_dims]) const
  {
    return cur_stride;
  }
};


/*!
 * Helper function to compute the strides with projections
 */
template <size_t j, size_t n_dims, typename IdxLin>
struct projection_stride_calculator {
  RAJA_INLINE RAJA_HOST_DEVICE constexpr IdxLin operator()(
      IdxLin cur_stride,
      IdxLin const (&sizes)[n_dims]) const
  {
    return projection_stride_calculator<j + 1, n_dims, IdxLin>{}(
        cur_stride * (sizes[j] ? sizes[j] : 1), sizes);
  }
};
///
template <size_t n_dims, typename IdxLin>
struct projection_stride_calculator<n_dims, n_dims, IdxLin> {
  RAJA_INLINE RAJA_HOST_DEVICE constexpr IdxLin operator()(
      IdxLin cur_stride,
      IdxLin const (&)[n_dims]) const
  {
    return cur_stride;
  }
};


/*!
 * Helper function to compute contribution of index to linear_index
 */
template <ptrdiff_t i_dim, ptrdiff_t stride_one_dim, typename IdxLin>
struct index_calculator {
  RAJA_INLINE RAJA_HOST_DEVICE constexpr IdxLin operator()(
      IdxLin index,
      IdxLin stride) const
  {
    // stride may not be 1
    return (index * stride);
  }
};
///
template <ptrdiff_t i_dim, typename IdxLin>
struct index_calculator<i_dim, i_dim, IdxLin> {
  RAJA_INLINE RAJA_HOST_DEVICE constexpr IdxLin operator()(
      IdxLin index,
      IdxLin ) const
  {
    // stride == 1, so don't bother with multiply
    return index;
  }
};

/*!
 * Helper function to compute indices from linear index
 */
template <ptrdiff_t i_dim, ptrdiff_t stride_one_dim, ptrdiff_t stride_max_dim,
          typename IdxLin>
struct to_index_calculator {
  RAJA_INLINE RAJA_HOST_DEVICE constexpr IdxLin operator()(
      IdxLin linear_index,
      IdxLin inv_stride,
      IdxLin inv_mod) const
  {
    // not stride one or max
    return ((linear_index / inv_stride) % inv_mod);
  }
};
///
template <ptrdiff_t i_dim, ptrdiff_t stride_max_dim,
          typename IdxLin>
struct to_index_calculator<i_dim, i_dim, stride_max_dim, IdxLin> {
  RAJA_INLINE RAJA_HOST_DEVICE constexpr IdxLin operator()(
      IdxLin linear_index,
      IdxLin ,
      IdxLin inv_mod) const
  {
    // inv_stride == 1, so don't bother with divide
    return (linear_index % inv_mod);
  }
};
///
template <ptrdiff_t i_dim, ptrdiff_t stride_one_dim,
          typename IdxLin>
struct to_index_calculator<i_dim, stride_one_dim, i_dim, IdxLin> {
  RAJA_INLINE RAJA_HOST_DEVICE constexpr IdxLin operator()(
      IdxLin linear_index,
      IdxLin inv_stride,
      IdxLin ) const
  {
    // inv_mod > (linear_index / inv_stride), so don't bother with modulo
    return (linear_index / inv_stride);
  }
};
///
template <ptrdiff_t i_dim,
          typename IdxLin>
struct to_index_calculator<i_dim, i_dim, i_dim, IdxLin> {
  RAJA_INLINE RAJA_HOST_DEVICE constexpr IdxLin operator()(
      IdxLin linear_index,
      IdxLin ,
      IdxLin ) const
  {
    // inv_stride == 1, so don't bother with divide
    // inv_mod > (linear_index / inv_stride), so don't bother with modulo
    return linear_index;
  }
};

struct LayoutBaseMarker {};

template <typename Range,
          typename IdxLin = Index_type,
          ptrdiff_t StrideOneDim = -1,
          ptrdiff_t StrideMaxDim = -1>
struct LayoutNoProjBase_impl;

template <camp::idx_t... RangeInts, typename IdxLin,
          ptrdiff_t StrideOneDim, ptrdiff_t StrideMaxDim>
struct LayoutNoProjBase_impl<camp::idx_seq<RangeInts...>, IdxLin,
                             StrideOneDim, StrideMaxDim>
    : LayoutBaseMarker
{
  using Self = LayoutNoProjBase_impl<camp::idx_seq<RangeInts...>, IdxLin,
                                     StrideOneDim, StrideMaxDim>;

  using IndexLinear = IdxLin;
  using IndexRange = camp::make_idx_seq_t<sizeof...(RangeInts)>;

  static_assert(std::is_same<camp::idx_seq<RangeInts...>, IndexRange>::value,
      "Range must in order");

  static constexpr size_t n_dims = sizeof...(RangeInts);
  static constexpr IdxLin limit = RAJA::operators::limits<IdxLin>::max();
  static constexpr ptrdiff_t stride_one_dim = StrideOneDim;
  static constexpr ptrdiff_t stride_max_dim = StrideMaxDim;

  IdxLin sizes[n_dims] = {0};
  IdxLin strides[n_dims] = {0};


  /*!
   * Default constructor with zero sizes and strides.
   */
  constexpr RAJA_INLINE LayoutNoProjBase_impl() = default;
  constexpr RAJA_INLINE LayoutNoProjBase_impl(LayoutNoProjBase_impl const &) = default;
  constexpr RAJA_INLINE LayoutNoProjBase_impl(LayoutNoProjBase_impl &&) = default;
  RAJA_INLINE LayoutNoProjBase_impl &operator=(LayoutNoProjBase_impl const &) =
      default;
  RAJA_INLINE LayoutNoProjBase_impl &operator=(LayoutNoProjBase_impl &&) =
      default;

  /*!
   * Construct a layout given the size of each dimension.
   * Calculates Strides s.t. stride is max at the left and 1 at the right.
   * Therefore if you use this constructor it is valid to set
   * StrideOneDim = n_dims-1 and StrideMaxDim = 0.
   */
  template <typename... Types,
            typename = concepts::enable_if<
                concepts::negate<std::is_base_of<LayoutBaseMarker, Types>>...>>
  RAJA_INLINE RAJA_HOST_DEVICE constexpr LayoutNoProjBase_impl(Types... ns)
      : sizes{static_cast<IdxLin>(stripIndexType(ns))...},
        strides{(detail::stride_calculator<RangeInts + 1, n_dims, IdxLin>{}(
            IdxLin(1), sizes))...}
  {
    static_assert(n_dims == sizeof...(Types),
                  "number of dimensions must match");
  }

  /*!
   *  Templated copy ctor from similar layout.
   */
  template <typename CIdxLin, ptrdiff_t CStrideOneDim, ptrdiff_t CStrideMaxDim>
  RAJA_INLINE RAJA_HOST_DEVICE constexpr LayoutNoProjBase_impl(
      const LayoutNoProjBase_impl<camp::idx_seq<RangeInts...>,
                                  CIdxLin, CStrideOneDim, CStrideMaxDim>
          &rhs)
      : sizes{static_cast<IdxLin>(rhs.sizes[RangeInts])...},
        strides{static_cast<IdxLin>(rhs.strides[RangeInts])...}
  {
  }

  /*!
   *  Construct a Layout given the size and stride of each dimension
   */
  RAJA_INLINE constexpr LayoutNoProjBase_impl(
      const std::array<IdxLin, n_dims> &sizes_in,
      const std::array<IdxLin, n_dims> &strides_in)
      : sizes{sizes_in[RangeInts]...},
        strides{strides_in[RangeInts]...}
  {
  }
  ///
  template <typename... SizeTypes, typename... StrideTypes>
  RAJA_INLINE RAJA_HOST_DEVICE constexpr LayoutNoProjBase_impl(
      const camp::tuple<SizeTypes...> &sizes_in,
      const camp::tuple<StrideTypes...> &strides_in)
      : sizes{camp::get<RangeInts>(sizes_in)...},
        strides{camp::get<RangeInts>(strides_in)...}
  {
  }

  /*!
   * Methods to performs bounds checking in layout objects
   */
  template<camp::idx_t N, typename Idx>
  RAJA_INLINE RAJA_HOST_DEVICE void BoundsCheckError(Idx idx) const
  {
    printf("Error at index %d, value %ld is not within bounds [0, %ld] \n",
           static_cast<int>(N), static_cast<long int>(idx), static_cast<long int>(sizes[N] - 1));
    RAJA_ABORT_OR_THROW("Out of bounds error \n");
  }

  template <camp::idx_t N>
  RAJA_INLINE RAJA_HOST_DEVICE void BoundsCheck() const
  {
  }

  template <camp::idx_t N, typename Idx, typename... Indices>
  RAJA_INLINE RAJA_HOST_DEVICE void BoundsCheck(Idx idx, Indices... indices) const
  {
    if(!(0<=idx && idx < static_cast<Idx>(sizes[N])))
    {
      BoundsCheckError<N>(idx);
    }
    RAJA_UNUSED_VAR(idx);
    BoundsCheck<N+1>(indices...);
  }

  /*!
   * Computes a linear space index from specified indices.
   * This is formed by the dot product of the indices and the layout strides.
   *
   * @param indices  Indices in the n-dimensional space of this layout
   * @return Linear space index.
   */

  template <typename... Indices>
  RAJA_INLINE RAJA_HOST_DEVICE RAJA_BOUNDS_CHECK_constexpr IdxLin operator()(
      Indices... indices) const
  {
    static_assert(n_dims == sizeof...(Indices),
        "Error: wrong number of indices");
#if defined (RAJA_BOUNDS_CHECK_INTERNAL)
    BoundsCheck<0>(indices...);
#endif
    // dot product of strides and indices
    return sum<IdxLin>(
      index_calculator<RangeInts, stride_one_dim, IdxLin>{}(
          indices, strides[RangeInts])... );
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
    static_assert(n_dims == sizeof...(Indices),
        "Error: wrong number of indices");
#if defined(RAJA_BOUNDS_CHECK_INTERNAL)
    IdxLin totSize = size();
    if(!(0 <= linear_index && linear_index < totSize)) {
      printf("Error! Linear index %ld is not within bounds [0, %ld]. \n",
             static_cast<long int>(linear_index), static_cast<long int>(totSize-1));
      RAJA_ABORT_OR_THROW("Out of bounds error \n");
     }
#endif
    return toIndicesHelper(linear_index, strides, sizes, std::forward<Indices>(indices)...);
  }

  /*!
   * Computes a total size of the layout's space.
   * This is the produce of each dimensions size.
   *
   * @return Total size spanned by indices
   */
  RAJA_INLINE RAJA_HOST_DEVICE constexpr IdxLin size() const
  {
    // Multiply together all of the sizes
    return foldl(RAJA::operators::multiplies<IdxLin>(), sizes[RangeInts]...);
  }

  template<camp::idx_t DIM>
  RAJA_INLINE
  RAJA_HOST_DEVICE
  constexpr
  IndexLinear get_dim_stride() const {
    return strides[DIM];
  }

protected:

  /*!
   * @internal
   *
   * Helper that uses the non-typed toIndices() function, and converts the
   * result to typed indices
   *
   */
  template <typename... Indices>
  RAJA_INLINE RAJA_HOST_DEVICE void toIndicesHelper(IdxLin linear_index,
                                                    IdxLin const (&inv_strides)[n_dims],
                                                    IdxLin const (&inv_mods)[n_dims],
                                                    Indices &&... indices) const
  {
    static_assert(n_dims == sizeof...(Indices),
        "Error: wrong number of indices");

    camp::sink((indices = (camp::decay<Indices>)(
      to_index_calculator<RangeInts, stride_one_dim, stride_max_dim, IdxLin>{}(
          linear_index, inv_strides[RangeInts], inv_mods[RangeInts]) ))...);
  }
};

template <camp::idx_t... RangeInts, typename IdxLin,
          ptrdiff_t StrideOneDim, ptrdiff_t StrideMaxDim>
constexpr size_t
    LayoutNoProjBase_impl<camp::idx_seq<RangeInts...>, IdxLin,
                    StrideOneDim, StrideMaxDim>::n_dims;
template <camp::idx_t... RangeInts, typename IdxLin,
          ptrdiff_t StrideOneDim, ptrdiff_t StrideMaxDim>
constexpr IdxLin
    LayoutNoProjBase_impl<camp::idx_seq<RangeInts...>, IdxLin,
                    StrideOneDim, StrideMaxDim>::limit;
template <camp::idx_t... RangeInts, typename IdxLin,
          ptrdiff_t StrideOneDim, ptrdiff_t StrideMaxDim>
constexpr ptrdiff_t
    LayoutNoProjBase_impl<camp::idx_seq<RangeInts...>, IdxLin,
                    StrideOneDim, StrideMaxDim>::stride_one_dim;
template <camp::idx_t... RangeInts, typename IdxLin,
          ptrdiff_t StrideOneDim, ptrdiff_t StrideMaxDim>
constexpr ptrdiff_t
    LayoutNoProjBase_impl<camp::idx_seq<RangeInts...>, IdxLin,
                    StrideOneDim, StrideMaxDim>::stride_max_dim;



template <typename Range,
          typename IdxLin = Index_type,
          ptrdiff_t StrideOneDim = -1,
          ptrdiff_t StrideMaxDim = -1>
struct LayoutBase_impl;

template <camp::idx_t... RangeInts, typename IdxLin,
          ptrdiff_t StrideOneDim, ptrdiff_t StrideMaxDim>
struct LayoutBase_impl<camp::idx_seq<RangeInts...>, IdxLin,
                       StrideOneDim, StrideMaxDim>
    : LayoutNoProjBase_impl<camp::idx_seq<RangeInts...>, IdxLin,
                            StrideOneDim, StrideMaxDim>
{
  using Self = LayoutBase_impl<camp::idx_seq<RangeInts...>, IdxLin,
                               StrideOneDim, StrideMaxDim>;
  using Base = LayoutNoProjBase_impl<camp::idx_seq<RangeInts...>, IdxLin,
                                     StrideOneDim, StrideMaxDim>;
  using Base::n_dims;
  using Base::limit;
  using Base::stride_one_dim;
  using Base::stride_max_dim;

  using Base::sizes;
  using Base::strides;
  IdxLin inv_strides[n_dims] = {0};
  IdxLin inv_mods[n_dims] = {0};


  /*!
   * Default constructor with zero sizes and strides.
   */
  constexpr RAJA_INLINE LayoutBase_impl() = default;
  constexpr RAJA_INLINE LayoutBase_impl(LayoutBase_impl const &) = default;
  constexpr RAJA_INLINE LayoutBase_impl(LayoutBase_impl &&) = default;
  RAJA_INLINE LayoutBase_impl &operator=(LayoutBase_impl const &) =
      default;
  RAJA_INLINE LayoutBase_impl &operator=(LayoutBase_impl &&) =
      default;

  /*!
   * Construct a layout given the size of each dimension.
   * Calculates Strides s.t. stride is max at the left and 1 at the right.
   * Therefore if you use this constructor it is valid to set
   * StrideOneDim = n_dims-1 and StrideMaxDim = 0.
   */
  template <typename... Types,
            typename = concepts::enable_if<
                concepts::negate<std::is_base_of<LayoutBaseMarker, Types>>...>>
  RAJA_INLINE RAJA_HOST_DEVICE constexpr LayoutBase_impl(Types... ns)
      : Base(camp::make_tuple(static_cast<IdxLin>(stripIndexType(ns))...),
             camp::make_tuple((detail::projection_stride_calculator<
                                  RangeInts + 1, n_dims, IdxLin>{}(
                static_cast<IdxLin>(stripIndexType(ns)) ? IdxLin(1) : IdxLin(0),
                {static_cast<IdxLin>(stripIndexType(ns))...}))...)),
        inv_strides{(strides[RangeInts] ? strides[RangeInts] : IdxLin(1))...},
        inv_mods{(sizes[RangeInts] ? sizes[RangeInts] : IdxLin(1))...}
  {
    static_assert(n_dims == sizeof...(Types),
                  "number of dimensions must match");
  }

  /*!
   *  Templated copy ctor from similar layout.
   */
  template <typename CIdxLin, ptrdiff_t CStrideOneDim, ptrdiff_t CStrideMaxDim>
  RAJA_INLINE RAJA_HOST_DEVICE constexpr LayoutBase_impl(
      const LayoutBase_impl<camp::idx_seq<RangeInts...>, CIdxLin,
                            CStrideOneDim, CStrideMaxDim>
          &rhs)
      : Base(static_cast<const LayoutNoProjBase_impl<camp::idx_seq<RangeInts...>, CIdxLin,
                                                     CStrideOneDim, CStrideMaxDim>&>(rhs)),
        inv_strides{static_cast<IdxLin>(rhs.inv_strides[RangeInts])...},
        inv_mods{static_cast<IdxLin>(rhs.inv_mods[RangeInts])...}
  {
  }


  /*!
   *  Construct a Layout given the size and stride of each dimension
   */
  RAJA_INLINE constexpr LayoutBase_impl(
      const std::array<IdxLin, n_dims> &sizes_in,
      const std::array<IdxLin, n_dims> &strides_in)
      : Base(sizes_in, strides_in),
        inv_strides{(strides[RangeInts] ? strides[RangeInts] : IdxLin(1))...},
        inv_mods{(sizes[RangeInts] ? sizes[RangeInts] : IdxLin(1))...}
  {
  }
  ///
  template < typename... SizeTypes, typename... StrideTypes >
  RAJA_INLINE RAJA_HOST_DEVICE constexpr LayoutBase_impl(
      const camp::tuple<SizeTypes...> &sizes_in,
      const camp::tuple<StrideTypes...> &strides_in)
      : Base(sizes_in, strides_in),
        inv_strides{(strides[RangeInts] ? strides[RangeInts] : IdxLin(1))...},
        inv_mods{(sizes[RangeInts] ? sizes[RangeInts] : IdxLin(1))...}
  {
  }

  /*!
   * Methods to performs bounds checking in layout objects
   */
  template<camp::idx_t N, typename Idx>
  RAJA_INLINE RAJA_HOST_DEVICE void BoundsCheckError(Idx idx) const
  {
    printf("Error at index %d, value %ld is not within bounds [0, %ld] \n",
           static_cast<int>(N), static_cast<long int>(idx), static_cast<long int>(sizes[N] - 1));
    RAJA_ABORT_OR_THROW("Out of bounds error \n");
  }

  template <camp::idx_t N>
  RAJA_INLINE RAJA_HOST_DEVICE void BoundsCheck() const
  {
  }

  template <camp::idx_t N, typename Idx, typename... Indices>
  RAJA_INLINE RAJA_HOST_DEVICE void BoundsCheck(Idx idx, Indices... indices) const
  {
    if((sizes[N] > 0) && !(0<=idx && idx < static_cast<Idx>(sizes[N])))
    {
      BoundsCheckError<N>(idx);
    }
    RAJA_UNUSED_VAR(idx);
    BoundsCheck<N+1>(indices...);
  }

  /*!
   * Computes a linear space index from specified indices.
   * This is formed by the dot product of the indices and the layout strides.
   *
   * @param indices  Indices in the n-dimensional space of this layout
   * @return Linear space index.
   */

  template <typename... Indices>
  RAJA_INLINE RAJA_HOST_DEVICE RAJA_BOUNDS_CHECK_constexpr IdxLin operator()(
      Indices&&... indices) const
  {
    static_assert(n_dims == sizeof...(Indices),
        "Error: wrong number of indices");
#if defined (RAJA_BOUNDS_CHECK_INTERNAL)
    BoundsCheck<0>(indices...);
#endif
    return sum<IdxLin>(
      index_calculator<RangeInts, stride_one_dim, IdxLin>{}(
          indices, strides[RangeInts])... );
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
    static_assert(n_dims == sizeof...(Indices),
        "Error: wrong number of indices");
#if defined(RAJA_BOUNDS_CHECK_INTERNAL)
    IdxLin totSize = Base::size();
    if((totSize > 0) &&
       (linear_index < 0 || linear_index >= totSize)) {
      printf("Error! Linear index %ld is not within bounds [0, %ld]. \n",
             static_cast<long int>(linear_index), static_cast<long int>(totSize-1));
      RAJA_ABORT_OR_THROW("Out of bounds error \n");
     }
#endif
    return Base::toIndicesHelper(linear_index, inv_strides, inv_mods, std::forward<Indices>(indices)...);
  }

  /*!
   * Computes a total size of the layout's space.
   * This is the produce of each dimensions size.
   * Projected dimensions are ignored.
   *
   * @return Total size spanned by indices
   */
  RAJA_INLINE RAJA_HOST_DEVICE constexpr IdxLin size() const
  {
    // Multiply together all of the sizes,
    // replacing 1 for any zero-sized dimensions
    return foldl(RAJA::operators::multiplies<IdxLin>(),
        ( (sizes[RangeInts] == IdxLin(0)) ? IdxLin(1) : sizes[RangeInts] )...);
  }
};


template <typename IdxLin, typename DimTuple, typename LayoutBase>
struct TypedLayoutBase_impl;

template <typename IdxLin, typename... DimTypes, typename LayoutBase>
struct TypedLayoutBase_impl<IdxLin, camp::tuple<DimTypes...>, LayoutBase>
    : LayoutBase
{

  using StrippedIdxLin = strip_index_type_t<IdxLin>;
  using Self = TypedLayoutBase_impl<IdxLin, camp::tuple<DimTypes...>, LayoutBase>;
  using Base = LayoutBase;
  using DimArr = std::array<StrippedIdxLin, sizeof...(DimTypes)>;

  using Base::n_dims;
  using Base::limit;
  using Base::stride_one_dim;
  using Base::stride_max_dim;

  static_assert(n_dims == sizeof...(DimTypes),
      "Error: number of dimension types does not match base layout");
  static_assert(std::is_same<StrippedIdxLin, typename Base::IndexLinear>::value,
      "Error: linear index types does not match base layout");

  // Pull in base constructors
  using Base::Base;


  /*!
   * Computes a linear space index from specified indices.
   * This is formed by the dot product of the indices and the layout strides.
   *
   * @param indices  Indices in the n-dimensional space of this layout
   * @return Linear space index.
   */
  RAJA_INLINE RAJA_HOST_DEVICE constexpr IdxLin operator()(
      DimTypes... indices) const
  {
    return IdxLin(Base::operator()(stripIndexType(indices)...));
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
  RAJA_INLINE RAJA_HOST_DEVICE void toIndices(IdxLin linear_index,
                                              DimTypes &... indices) const
  {
    toTypedIndicesHelper(typename Base::IndexRange{},
                         std::forward<IdxLin>(linear_index),
                         std::forward<DimTypes &>(indices)...);
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
  RAJA_INLINE RAJA_HOST_DEVICE void toTypedIndicesHelper(camp::idx_seq<RangeInts...>,
                                                         IdxLin linear_index,
                                                         Indices &... indices) const
  {
    static_assert(n_dims == sizeof...(Indices),
        "Error: wrong number of indices");
    typename Base::IndexLinear locals[n_dims];
    Base::toIndices(stripIndexType(linear_index), locals[RangeInts]...);
		camp::sink( (indices = static_cast<Indices>(locals[RangeInts]))... );
  }
};

}  // namespace detail

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
template <size_t n_dims,
          typename IdxLin = Index_type,
          ptrdiff_t StrideOneDim = -1,
          ptrdiff_t StrideMaxDim = -1>
using Layout =
    detail::LayoutBase_impl<camp::make_idx_seq_t<n_dims>, IdxLin,
                            StrideOneDim, StrideMaxDim>;

/*!
 * @brief A mapping of n-dimensional index space to a linear index space.
 *
 * This is the same as Layout, but does not allow projections.
 */
template <size_t n_dims,
          typename IdxLin = Index_type,
          ptrdiff_t StrideOneDim = -1,
          ptrdiff_t StrideMaxDim = -1>
using LayoutNoProj =
    detail::LayoutNoProjBase_impl<camp::make_idx_seq_t<n_dims>, IdxLin,
                                  StrideOneDim, StrideMaxDim>;

/*!
 * @brief A mapping of n-dimensional index space to a linear index space.
 *
 * This is the same as Layout, but allows the use of different types for each
 * index. Intended for use with strongly typed indices.
 */
template <typename IdxLin, typename DimTuple,
          ptrdiff_t StrideOneDim = -1,
          ptrdiff_t StrideMaxDim = -1>
using TypedLayout = detail::TypedLayoutBase_impl<IdxLin, DimTuple,
                        Layout<camp::tuple_size<DimTuple>::value, strip_index_type_t<IdxLin>,
                               StrideOneDim, StrideMaxDim>>;

/*!
 * @brief A mapping of n-dimensional index space to a linear index space.
 *
 * This is the same as TypedLayout, but does not allow projections.
 */
template <typename IdxLin, typename DimTuple,
          ptrdiff_t StrideOneDim = -1,
          ptrdiff_t StrideMaxDim = -1>
using TypedLayoutNoProj = detail::TypedLayoutBase_impl<IdxLin, DimTuple,
                        LayoutNoProj<camp::tuple_size<DimTuple>::value, strip_index_type_t<IdxLin>,
                               StrideOneDim, StrideMaxDim>>;


/*!
 * Convert a non-stride-one Layout to a stride-1 Layout
 *
 */
template <ptrdiff_t s1_dim, size_t n_dims, typename IdxLin, ptrdiff_t StrideMaxDim>
RAJA_INLINE Layout<n_dims, IdxLin, s1_dim, StrideMaxDim> make_stride_one(
    Layout<n_dims, IdxLin, -1, StrideMaxDim> const &l)
{
  return Layout<n_dims, IdxLin, s1_dim, StrideMaxDim>(l);
}

/*!
 * Convert a non-stride-one TypedLayout to a stride-1 TypedLayout
 *
 */
template <ptrdiff_t s1_dim, typename IdxLin, typename IdxTuple, ptrdiff_t StrideMaxDim>
RAJA_INLINE TypedLayout<IdxLin, IdxTuple, s1_dim, StrideMaxDim> make_stride_one(
    TypedLayout<IdxLin, IdxTuple, -1, StrideMaxDim> const &l)
{
  // strip l to it's base-class type
  using Base = typename TypedLayout<IdxLin, IdxTuple, -1, StrideMaxDim>::Base;
  Base const &b = (Base const &)l;

  // Use non-typed layout to initialize new typed layout
  return TypedLayout<IdxLin, IdxTuple, s1_dim, StrideMaxDim>(b);
}


}  // namespace RAJA

#endif
