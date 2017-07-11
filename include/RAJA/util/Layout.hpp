/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   RAJA header file defining layout operations for forallN templates.
 *
 ******************************************************************************
 */

#ifndef RAJA_LAYOUT_HPP
#define RAJA_LAYOUT_HPP

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

#include <iostream>
#include <limits>
#include "RAJA/config.hpp"
#include "RAJA/index/IndexValue.hpp"
#include "RAJA/internal/LegacyCompatibility.hpp"
#include "RAJA/util/Permutations.hpp"

namespace RAJA
{

template <typename Range, typename IdxLin = Index_type>
struct LayoutBase_impl {
};

template <size_t... RangeInts, typename IdxLin>
struct LayoutBase_impl<VarOps::index_sequence<RangeInts...>, IdxLin> {
public:
  typedef IdxLin IndexLinear;
  typedef VarOps::make_index_sequence<sizeof...(RangeInts)> IndexRange;

  static constexpr size_t n_dims = sizeof...(RangeInts);
  static constexpr size_t limit = RAJA::operators::limits<IdxLin>::max();

  // const char *index_types[sizeof...(RangeInts)];

  IdxLin sizes[n_dims];
  IdxLin strides[n_dims];
  IdxLin inv_strides[n_dims];
  IdxLin inv_mods[n_dims];


  /*!
   * Helper function to compute the strides
   */


  /*!
   * Default constructor with zero sizes and strides.
   */
  RAJA_INLINE RAJA_HOST_DEVICE LayoutBase_impl()
  {
    for(size_t i = 0;i < n_dims;++ i){
      sizes[i] = strides[i] = 0;
      inv_strides[i] = inv_mods[i] = 1;
    }
  }

  /*!
   * Construct a layout given the size of each dimension.
   *
   * @todo this should be constexpr in c++14 mode
   */
  template <typename... Types>
  RAJA_INLINE
  RAJA_HOST_DEVICE
  LayoutBase_impl(Types... ns)
      : sizes{convertIndex<IdxLin>(ns)...}
  {
    static_assert(n_dims == sizeof...(Types),
                  "number of dimensions must "
                  "match");
    for (size_t i = 0; i < n_dims; i++) {
      // If the size of dimension i is zero, then the stride is zero
      strides[i] = sizes[i] ? 1 : 0;

      for (size_t j = i+1; j < n_dims; j++) {
        // only take product of non-zero sizes
        strides[i] *= sizes[j] ? sizes[j] : 1;
      }
    }

    computeInverse();
  }

  /*!
   *  Copy ctor.
   */
  constexpr
  RAJA_INLINE
  RAJA_HOST_DEVICE
  LayoutBase_impl(const LayoutBase_impl<IndexRange, IdxLin> &rhs)
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
  RAJA_INLINE
  LayoutBase_impl(
      const std::array<IdxLin, n_dims> &sizes_in,
      const std::array<IdxLin, n_dims> &strides_in)
      : sizes{sizes_in[RangeInts]...},
        strides{strides_in[RangeInts]...}
  {
    computeInverse();
  }



  /*!
   * Computes a linear space index from specified indices.
   * This is formed by the dot product of the indices and the layout strides.
   *
   * @param indices  Indices in the n-dimensional space of this layout
   * @return Linear space index.
   */
  template <typename... Indices>
  RAJA_INLINE
  RAJA_HOST_DEVICE
  constexpr
  IdxLin operator()(
      Indices... indices) const
  {
    return VarOps::sum<IdxLin>((indices * strides[RangeInts])...);
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
  RAJA_INLINE
  RAJA_HOST_DEVICE
  void toIndices(IdxLin linear_index, Indices &... indices) const
  {
    VarOps::ignore_args(
        (indices = (linear_index / inv_strides[RangeInts]) % inv_mods[RangeInts])...);
  }

private:

  /*!
   * @internal
   *
   * Computes the inverse mapping used by toIndices given the forward mapping
   * described by strides[] and sizes[]
   */
  RAJA_INLINE
  RAJA_HOST_DEVICE
  void computeInverse(){
    // Inverse strides and mods map directly from strides and sizes,
    // except when a size (or stride) is zero for a projective layout.
    // In this case, having a stride and size of 1 will ensure that
    // toIndices for that dimension is always 0
    for (size_t i = 0; i < n_dims; i++) {
      inv_strides[i] = strides[i] ? strides[i] : 1;
      inv_mods[i] =    sizes[i]   ? sizes[i]   : 1;
    }
  }
};

template <size_t... RangeInts, typename IdxLin>
constexpr size_t
    LayoutBase_impl<VarOps::index_sequence<RangeInts...>, IdxLin>::n_dims;
template <size_t... RangeInts, typename IdxLin>
constexpr size_t
    LayoutBase_impl<VarOps::index_sequence<RangeInts...>, IdxLin>::limit;

template <size_t n_dims, typename IdxLin = Index_type>
using Layout = LayoutBase_impl<VarOps::make_index_sequence<n_dims>, IdxLin>;

template <typename IdxLin, typename... DimTypes>
struct TypedLayout : public Layout<sizeof...(DimTypes), Index_type> {
  using Self = TypedLayout<IdxLin, DimTypes...>;
  using Base = Layout<sizeof...(DimTypes), Index_type>;
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
  RAJA_INLINE
  RAJA_HOST_DEVICE
  constexpr
  IdxLin operator()(Indices... indices) const
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
  RAJA_INLINE
  RAJA_HOST_DEVICE
  void toIndices(IdxLin linear_index, Indices &... indices) const
  {
    toIndicesHelper(VarOps::make_index_sequence<sizeof...(DimTypes)>{},
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
  template <typename... Indices, size_t... RangeInts>
  RAJA_INLINE
  RAJA_HOST_DEVICE
  void toIndicesHelper(
      VarOps::index_sequence<RangeInts...>,
      IdxLin linear_index,
      Indices &... indices) const
  {
    Index_type locals[sizeof...(DimTypes)];
    Base::toIndices(convertIndex<Index_type>(linear_index),
                    locals[RangeInts]...);
    VarOps::ignore_args((indices = Indices{locals[RangeInts]})...);
  }
};


}  // namespace RAJA

#endif
