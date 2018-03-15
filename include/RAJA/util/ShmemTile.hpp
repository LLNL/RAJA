/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Header file for multi-dimensional shared memory tile Views.
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


#ifndef RAJA_util_ShmemTile_HPP
#define RAJA_util_ShmemTile_HPP


#include "RAJA/config.hpp"
#include "RAJA/util/StaticLayout.hpp"

#include <iostream>
#include <type_traits>

namespace RAJA
{

/*!
 * Provides a multi-dimensional tiled View of shared memory data.
 *
 * IndexPolicies provide mappings of each dimension into shmem indicies.
 * This is especially useful for mapping global loop indices into cuda block-
 * local indices.
 *
 * The dimension sizes specified are the block-local sizes, and define the
 * amount of shared memory to be requested.
 */
template <typename ShmemPol,
          typename T,
          typename Args,
          typename Sizes,
          typename Segments>
struct ShmemTile;

template <typename ShmemPol,
          typename T,
          camp::idx_t... Args,
          RAJA::Index_type... Sizes,
          typename... Segments>
struct ShmemTile<ShmemPol,
                 T,
                 RAJA::ArgList<Args...>,
                 SizeList<Sizes...>,
                 camp::tuple<Segments...>> : public internal::SharedMemoryBase {
  static_assert(sizeof...(Args) == sizeof...(Sizes),
                "ArgList and SizeList must be same length");

  using self_t = ShmemTile<ShmemPol,
                           T,
                           RAJA::ArgList<Args...>,
                           SizeList<Sizes...>,
                           camp::tuple<Segments...>>;
  // compute the index tuple that kernel is going to use
  using segment_tuple_t = camp::tuple<Segments...>;
  using index_tuple_t = RAJA::internal::index_tuple_from_segments<
      typename segment_tuple_t::TList>;

  // compute the indices that we are going to use
  using arg_tuple_t =
      camp::tuple<camp::at_v<typename index_tuple_t::TList, Args>...>;

  // typed layout to map indices to shmem space
  using layout_t =
      RAJA::TypedStaticLayout<typename arg_tuple_t::TList, Sizes...>;

  // shared memory object type
  using shmem_t = SharedMemory<ShmemPol, T, layout_t::s_size>;
  using element_t = T;
  shmem_t shmem;

  int offsets[sizeof...(Segments)];

  RAJA_SUPPRESS_HD_WARN
  RAJA_INLINE
  RAJA_HOST_DEVICE
  constexpr ShmemTile() : shmem() {}

  RAJA_SUPPRESS_HD_WARN
  RAJA_INLINE
  RAJA_HOST_DEVICE
  ShmemTile(ShmemTile const &c) : shmem(c.shmem) {}

  RAJA_INLINE
  RAJA_HOST_DEVICE
  size_t shmem_setup_buffer(size_t offset)
  {
    return shmem.shmem_setup_buffer(offset);
  }

  template <typename OffsetTuple>
  RAJA_INLINE RAJA_HOST_DEVICE void shmem_set_window(
      OffsetTuple const &offset_tuple)
  {
    VarOps::ignore_args(
        (offsets[Args] = convertIndex<int>(camp::get<Args>(offset_tuple)))...);
  }


  RAJA_SUPPRESS_HD_WARN
  RAJA_INLINE
  RAJA_HOST_DEVICE
  element_t &operator()(
      camp::at_v<typename index_tuple_t::TList, Args>... idx) const
  {
    return shmem[layout_t::s_oper((idx - offsets[Args])...)];
  }
};


}  // end namespace RAJA


#endif
