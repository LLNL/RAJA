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

#include <iostream>
#include <type_traits>

#include "RAJA/util/StaticLayout.hpp"

namespace RAJA
{

//RAJA memory object wrapper policies
struct cpu_tile_mem;

struct cuda_shared_mem;
struct cuda_priv_mem;

/*!
 * Memory Wrapper
 */
template<typename Pol, typename DataType>
struct MemWrapper
{
  DataType *m_MemObj = nullptr;
  using type = DataType;
  using element_t = typename DataType::element_t;
  using pol_t = Pol;

  template<typename... Indices>
  RAJA_HOST_DEVICE
  element_t &operator()(Indices... indices) const
  {
    return (*m_MemObj).data[DataType::layout_t::s_oper(indices...)];
  }
};


/*!
 * Memory Wrapper specialization 
 * Stores data in thread register space
 */
template<typename DataType>
struct MemWrapper<RAJA::cuda_priv_mem, DataType>
{
  DataType *m_MemObj = nullptr;
  using type = DataType;
  using element_t = typename DataType::element_t;
  using pol_t = RAJA::cuda_priv_mem;

  template<typename... Indices>
  RAJA_HOST_DEVICE
  element_t &operator()(Indices... indices) const
  {
    return (*m_MemObj).data[0];
  }
};


/*!
 * RAJA memory object (shared mem 3.0)
 */
template<typename T, typename Sizes>
struct MemObj{};

template<typename T, camp::idx_t ... Sizes>
struct MemObj<T, RAJA::SizeList<Sizes...> >
{
  using self_t = MemObj<T, SizeList<Sizes...> >;
  using element_t = T;
  using layout_t = StaticLayout<Sizes...>;
  static const camp::idx_t NoElem = layout_t::size();
  T data[NoElem];
};


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

  int offsets[sizeof...(Segments)] = {0}; //set to zero

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
        (offsets[Args] = stripIndexType(camp::get<Args>(offset_tuple)))...);
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
