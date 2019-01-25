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
// Copyright (c) 2016-19, Lawrence Livermore National Security, LLC.
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


#ifndef RAJA_util_LocalArray_HPP
#define RAJA_util_LocalArray_HPP

#include "RAJA/config.hpp"

#include <iostream>
#include <type_traits>

#include "RAJA/util/StaticLayout.hpp"

namespace RAJA
{

//Policies for RAJA local arrays
struct cpu_tile_mem;
struct cuda_thread_mem;
struct cuda_shared_mem;

template<camp::idx_t ... Sizes>
using ParamList = camp::idx_seq<Sizes...>;

/*!
 * RAJA local array
 * Holds a pointer and information necessary
 * to allocate a static array.
 *
 * Once intialized they can be treated as an N dimensional array
 * on the CPU stack, CUDA thread private memory,
 * or CUDA shared memory. Intialization occurs within
 * the RAJA::Kernel statement ``InitLocalArray"
 *
 * An accessor is provided to enable multi-dimensional indexing.
 * Two versions are created below, a strongly typed version and
 * a non-strongly typed version.
 */
template<typename DataType, typename Perm, typename Sizes, typename... IndexTypes>
struct TypedLocalArray
{
};

template<typename DataType, camp::idx_t ... Perm, camp::idx_t ...Sizes, typename... IndexTypes>
struct TypedLocalArray<DataType, camp::idx_seq<Perm...>, RAJA::SizeList<Sizes...>, IndexTypes...>
{
  DataType *m_arrayPtr = nullptr;
  using element_t = DataType;
  using layout_t = StaticLayout<camp::idx_seq<Perm...>, Sizes...>;
  static const camp::idx_t NumElem = layout_t::size();

  RAJA_HOST_DEVICE
  element_t &operator()(IndexTypes ...indices) const
  {
    return  m_arrayPtr[layout_t::s_oper(stripIndexType(indices)...)];
  }
};

template<typename DataType, typename Perm, typename Sizes>
struct LocalArray
{
};

template<typename DataType, camp::idx_t ... Perm, camp::idx_t ...Sizes>
struct LocalArray<DataType, camp::idx_seq<Perm...>, RAJA::SizeList<Sizes...> >
{
  DataType *m_arrayPtr = nullptr;
  using element_t = DataType;
  using layout_t = StaticLayout<camp::idx_seq<Perm...>, Sizes...>;
  static const camp::idx_t NumElem = layout_t::size();

  template<typename ...Indices>
  RAJA_HOST_DEVICE
  element_t &operator()(Indices ...indices) const
  {
    return m_arrayPtr[layout_t::s_oper(indices...)];
  }

};


}  // end namespace RAJA


#endif
