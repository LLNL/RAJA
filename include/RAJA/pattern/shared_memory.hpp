/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   RAJA header file containing shared memory object type
 *
 ******************************************************************************
 */

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-17, Lawrence Livermore National Security, LLC.
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

#ifndef RAJA_pattern_shared_memory_HPP
#define RAJA_pattern_shared_memory_HPP


#include "RAJA/config.hpp"
#include "RAJA/util/defines.hpp"
#include "camp/camp.hpp"
#include <stddef.h>
#include <memory>
#include <vector>

namespace RAJA
{



/*!
 * Creates a shared memory object with elements of type T.
 * The Policy determines
 */
template<typename SharedPolicy, typename T>
struct SharedMemory {
};




/*!
 * Identity map shared memory policy for SharedMemoryView
 */
struct ident_shmem {
  template<typename T>
  RAJA_INLINE
  RAJA_HOST_DEVICE
  static T apply(ptrdiff_t, T idx){
    return idx;
  }
};



/*!
 * Provides a multi-dimensional View of shared memory data.
 *
 * IndexPolicies provide mappings of each dimension into shmem indicies.
 * This is especially useful for mapping global loop indices into cuda block-
 * local indices.
 *
 * The dimension sizes specified are the block-local sizes, and define the
 * amount of shared memory to be requested.
 */
template<typename ShmemType, typename LayoutType, typename ... IndexPolicies>
struct SharedMemoryView {
  using self = SharedMemoryView<ShmemType, LayoutType, IndexPolicies...>;
  using shmem_type = ShmemType;
  using layout_type = LayoutType;
  using index_policies = camp::list<IndexPolicies...>;

  using element_type = typename shmem_type::element_type;

  static constexpr size_t n_dims = sizeof...(IndexPolicies);

  layout_type layout;
  shmem_type shared_memory;


  static_assert(n_dims == LayoutType::n_dims,
      "Number of index policies must match layout dimensions");


  /*!
   * Constructs from an existing Layout object
   * @param layout0
   */
  RAJA_INLINE
  explicit SharedMemoryView(LayoutType const &layout0) :
    layout(layout0),
    shared_memory(layout.size())
  {}

  /*!
   * Constructs from a list of index sizes
   */
  template<typename ... SizeTypes>
  RAJA_INLINE
  SharedMemoryView(SizeTypes ... sizes) :
    layout(sizes...),
    shared_memory(layout.size())
  {}


  RAJA_INLINE
  RAJA_HOST_DEVICE
  SharedMemoryView(self const &c) :
    layout(c.layout),
    shared_memory(c.shared_memory)
  {}

  template<camp::idx_t ... Seq, typename ... IdxTypes>
  RAJA_INLINE
  ptrdiff_t computeIndex(camp::idx_seq<Seq...> const &, IdxTypes ... idx) const {
    return layout( IndexPolicies::apply(layout.sizes[Seq], idx)... );
  }


  template<typename ... IdxTypes>
  RAJA_INLINE
  element_type &operator()(IdxTypes ... idx) const {

    // compute the indices with the layout and return our shared memory data
    using loop_idx = typename camp::make_idx_seq<sizeof...(IdxTypes)>::type;


    return shared_memory[computeIndex(loop_idx{}, idx ...)];
  }

};




namespace detail {
  void startSharedMemorySetup();

  size_t registerSharedMemoryObject(void *object, size_t shmem_size);

  void finishSharedMemorySetup();

  size_t getSharedMemorySize();

}



}  // namespace RAJA

#endif
