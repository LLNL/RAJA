/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   RAJA header file containing CUDA shared memory object and policy
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

#ifndef RAJA_policy_cuda_shared_memory_HPP
#define RAJA_policy_cuda_shared_memory_HPP


#include "RAJA/config.hpp"
#include "RAJA/pattern/shared_memory.hpp"
#include "RAJA/policy/cuda/policy.hpp"

#ifdef RAJA_ENABLE_CUDA

#include "RAJA/util/defines.hpp"

namespace RAJA
{







/*!
 * Shared Memory object for CUDA kernels.
 *
 * Indexing into this is [0, N), regardless of what block or thread you are.
 *
 * The data is always in CUDA shared memory, so it's block-local.
 */
template<typename T>
struct SharedMemory<cuda_shmem, T> {
  using self = SharedMemory<cuda_shmem, T>;
  using element_type = T;

  ptrdiff_t offset; // offset into dynamic shared memory, in bytes
  void *parent;     // pointer to original object
  size_t num_elements; // number of element in this object

  RAJA_INLINE
  RAJA_HOST_DEVICE
  explicit SharedMemory(size_t N) :
  offset(-1), parent((void*)this), num_elements(N) {}

  RAJA_INLINE
  RAJA_HOST_DEVICE
   SharedMemory(self const &c) :
  offset(c.offset), parent(c.parent), num_elements(c.num_elements)
  {
    // only implement the registration on the HOST
#ifndef __CUDA_ARCH__
    offset = RAJA::detail::registerSharedMemoryObject(parent, num_elements*sizeof(T));
#endif
  }


  template<typename IDX>
  RAJA_INLINE
  RAJA_DEVICE
  T &operator[](IDX i) const {
    // Get the pointer to beginning of dynamic shared memory
    extern __shared__ char my_ptr[];

    // Convert this to a pointer of type T at the beginning of OUR shared mem
    T *T_ptr = reinterpret_cast<T*>((&my_ptr[0]) + offset);

    // Return the i'th element of our buffer
    return T_ptr[i];
  }

};








template<typename T, typename LayoutType, typename ... IndexPolicies>
struct SharedMemoryView<SharedMemory<cuda_shmem, T>, LayoutType, IndexPolicies...> {
  using self = SharedMemoryView<SharedMemory<cuda_shmem, T>, LayoutType, IndexPolicies...>;
  using shmem_type = SharedMemory<cuda_shmem, T>;
  using layout_type = LayoutType;
  using index_policies = camp::list<IndexPolicies...>;

  using element_type = T;

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
  RAJA_DEVICE
  ptrdiff_t computeIndex(camp::idx_seq<Seq...> const &, IdxTypes ... idx) const {
    return layout( IndexPolicies::apply(layout.sizes[Seq], idx)... );
  }


  template<typename ... IdxTypes>
  RAJA_INLINE
  RAJA_DEVICE
  element_type &operator()(IdxTypes ... idx) const {

    // compute the indices with the layout and return our shared memory data
    using loop_idx = typename camp::make_idx_seq<sizeof...(IdxTypes)>::type;


    return shared_memory[computeIndex(loop_idx{}, idx ...)];
  }

};




}  // namespace RAJA

#endif // RAJA_ENABLE_CUDA

#endif
