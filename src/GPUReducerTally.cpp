/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Header file containing RAJA reduction templates for CUDA execution.
 *
 *          These methods should work on any platform that supports
 *          CUDA devices.
 *
 ******************************************************************************
 */

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-21, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "RAJA/config.hpp"

#include <list>
#include <map>
#include <type_traits>

#include "RAJA/util/GPUReducerTally.hpp"

#if defined(RAJA_ENABLE_OPENMP) && !defined(_OPENMP)
#error RAJA configured with ENABLE_OPENMP, but OpenMP not supported by current compiler
#endif

namespace RAJA
{

namespace detail
{

#if defined(RAJA_ENABLE_OPENMP)

namespace impl
{

template < typename storage_type >
typename std::enable_if<(sizeof(storage_type) >= sizeof(omp::mutex))>::type
create_mutex(storage_type& storage)
{
  static_assert(alignof(storage_type) >= alignof(omp::mutex),
      "storage_type not properly aligned");
  new(&reinterpret_cast<omp::mutex&>(storage)) omp::mutex();
}
///
template < typename storage_type >
typename std::enable_if<(sizeof(storage_type) < sizeof(omp::mutex))>::type
create_mutex(storage_type& storage)
{
  static_assert(sizeof(storage_type) >= sizeof(omp::mutex*),
      "storage_type is too small");
  static_assert(alignof(storage_type) >= alignof(omp::mutex*),
      "storage_type not properly aligned");
  reinterpret_cast<omp::mutex*&>(storage) = new omp::mutex();
}

template < typename storage_type >
typename std::enable_if<(sizeof(storage_type) >= sizeof(omp::mutex)),
    omp::mutex&>::type
to_mutex(storage_type& storage)
{
  static_assert(alignof(storage_type) >= alignof(omp::mutex),
      "storage_type not properly aligned");
  return reinterpret_cast<omp::mutex&>(storage);
}
///
template < typename storage_type >
typename std::enable_if<(sizeof(storage_type) < sizeof(omp::mutex)),
    omp::mutex&>::type
to_mutex(storage_type& storage)
{
  static_assert(sizeof(storage_type) >= sizeof(omp::mutex*),
      "storage_type is too small");
  static_assert(alignof(storage_type) >= alignof(omp::mutex*),
      "storage_type not properly aligned");
  return *reinterpret_cast<omp::mutex*&>(storage);
}

template < typename storage_type >
typename std::enable_if<(sizeof(storage_type) >= sizeof(omp::mutex))>::type
destroy_mutex(storage_type& storage)
{
  static_assert(alignof(storage_type) >= alignof(omp::mutex),
      "storage_type not properly aligned");
  reinterpret_cast<omp::mutex&>(storage).~mutex();
}
///
template < typename storage_type >
typename std::enable_if<(sizeof(storage_type) < sizeof(omp::mutex))>::type
destroy_mutex(storage_type& storage)
{
  static_assert(sizeof(storage_type) >= sizeof(omp::mutex*),
      "storage_type is too small");
  static_assert(alignof(storage_type) >= alignof(omp::mutex*),
      "storage_type not properly aligned");
  delete reinterpret_cast<omp::mutex*&>(storage);
}

}


template <typename Resource>
void GPUReducerTally<Resource>::create_mutex()
{
  impl::create_mutex(m_mutex_storage);
}

template <typename Resource>
void GPUReducerTally<Resource>::destroy_mutex()
{
  impl::destroy_mutex(m_mutex_storage);
}

template <typename Resource>
void GPUReducerTally<Resource>::lock_and_call(void(*func)(void*), void* data)
{
  lock_guard<omp::mutex> lock(impl::to_mutex(m_mutex_storage));
  func(data);
}

#endif

template <typename Resource>
typename GPUReducerTally<Resource>::DevicePointers*
GPUReducerTally<Resource>::new_value_impl(
    identifier id,
    void*& pinned_ptr,        size_t pinned_memory_size,
    void*& device_ptr,        size_t device_memory_size,
    void*& device_zeroed_ptr, size_t device_zeroed_memory_size)
{
#if defined(RAJA_ENABLE_OPENMP)
  lock_guard<omp::mutex> lock(impl::to_mutex(m_mutex_storage));
#endif
  // find PointerLists for id
  id_to_lists_iterator id_i = m_id_to_lists.find(id);

  // allocate PointerLists if not found
  if (id_i == m_id_to_lists.end()) {
    auto ib = m_id_to_lists.emplace(id, PointerLists{});
    id_i = ib.first;
  }

  // allocate pinned_ptr
  id_i->second.value_list.emplace_front(
      resource_info::get_pinned_allocator().allocate(pinned_memory_size));

  // find DevicePointers with enough storage
  size_to_memory_iterator sm_i;
  for (sm_i = id_i->second.size_to_memory_map.lower_bound(device_memory_size);
       sm_i != id_i->second.size_to_memory_map.end();
       ++sm_i) {
    if (!sm_i->second.in_use) {
      break;
    }
  }

  // allocate DevicePointers if not found
  if (sm_i == id_i->second.size_to_memory_map.end()) {
    sm_i = id_i->second.size_to_memory_map.emplace(
      device_memory_size,
      DevicePointers{
        false,
        resource_info::get_device_allocator().
            allocate(device_memory_size),
        resource_info::get_device_zeroed_allocator().
            allocate(device_zeroed_memory_size)
      });
  }

  // indicate DevicePointers is in use
  sm_i->second.in_use = true;

  pinned_ptr        = id_i->second.value_list.front();
  device_ptr        = sm_i->second.device_ptr;
  device_zeroed_ptr = sm_i->second.device_zeroed_ptr;

  return &sm_i->second;
}

// explicit template instantiation of required GPUReducerTally
#if defined(RAJA_ENABLE_CUDA)
template class GPUReducerTally<camp::resources::Cuda>;
#endif

#if defined(RAJA_ENABLE_HIP)
template class GPUReducerTally<camp::resources::Hip>;
#endif

}  // end namespace detail

}  // namespace RAJA
