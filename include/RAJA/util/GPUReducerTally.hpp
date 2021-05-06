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

#ifndef RAJA_util_GPUReducerTally_HPP
#define RAJA_util_GPUReducerTally_HPP

#include "RAJA/config.hpp"

#include <list>
#include <map>
#include <type_traits>

#include "RAJA/util/macros.hpp"
#include "RAJA/util/Allocator.hpp"
#include "RAJA/util/mutex.hpp"
#include "RAJA/util/types.hpp"
#include "RAJA/util/SoAPtr.hpp"
#include "RAJA/util/camp_aliases.hpp"

#if defined(RAJA_ENABLE_CUDA)
#include "RAJA/policy/cuda/MemUtils_CUDA.hpp"
#include "RAJA/policy/cuda/raja_cudaerrchk.hpp"
#endif

#if defined(RAJA_ENABLE_HIP)
#include "RAJA/policy/hip/MemUtils_HIP.hpp"
#include "RAJA/policy/hip/raja_hiperrchk.hpp"
#endif


namespace RAJA
{

namespace detail
{

template < typename Resource >
struct ResourceInfo;

#if defined(RAJA_ENABLE_CUDA)

template < >
struct ResourceInfo<camp::resources::Cuda>
{
  static Allocator& get_pinned_allocator() {
    return cuda::get_pinned_allocator();
  }
  static Allocator& get_device_allocator() {
    return cuda::get_device_allocator();
  }
  static Allocator& get_device_zeroed_allocator() {
    return cuda::get_device_zeroed_allocator();
  }
  using identifier = cudaStream_t;
  static cudaStream_t get_identifier(camp::resources::Cuda& r)
  {
    return r.get_stream();
  }
  static void synchronize(cudaStream_t s)
  {
    cuda::synchronize(s);
  }
  static bool get_tl_launch_info(size_t& num_teams, cudaStream_t& id)
  {
    RAJA::cuda::detail::LaunchInfo* tl_launch_info = cuda::get_tl_launch_info();
    if (tl_launch_info != nullptr) {
      num_teams = tl_launch_info->gridDim.x *
                  tl_launch_info->gridDim.y *
                  tl_launch_info->gridDim.z ;
      id = tl_launch_info->stream;
      return true;
    } else {
      return false;
    }
  }
};

#endif


#if defined(RAJA_ENABLE_HIP)

template < >
struct ResourceInfo<camp::resources::Hip>
{
  static Allocator& get_pinned_allocator() {
    return hip::get_pinned_allocator();
  }
  static Allocator& get_device_allocator() {
    return hip::get_device_allocator();
  }
  static Allocator& get_device_zeroed_allocator() {
    return hip::get_device_zeroed_allocator();
  }
  using identifier = hipStream_t;
  static hipStream_t get_identifier(camp::resources::Hip& r)
  {
    return r.get_stream();
  }
  static void synchronize(hipStream_t s)
  {
    hip::synchronize(s);
  }
  static bool get_tl_launch_info(size_t& num_teams, hipStream_t& id)
  {
    RAJA::hip::detail::LaunchInfo* tl_launch_info = hip::get_tl_launch_info();
    if (tl_launch_info != nullptr) {
      num_teams = tl_launch_info->gridDim.x *
                  tl_launch_info->gridDim.y *
                  tl_launch_info->gridDim.z ;
      id = tl_launch_info->stream;
      return true;
    } else {
      return false;
    }
  }
};

#endif


//! Object that manages pinned memory buffers for reduction results
//  use one per reducer object
template <typename Resource>
class GPUReducerTally
{
  using resource_info = ResourceInfo<Resource>;
  using identifier = typename resource_info::identifier;
public:
  struct DevicePointers {
    bool in_use;
    void* device_ptr;
    void* device_zeroed_ptr;
  };

  using value_list_type = std::list<void*>;
  using value_list_iterator = typename value_list_type::iterator;

  using size_to_memory_type = std::multimap<size_t, DevicePointers>;
  using size_to_memory_iterator = typename size_to_memory_type::iterator;

  //! Object per id to keep track of pinned memory nodes
  struct PointerLists {
    value_list_type value_list;
    size_to_memory_type size_to_memory_map;
  };

  using id_to_lists_type = std::map<identifier, PointerLists>;
  using id_to_lists_iterator = typename id_to_lists_type::iterator;

  //! Iterator over streams used by reducer
  class StreamIterator
  {
  public:
    StreamIterator() = delete;

    StreamIterator(id_to_lists_iterator id_i) : m_id_i(id_i) {}

    const StreamIterator& operator++()
    {
      ++m_id_i;
      return *this;
    }

    StreamIterator operator++(int)
    {
      StreamIterator ret = *this;
      this->operator++();
      return ret;
    }

    identifier const& operator*() { return m_id_i->first; }

    bool operator==(const StreamIterator& rhs) const
    {
      return m_id_i == rhs.m_id_i;
    }

    bool operator!=(const StreamIterator& rhs) const
    {
      return !this->operator==(rhs);
    }

  private:
    id_to_lists_iterator m_id_i;
  };

  //! Iterator over all values generated by reducer
  template < typename T >
  class PointerListsIterator
  {
  public:
    PointerListsIterator() = delete;

    PointerListsIterator(id_to_lists_iterator id_i, value_list_iterator vl_i)
      : m_id_i(id_i)
      , m_vl_i(vl_i)
    { }

    const PointerListsIterator& operator++()
    {
      if (m_vl_i != m_id_i->second.value_list.end()) {
        ++m_vl_i;
      } else {
        ++m_id_i;
        m_vl_i = m_id_i->second.value_list.begin();
      }
      return *this;
    }

    PointerListsIterator operator++(int)
    {
      PointerListsIterator ret = *this;
      this->operator++();
      return ret;
    }

    T& operator*() { return *static_cast<T*>(*m_vl_i); }

    bool operator==(const PointerListsIterator& rhs) const
    {
      return m_id_i == rhs.m_id_i && m_vl_i == rhs.m_vl_i;
    }

    bool operator!=(const PointerListsIterator& rhs) const
    {
      return !this->operator==(rhs);
    }

  private:
    id_to_lists_iterator m_id_i;
    value_list_iterator m_vl_i;
  };

  GPUReducerTally()
  {
#if defined(RAJA_ENABLE_OPENMP)
    create_mutex();
#endif
  }

  GPUReducerTally(const GPUReducerTally&) = delete;

  ~GPUReducerTally()
  {
    free_list();
#if defined(RAJA_ENABLE_OPENMP)
    destroy_mutex();
#endif
  }

  //! get begin iterator over streams
  StreamIterator streamBegin() { return {m_id_to_lists.begin()}; }

  //! get end iterator over streams
  StreamIterator streamEnd() { return {m_id_to_lists.end()}; }

  //! get begin iterator over values
  template < typename T >
  PointerListsIterator<T> begin()
  {
    id_to_lists_iterator id_i = m_id_to_lists.begin();
    value_list_iterator  vl_i{};
    if (id_i != m_id_to_lists.end()) {
      vl_i = id_i->second.value_list.begin();
    }
    return {id_i, vl_i};
  }

  //! get end iterator over values
  template < typename T >
  PointerListsIterator<T> end()
  {
    id_to_lists_iterator id_i = m_id_to_lists.end();
    value_list_iterator  vl_i{};
    if (id_i != m_id_to_lists.begin()) {
      --id_i;
      vl_i = id_i->second.value_list.end();
    }
    return {id_i, vl_i};
  }

  //! get new value and pointers based on arguments
  template < typename T >
  DevicePointers* new_value(size_t num_teams,
                            identifier id,
                            T*& value_ptr,
                            T*& device_atomic_ptr,
                            unsigned int*& device_count_ptr)
  {
    void* pinned_memory = nullptr;
    void* device_memory = nullptr;
    void* device_zeroed_memory = nullptr;
    const size_t pinned_memory_size = sizeof(T);
    const size_t device_memory_size = sizeof(T);
    const size_t device_zeroed_memory_size = sizeof(unsigned int);

    DevicePointers* n = new_value_impl(id,
                                       pinned_memory,        pinned_memory_size,
                                       device_memory,        device_memory_size,
                                       device_zeroed_memory, device_zeroed_memory_size);

    value_ptr         = static_cast<T*>(pinned_memory);
    device_atomic_ptr = static_cast<T*>(device_memory);
    device_count_ptr  = static_cast<unsigned int*>(device_zeroed_memory);

    return n;
  }

  //! get new value and pointers based on thread local launch info
  template < typename T >
  DevicePointers* new_value_tl(T*& value_ptr,
                               T*& device_atomic_ptr,
                               unsigned int*& device_count_ptr)
  {
    size_t num_teams;
    identifier id;
    if (resource_info::get_tl_launch_info(num_teams, id)) {

      return new_value(num_teams,
                       id,
                       value_ptr,
                       device_atomic_ptr,
                       device_count_ptr);
    } else {
      return nullptr;
    }
  }

  //! get new value and pointers based on arguments
  template < typename T >
  DevicePointers* new_value(size_t num_teams,
                            identifier id,
                            T*& value_ptr,
                            SoAPtr<T>& device_soa_ptr,
                            unsigned int*& device_count_ptr)
  {
    void* pinned_memory = nullptr;
    void* device_memory = nullptr;
    void* device_zeroed_memory = nullptr;
    const size_t pinned_memory_size = sizeof(T);
    const size_t device_memory_size = device_soa_ptr.allocationSize(num_teams);
    const size_t device_zeroed_memory_size = sizeof(unsigned int);

    DevicePointers* n = new_value_impl(id,
                                       pinned_memory,        pinned_memory_size,
                                       device_memory,        device_memory_size,
                                       device_zeroed_memory, device_zeroed_memory_size);

    value_ptr         = static_cast<T*>(pinned_memory);
    device_soa_ptr.setMemory(num_teams, device_memory);
    device_count_ptr  = static_cast<unsigned int*>(device_zeroed_memory);

    return n;
  }

  //! get new value and pointers based on thread local launch info
  template < typename T >
  DevicePointers* new_value_tl(T*& value_ptr,
                               SoAPtr<T>& device_soa_ptr,
                               unsigned int*& device_count_ptr)
  {
    size_t num_teams;
    identifier id;
    if (resource_info::get_tl_launch_info(num_teams, id)) {

      return new_value(num_teams,
                       id,
                       value_ptr,
                       device_soa_ptr,
                       device_count_ptr);
    } else {
      return nullptr;
    }
  }

  void reuse_memory(DevicePointers* mn)
  {
    mn->in_use = false;
  }

  //! synchronize all streams used
  void synchronize_streams()
  {
    auto end = streamEnd();
    for (auto s = streamBegin(); s != end; ++s) {
      resource_info::synchronize(*s);
    }
  }

  //! all values used in all streams
  void free_list()
  {
    for (auto& id_and_lists : m_id_to_lists) {
      PointerLists& pl = id_and_lists.second;
      for (void* value_ptr : pl.value_list) {
        resource_info::get_pinned_allocator().deallocate(value_ptr);
      }
      pl.value_list.clear();
      for (auto& size_and_memory : pl.size_to_memory_map) {
        resource_info::get_device_allocator().deallocate(
            size_and_memory.second.device_ptr);
        resource_info::get_device_zeroed_allocator().deallocate(
            size_and_memory.second.device_zeroed_ptr);
        if (size_and_memory.second.in_use) {
          RAJA_ABORT_OR_THROW(
              "GPUReducerTally: Can't free DevicePointers that is in use");
        }
      }
      pl.size_to_memory_map.clear();
    }
    m_id_to_lists.clear();
  }

#if defined(RAJA_ENABLE_OPENMP)
  void lock_and_call(void(*func)(void*), void* data);
#endif

private:
  id_to_lists_type m_id_to_lists;

#if defined(RAJA_ENABLE_OPENMP)
  static const size_t mutex_storage_size = (16u > sizeof(void*)) ? 16u : sizeof(void*);
  using mutex_storage_type = typename std::aligned_storage<mutex_storage_size, alignof(std::max_align_t)>::type;

  mutex_storage_type m_mutex_storage;

  void create_mutex();
  void destroy_mutex();
#endif

  DevicePointers* new_value_impl(
      identifier id,
      void*& pinned_ptr,        size_t pinned_memory_size,
      void*& device_ptr,        size_t device_memory_size,
      void*& device_zeroed_ptr, size_t device_zeroed_memory_size);
};

}  // end namespace detail

}  // namespace RAJA

#endif  // closing endif for header file include guard
