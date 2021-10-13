/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   RAJA header file containing an implementation of a memory pool.
 *
 ******************************************************************************
 */

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-21, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef RAJA_UTIL_ALLOCATORPOOL_HPP
#define RAJA_UTIL_ALLOCATORPOOL_HPP

#include "RAJA/config.hpp"

#include <cassert>
#include <cstddef>
#include <cstdlib>
#include <list>
#include <map>
#include <string>

#include "RAJA/util/align.hpp"
#include "RAJA/util/mutex.hpp"
#include "RAJA/util/Allocator.hpp"

// Note that this header is not included in RAJA.hpp
// to avoid this warning when using openmp enabled headers with
// non-openmp compilers
#if defined(RAJA_ENABLE_OPENMP) && !defined(_OPENMP)
#error RAJA configured with ENABLE_OPENMP, but OpenMP not supported by current compiler
#endif

namespace RAJA
{

namespace detail
{

//! example allocator for AllocatorPool using allocate/deallocate
struct HostBaseAllocator {

  // returns a valid pointer on success, nullptr on failure
  void* allocate(std::size_t nbytes)
  {
    return std::malloc(nbytes);
  }

  // returns true on success, false on failure
  void deallocate(void* ptr)
  {
    std::free(ptr);
  }
};

//! Class abstracting a range in memory
struct MemoryChunk {
  void* begin = nullptr;
  void* end   = nullptr;

  MemoryChunk() = default;

  MemoryChunk(void* ptr, size_t nbytes)
    : begin(ptr)
    , end(static_cast<char*>(begin) + nbytes)
  { }

  MemoryChunk(void* begin_, void* end_)
    : begin(begin_)
    , end(end_)
  { }

  size_t nbytes() const
  {
    return static_cast<char*>(end) -
           static_cast<char*>(begin);
  }

  explicit operator bool() const
  {
    return begin != nullptr;
  }
};

/*! \class MemoryArena
 ******************************************************************************
 *
 * \brief  MemoryArena is a map based subclass for class Allocator
 * provides book-keeping to divy a large chunk of pre-allocated memory to avoid
 * the overhead of  malloc/free or cudaMalloc/cudaFree, etc
 *
 * get/give are the primary calls used by class Allocator to get aligned memory
 * from the pool or give it back
 *
 *
 ******************************************************************************
 */
struct MemoryArena
{
  using free_type = std::map<void*, void*>;
  using free_value_type = typename free_type::value_type;
  using used_type = std::map<void*, void*>;
  using used_value_type = typename used_type::value_type;


  MemoryArena(MemoryChunk mem)
    : m_allocation(mem)
  {
    if (m_allocation.begin == nullptr) {
      RAJA_ABORT_OR_THROW("RAJA::detail::MemoryArena Attempt to create with no memory");
    }
    m_free_space[m_allocation.begin] = m_allocation.end ;
  }

  MemoryArena(MemoryArena const&) = delete;
  MemoryArena& operator=(MemoryArena const&) = delete;

  MemoryArena(MemoryArena&&) = default;
  MemoryArena& operator=(MemoryArena&&) = default;

  size_t capacity() const
  {
    return m_allocation.nbytes();
  }

  bool unused() const
  {
    return m_used_space.empty();
  }

  MemoryChunk get_allocation()
  {
    return m_allocation;
  }

  MemoryChunk get(size_t nbytes, size_t alignment)
  {
    MemoryChunk mem;
    if (capacity() >= nbytes) {
      free_type::iterator end = m_free_space.end();
      for (free_type::iterator iter = m_free_space.begin(); iter != end;
           ++iter) {

        void* adj_ptr = iter->first;
        size_t cap =
            static_cast<char*>(iter->second) - static_cast<char*>(adj_ptr);

        if (::RAJA::align(alignment, nbytes, adj_ptr, cap)) {

          mem = MemoryChunk(adj_ptr, nbytes);

          remove_free_chunk(iter,
                            mem);

          add_used_chunk(mem);

          break;
        }
      }
    }
    return mem;
  }

  MemoryChunk give(void* ptr)
  {
    MemoryChunk mem;
    if (m_allocation.begin <= ptr && ptr < m_allocation.end) {

      used_type::iterator found = m_used_space.find(ptr);

      if (found != m_used_space.end()) {

        mem = MemoryChunk(found->first, found->second);

        add_free_chunk(mem);

        m_used_space.erase(found);

      } else {
        RAJA_ABORT_OR_THROW("RAJA::detail::MemoryArena::give invalid ptr");
      }
    }
    return mem;
  }

private:
  void add_free_chunk(MemoryChunk mem)
  {
    // integrates a chunk of memory into free_space
    free_type::iterator invl = m_free_space.end();
    free_type::iterator next = m_free_space.lower_bound(mem.begin);

    // check if prev exists
    if (next != m_free_space.begin()) {
      // check if prev can cover [begin, end)
      free_type::iterator prev = next;
      --prev;
      if (prev->second == mem.begin) {
        // extend prev to cover [begin, end)
        prev->second = mem.end;

        // check if prev can cover next too
        if (next != invl) {
          assert(next->first != mem.begin);

          if (next->first == mem.end) {
            // extend prev to cover next too
            prev->second = next->second;

            // remove redundant next
            m_free_space.erase(next);
          }
        }
        return;
      }
    }

    if (next != invl) {
      assert(next->first != mem.begin);

      if (next->first == mem.end) {
        // extend next to cover [begin, end)
        m_free_space.insert(next, free_value_type{mem.begin, next->second});
        m_free_space.erase(next);

        return;
      }
    }

    // no free space adjacent to this chunk, add separate free chunk [begin,
    // end)
    m_free_space.insert(next, free_value_type{mem.begin, mem.end});
  }

  void remove_free_chunk(free_type::iterator iter, MemoryChunk mem)
  {
    void* ptr = iter->first;
    void* ptr_end = iter->second;

    // fixup m_free_space, shrinking and adding chunks as needed
    if (ptr != mem.begin) {

      // shrink end of current free region to [ptr, begin)
      iter->second = mem.begin;

      if (mem.end != ptr_end) {

        // insert free region [end, ptr_end) after current free region
        free_type::iterator next = iter;
        ++next;
        m_free_space.insert(next, free_value_type{mem.end, ptr_end});
      }

    } else if (mem.end != ptr_end) {

      // shrink beginning of current free region to [end, ptr_end)
      free_type::iterator next = iter;
      ++next;
      m_free_space.insert(next, free_value_type{mem.end, ptr_end});
      m_free_space.erase(iter);

    } else {

      // can not reuse current region, erase
      m_free_space.erase(iter);
    }
  }

  void add_used_chunk(MemoryChunk mem)
  {
    // simply inserts a chunk of memory into used_space
    m_used_space.insert(used_value_type{mem.begin, mem.end});
  }

  MemoryChunk m_allocation;
  free_type m_free_space;
  used_type m_used_space;
};

} /* end namespace detail */


/*! \class AllocatorPool
 ******************************************************************************
 *
 * \brief  AllocatorPool provides a a RAJA::Allocator that is a basic memory
 *  pool on top of the given allocator_type.
 *
 * This is used for RAJA's internal allocations by default, but a different
 * Allocator can be set for specific backend allocators. For example
 * RAJA::cuda::set_device_allocator allows the user to change the device
 * allocator used by RAJA internally.
 *
 * AllocatorPool uses MemoryArena to do the heavy lifting of maintaining
 * access to the used/free space.
 *
 * The following are some examples
 *
 * //! example allocator for AllocatorPool using allocate/deallocate
 * struct host_allocator {
 *
 *   // returns a valid pointer on success, nullptr on failure
 *   void* allocate(std::size_t nbytes)
 *   {
 *     return std::malloc(nbytes);
 *   }
 *
 *   // returns true on success, false on failure
 *   void deallocate(void* ptr)
 *   {
 *     std::free(ptr);
 *   }
 * };
 *
 * RAJA::Allocator* aloc =
 *     new RAJA::AllocatorPool<host_allocator>();
 *
 ******************************************************************************
 */
template <typename allocator_t>
struct AllocatorPool : Allocator
{
  using allocator_type = allocator_t;

  static const size_t default_default_arena_size = 32ull * 1024ull * 1024ull;

  AllocatorPool(std::string const& name,
                allocator_type const& aloc = allocator_type{},
                size_t default_arena_size = default_default_arena_size)
    : m_default_arena_size(default_arena_size)
    , m_alloc(aloc)
    , m_name(name) // std::string("RAJA::AllocatorPool<")+m_alloc.getName()+">")
  {
  }

  // not copyable or movable
  AllocatorPool(AllocatorPool const&) = delete;
  AllocatorPool(AllocatorPool &&) = delete;
  AllocatorPool& operator=(AllocatorPool const&) = delete;
  AllocatorPool& operator=(AllocatorPool &&) = delete;

  virtual ~AllocatorPool()
  {
    // When used with static storage duration, it is possible to encounter
    // errors like cudaErrorCudartUnloading with cudaFree. So do not call
    // release here to avoid potential cuda calls and errors.
  }

  void* allocate(size_t nbytes,
                 size_t alignment = alignof(std::max_align_t)) final
  {
    if (nbytes == 0) return nullptr;

#if defined(RAJA_ENABLE_OPENMP)
    lock_guard<omp::mutex> lock(m_mutex);
#endif

    detail::MemoryChunk mem;

    // find a usable memory chunk in an existing arena
    arena_container_type::iterator end = m_arenas.end();
    for (arena_container_type::iterator iter = m_arenas.begin(); iter != end;
         ++iter) {
      mem = iter->get(nbytes, alignment);
      if (mem.begin != nullptr) {
        break;
      }
    }

    // allocate a new memory chunk
    if (mem.begin == nullptr) {
      const size_t alloc_size =
          std::max(nbytes + alignment, m_default_arena_size);
      detail::MemoryChunk arena_mem(m_alloc.allocate(alloc_size), alloc_size);
      if (arena_mem.begin != nullptr) {
        m_arenas.emplace_front(arena_mem);
        m_actualSize += m_arenas.front().capacity();
        mem = m_arenas.front().get(nbytes, alignment);
      } else{
        RAJA_ABORT_OR_THROW("RAJA::AllocatorPool::allocate arena allocation failed");
      }
    }

    // update stats
    m_currentSize += mem.nbytes();
    if (m_currentSize > m_highWatermark) {
      m_highWatermark = m_currentSize;
    }
    m_allocationCount += 1;

    return mem.begin;
  }

  void deallocate(void* ptr) final
  {
#if defined(RAJA_ENABLE_OPENMP)
    lock_guard<omp::mutex> lock(m_mutex);
#endif

    // find arena that owns ptr and return it
    detail::MemoryChunk mem;
    arena_container_type::iterator end = m_arenas.end();
    for (arena_container_type::iterator iter = m_arenas.begin(); iter != end;
         ++iter) {
      if ( (mem = iter->give(ptr)) ) {
        ptr = nullptr;
        // update stats
        m_currentSize -= mem.nbytes();
        m_allocationCount -= 1;
        break;
      }
    }

    if (ptr != nullptr) {
      RAJA_ABORT_OR_THROW("RAJA::AllocatorPool::deallocate unknown pointer");
    }
  }

  void release() final
  {
#if defined(RAJA_ENABLE_OPENMP)
    lock_guard<omp::mutex> lock(m_mutex);
#endif

    for (auto it = m_arenas.begin(); it != m_arenas.end(); /* do nothing */) {
      if (it->unused()) {
        // update stats
        m_actualSize -= it->capacity();
        // deallocate memory
        detail::MemoryChunk mem = it->get_allocation();
        m_alloc.deallocate(mem.begin);
        // erase
        it = m_arenas.erase(it);
      } else {
        ++it;
      }
    }
  }

  size_t get_arena_size()
  {
#if defined(RAJA_ENABLE_OPENMP)
    lock_guard<omp::mutex> lock(m_mutex);
#endif

    return m_default_arena_size;
  }

  size_t set_arena_size(size_t new_arena_size)
  {
#if defined(RAJA_ENABLE_OPENMP)
    lock_guard<omp::mutex> lock(m_mutex);
#endif

    size_t prev_size = m_default_arena_size;
    m_default_arena_size = new_arena_size;
    return prev_size;
  }

  size_t getHighWatermark() const noexcept final
  {
    return m_highWatermark;
  }

  size_t getCurrentSize() const noexcept final
  {
    return m_currentSize;
  }

  size_t getActualSize() const noexcept final
  {
    return m_actualSize;
  }

  size_t getAllocationCount() const noexcept final
  {
    return m_allocationCount;
  }

  const std::string& getName() const noexcept final
  {
    return m_name;
  }

  // Platform getPlatform() const noexcept final
  // {
  //   return m_alloc.getPlatform();
  // }

private:
  using arena_container_type = std::list<detail::MemoryArena>;

  arena_container_type m_arenas;
  size_t m_default_arena_size;
  allocator_t m_alloc;
  std::string m_name;

  size_t m_highWatermark = 0;
  size_t m_currentSize = 0;
  size_t m_actualSize = 0;
  size_t m_allocationCount = 0;

#if defined(RAJA_ENABLE_OPENMP)
  omp::mutex m_mutex;
#endif
};

} /* end namespace RAJA */


#endif /* RAJA_UTIL_ALLOCATORPOOL_HPP */
