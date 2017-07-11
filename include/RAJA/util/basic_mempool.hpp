/*
 * $Id$
 */

/*!
 *******************************************************************************
 * \file basic_mempool.hxx
 *
 * \date June 07, 2017
 * \author Jason Burmark (burmark1@llnl.gov)
 *******************************************************************************
 */

#ifndef BASIC_MEMPOOL_HXX_
#define BASIC_MEMPOOL_HXX_

//#include "Messages.h"
//#include "MemoryMacros.h"

#include <cassert>
#include <list>
#include <map>
#include <cstddef>

namespace RAJA {

namespace basic_mempool {

namespace detail {


class memory_arena {
public:

  // taken from libc++
  static void* align(size_t alignment, size_t size, void*& ptr, size_t& space) noexcept
  {
      void* r = nullptr;
      if (size <= space)
      {
          char* p1 = static_cast<char*>(ptr);
          char* p2 = reinterpret_cast<char*>(reinterpret_cast<size_t>(p1 + (alignment - 1)) & -alignment);
          size_t d = static_cast<size_t>(p2 - p1);
          if (d <= space - size)
          {
              r = p2;
              ptr = r;
              space -= d;
          }
      }
      return r;
  }

  using free_type = std::map<void*, void*>;
  using free_value_type = typename free_type::value_type;
  using used_type = std::map<void*, void*>;
  using used_value_type = typename used_type::value_type;

  memory_arena(void* ptr, size_t size)
    : m_allocation{ ptr, static_cast<char*>(ptr)+size },
      m_free_space({ free_value_type{ptr, static_cast<char*>(ptr)+size} }),
      m_used_space()
  {
    //const char me[] = "memory_arena::memory_arena";
    if (m_allocation.begin == nullptr) {
      //ctlerror_any(me, "Attempt to create memory_arena with no memory");
    }
  }

  memory_arena(memory_arena const&) = delete;
  memory_arena& operator=(memory_arena const&) = delete;

  memory_arena(memory_arena &&) = default;
  memory_arena& operator=(memory_arena &&) = default;

  ~memory_arena()
  {

  }

  size_t capacity()
  {
    return static_cast<char*>(m_allocation.end) - static_cast<char*>(m_allocation.begin);
  }

  bool unused ()
  {
    return m_used_space.empty();
  }

  void* get_allocation()
  {
    return m_allocation.begin;
  }

  void* get(size_t nbytes, size_t alignment)
  {
    void* ptr_out = nullptr;
    if (capacity() >= nbytes) {
      free_type::iterator end = m_free_space.end();
      for (free_type::iterator iter = m_free_space.begin(); iter != end; ++iter) {

        void* adj_ptr = iter->first;
        size_t cap = static_cast<char*>(iter->second) - static_cast<char*>(adj_ptr);

        if (align(alignment, nbytes, adj_ptr, cap)) {

          ptr_out = adj_ptr;

          remove_free_chunk(iter, adj_ptr, static_cast<char*>(adj_ptr) + nbytes);

          add_used_chunk(adj_ptr, static_cast<char*>(adj_ptr) + nbytes);
          
          break;
        }

      }
    }
    return ptr_out;
  }

  bool give(void* ptr)
  {
    //const char me[] = "memory_arena::give";
    if ( m_allocation.begin <= ptr && ptr < m_allocation.end ) {

      used_type::iterator found = m_used_space.find(ptr);

      if ( found != m_used_space.end() ) {

        add_free_chunk(found->first, found->second);

        m_used_space.erase(found);

      } else {
        //ctlerror_any(me, "Invalid free %p", ptr);
      }

      return true;
    } else {
      return false;
    }
  }

private:
  struct memory_chunk {
    void* begin;
    void* end;
  };

  void add_free_chunk(void* begin, void* end)
  {
    // integrates a chunk of memory into free_space
    free_type::iterator invl = m_free_space.end();
    free_type::iterator next = m_free_space.lower_bound(begin);

    // check if prev exists
    if (next != m_free_space.begin()) {
      // check if prev can cover [begin, end)
      free_type::iterator prev = next; --prev;
      if (prev->second == begin) {
        // extend prev to cover [begin, end)
        prev->second = end;

        // check if prev can cover next too
        if (next != invl) {
          //ARES_ASSERT_ERROR(next->first != begin);
          if (next->first == end) {
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
      //ARES_ASSERT_ERROR(next->first != begin);
      if (next->first == end) {
        // extend next to cover [begin, end)
        m_free_space.insert(next, free_value_type{begin, next->second});
        m_free_space.erase(next);

        return;
      }
    }

    // no free space adjacent to this chunk, add seperate free chunk [begin, end)
    m_free_space.insert(next, free_value_type{begin, end});
  }

  void remove_free_chunk(free_type::iterator iter, void* begin, void* end) {

    void* ptr = iter->first;
    void* ptr_end = iter->second;

    // fixup m_free_space, shrinking and adding chunks as needed
    if (ptr != begin) {

      // shrink end of current free region to [ptr, begin)
      iter->second = begin;

      if (end != ptr_end) {

        // insert free region [end, ptr_end) after current free region
        free_type::iterator next = iter; ++next;
        m_free_space.insert(next, free_value_type{end, ptr_end});
      }

    } else if (end != ptr_end) {

      // shrink beginning of current free region to [end, ptr_end)
      free_type::iterator next = iter; ++next;
      m_free_space.insert(next, free_value_type{end, ptr_end});
      m_free_space.erase(iter);

    } else {

      // can not reuse current region, erase
      m_free_space.erase(iter);
    }
  }

  void add_used_chunk(void* begin, void* end)
  {
    // simply inserts a chunk of memory into used_space
    m_used_space.insert(used_value_type{begin, end});
  }

  memory_chunk m_allocation;
  free_type m_free_space;
  used_type m_used_space;
};

} /* end namespace detail */



template < typename allocator_t >
class mempool {
public:
  using allocator_type = allocator_t;

  static inline mempool<allocator_t>& getInstance()
  {
    static mempool<allocator_t> pool{};
    return pool;
  }

  static const size_t default_default_arena_size = 32ull*1024ull*1024ull;

  mempool()
    : m_arenas(),
      m_default_arena_size(default_default_arena_size),
      m_alloc()
  {

  }

  size_t default_arena_size()
  {
    return m_default_arena_size;
  }

  size_t default_arena_size(size_t new_size)
  {
    size_t prev_size = m_default_arena_size;
    m_default_arena_size = new_size;
    return prev_size;
  }

  template <typename T>
  T* malloc(size_t nTs, size_t alignment = alignof(T))
  {
    const size_t size = nTs*sizeof(T);
    void* ptr = nullptr;
    arena_container_type::iterator end = m_arenas.end();
    for (arena_container_type::iterator iter = m_arenas.begin(); iter != end; ++iter ) {
      ptr = iter->get(size, alignment);
      if (ptr != nullptr) {
        break;
      }
    }

    if (ptr == nullptr) {
      const size_t alloc_size = std::max(size+alignment, m_default_arena_size);
      void* arena_ptr = m_alloc.malloc(alloc_size);
      if (arena_ptr != nullptr) {
        m_arenas.emplace_front(arena_ptr, alloc_size);
        ptr = m_arenas.front().get(size, alignment);
      }
    }

    return static_cast<T*>(ptr);
  }

  void free(const void* cptr)
  {
    //const char me[] = "mempool::free";
    void* ptr = const_cast<void*>(cptr);
    arena_container_type::iterator end = m_arenas.end();
    for (arena_container_type::iterator iter = m_arenas.begin(); iter != end; ++iter ) {
      if (iter->give(ptr)) {
        ptr = nullptr;
        break;
      }
    }
    if (ptr != nullptr) {
      //ctlerror_any(me, "Unknown pointer %p", ptr);
    }
  }

private:
  using arena_container_type = std::list<detail::memory_arena>;

  arena_container_type m_arenas;
  size_t m_default_arena_size;
  allocator_t m_alloc;
};


#if 0
struct generic_allocator {

  // returns a valid pointer on success, nullptr on failure
  void* malloc(size_t nbytes)
  {
    return MALLOT(char, nbytes);
  }

  // returns true on success, false on failure
  bool free(void* ptr)
  {
    FREEMEM(ptr);
    return true;
  }

};

#endif

//#ifdef USE_CUDA
struct cuda_pinned_allocator {

  // returns a valid pointer on success, nullptr on failure
  void* malloc(size_t nbytes)
  {
    void* ptr;
    cudaHostAlloc(&ptr, nbytes, cudaHostAllocMapped);
    return ptr;
  }

  // returns true on success, false on failure
  bool free(void* ptr)
  {
    cudaFreeHost(ptr);
    return true;
  }

};


struct cuda_allocator {

  // returns a valid pointer on success, nullptr on failure
  void* malloc(size_t nbytes)
  {
    void* ptr;
    cudaMalloc(&ptr, nbytes);
    return ptr;
  }

  // returns true on success, false on failure
  bool free(void* ptr)
  {
    cudaFree(ptr);
    return true;
  }

};


//#endif

} /* end namespace basic_mempool */

} /* end namespace RAJA */



#endif /* BASIC_MEMPOOL_HXX_ */
