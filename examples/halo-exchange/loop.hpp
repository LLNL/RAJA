//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-21, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef RAJA_EXAMPLES_HALOEXCHANGE_LOOP_HPP
#define RAJA_EXAMPLES_HALOEXCHANGE_LOOP_HPP

#include <cstddef>

#include "RAJA/RAJA.hpp"
#include "../memoryManager.hpp"


/*
  Policies
*/
using loop_raja_forall_seq_policy = RAJA::loop_exec;
#if defined(RAJA_ENABLE_OPENMP)
using loop_raja_forall_omp_policy = RAJA::omp_parallel_for_exec;
#endif
#if defined(RAJA_ENABLE_CUDA)
const int CUDA_BLOCK_SIZE = 256;
const int CUDA_WORKGROUP_BLOCK_SIZE = 1024;
using loop_raja_forall_cuda_policy = RAJA::cuda_exec_async<CUDA_BLOCK_SIZE>;
#endif
#if defined(RAJA_ENABLE_HIP)
const int HIP_BLOCK_SIZE = 256;
const int HIP_WORKGROUP_BLOCK_SIZE = 1024;
using loop_raja_forall_hip_policy = RAJA::hip_exec_async<HIP_BLOCK_SIZE>;
#endif


enum struct LoopPattern : int
{
  invalid = 0
 ,seq
 ,raja_seq
 ,raja_fused_seq
#if defined(RAJA_ENABLE_OPENMP)
 ,raja_omp
 ,raja_fused_omp
#endif
#if defined(RAJA_ENABLE_CUDA)
 ,raja_cuda
 ,raja_fused_cuda
#endif
#if defined(RAJA_ENABLE_HIP)
 ,raja_hip
 ,raja_fused_hip
#endif
 ,End
};


namespace detail
{

inline LoopPattern& getLoopPattern()
{
  static LoopPattern pattern = LoopPattern::invalid;
  return pattern;
}

inline RAJA::resources::Host& getHostResource()
{
  static RAJA::resources::Host res = RAJA::resources::Host::get_default();
  return res;
}

#if defined(RAJA_ENABLE_CUDA)
inline RAJA::resources::Cuda& getCudaResource()
{
  static RAJA::resources::Cuda res = RAJA::resources::Cuda::get_default();
  return res;
}
#endif

#if defined(RAJA_ENABLE_HIP)
inline RAJA::resources::Hip& getHipResource()
{
  static RAJA::resources::Hip res = RAJA::resources::Hip::get_default();
  return res;
}
#endif


template < typename T >
struct memory_manager_allocator
{
  using value_type = T;

  memory_manager_allocator() = default;

  template < typename U >
  constexpr memory_manager_allocator(memory_manager_allocator<U> const&) noexcept
  { }

  /*[[nodiscard]]*/
  value_type* allocate(size_t num)
  {
    if (num > std::numeric_limits<size_t>::max() / sizeof(value_type)) {
      throw std::bad_alloc();
    }

    value_type *ptr = memoryManager::allocate<value_type>(num);

    if (!ptr) {
      throw std::bad_alloc();
    }

    return ptr;
  }

  void deallocate(value_type* ptr, size_t) noexcept
  {
    value_type* ptrc = static_cast<value_type*>(ptr);
    memoryManager::deallocate(ptrc);
  }
};

template <typename T, typename U>
bool operator==(memory_manager_allocator<T> const&, memory_manager_allocator<U> const&)
{
  return true;
}

template <typename T, typename U>
bool operator!=(memory_manager_allocator<T> const& lhs, memory_manager_allocator<U> const& rhs)
{
  return !(lhs == rhs);
}

#if defined(RAJA_ENABLE_CUDA)

template < typename T >
struct cuda_pinned_allocator
{
  using value_type = T;

  cuda_pinned_allocator() = default;

  template < typename U >
  constexpr cuda_pinned_allocator(cuda_pinned_allocator<U> const&) noexcept
  { }

  /*[[nodiscard]]*/
  value_type* allocate(size_t num)
  {
    if (num > std::numeric_limits<size_t>::max() / sizeof(value_type)) {
      throw std::bad_alloc();
    }

    value_type *ptr =
        RAJA::cuda::pinned_mempool_type::getInstance().template malloc<value_type>(num);

    if (!ptr) {
      throw std::bad_alloc();
    }

    return ptr;
  }

  void deallocate(value_type* ptr, size_t) noexcept
  {
    RAJA::cuda::pinned_mempool_type::getInstance().free(ptr);
  }
};

template <typename T, typename U>
bool operator==(cuda_pinned_allocator<T> const&, cuda_pinned_allocator<U> const&)
{
  return true;
}

template <typename T, typename U>
bool operator!=(cuda_pinned_allocator<T> const& lhs, cuda_pinned_allocator<U> const& rhs)
{
  return !(lhs == rhs);
}

#endif

#if defined(RAJA_ENABLE_HIP)

template < typename T >
struct hip_pinned_allocator
{
  using value_type = T;

  hip_pinned_allocator() = default;

  template < typename U >
  constexpr hip_pinned_allocator(hip_pinned_allocator<U> const&) noexcept
  { }

  /*[[nodiscard]]*/
  value_type* allocate(size_t num)
  {
    if (num > std::numeric_limits<size_t>::max() / sizeof(value_type)) {
      throw std::bad_alloc();
    }

    value_type *ptr =
        RAJA::hip::pinned_mempool_type::getInstance().template malloc<value_type>(num);

    if (!ptr) {
      throw std::bad_alloc();
    }

    return ptr;
  }

  void deallocate(value_type* ptr, size_t) noexcept
  {
    RAJA::hip::pinned_mempool_type::getInstance().free(ptr);
  }
};

template <typename T, typename U>
bool operator==(hip_pinned_allocator<T> const&, hip_pinned_allocator<U> const&)
{
  return true;
}

template <typename T, typename U>
bool operator!=(hip_pinned_allocator<T> const& lhs, hip_pinned_allocator<U> const& rhs)
{
  return !(lhs == rhs);
}

#endif

inline memory_manager_allocator<char>& getHostBufferAllocator()
{
  static memory_manager_allocator<char> aloc;
  return aloc;
}

#if defined(RAJA_ENABLE_CUDA)
inline cuda_pinned_allocator<char>& getCudaBufferAllocator()
{
  static cuda_pinned_allocator<char> aloc;
  return aloc;
}
#endif

#if defined(RAJA_ENABLE_HIP)
inline hip_pinned_allocator<char>& getHipBufferAllocator()
{
  static hip_pinned_allocator<char> aloc;
  return aloc;
}
#endif

} // namespace detail


struct SetLoopPatternScope
{
  SetLoopPatternScope(LoopPattern new_pattern)
    : m_old_pattern(detail::getLoopPattern())
  {
    detail::getLoopPattern() = new_pattern;
  }

  ~SetLoopPatternScope()
  {
    detail::getLoopPattern() = m_old_pattern;
  }

private:
  LoopPattern m_old_pattern;
};

inline const char* get_loop_pattern_name()
{
  switch (detail::getLoopPattern()) {
    case LoopPattern::invalid:
    {
      return "invalid";
    } break;
    case LoopPattern::seq:
    {
      return "seq";
    } break;
    case LoopPattern::raja_seq:
    {
      return "raja_seq";
    } break;
    case LoopPattern::raja_fused_seq:
    {
      return "raja_fused_seq";
    } break;
#if defined(RAJA_ENABLE_OPENMP)
    case LoopPattern::raja_omp:
    {
      return "raja_omp";
    } break;
    case LoopPattern::raja_fused_omp:
    {
      return "raja_fused_omp";
    } break;
#endif
#if defined(RAJA_ENABLE_CUDA)
    case LoopPattern::raja_cuda:
    {
      return "raja_cuda";
    } break;
    case LoopPattern::raja_fused_cuda:
    {
      return "raja_fused_cuda";
    } break;
#endif
#if defined(RAJA_ENABLE_HIP)
    case LoopPattern::raja_hip:
    {
      return "raja_hip";
    } break;
    case LoopPattern::raja_fused_hip:
    {
      return "raja_fused_hip";
    } break;
#endif
    default:
    {
      assert(0);
      return "error";
    } break;
  }
}

inline RAJA::resources::Resource get_loop_pattern_resource()
{
  switch (detail::getLoopPattern()) {
    case LoopPattern::seq:
    case LoopPattern::raja_seq:
    case LoopPattern::raja_fused_seq:
#if defined(RAJA_ENABLE_OPENMP)
    case LoopPattern::raja_omp:
    case LoopPattern::raja_fused_omp:
#endif
    {
      return detail::getHostResource();
    } break;
#if defined(RAJA_ENABLE_CUDA)
    case LoopPattern::raja_cuda:
    case LoopPattern::raja_fused_cuda:
    {
      return detail::getCudaResource();
    } break;
#endif
#if defined(RAJA_ENABLE_HIP)
    case LoopPattern::raja_hip:
    case LoopPattern::raja_fused_hip:
    {
      return detail::getHipResource();
    } break;
#endif
    default:
    {
      assert(0);
      return detail::getHostResource();
    } break;
  }
}

inline void loop_synchronize()
{
  switch (detail::getLoopPattern()) {
    case LoopPattern::seq:
    {
    } break;
    case LoopPattern::raja_seq:
    case LoopPattern::raja_fused_seq:
#if defined(RAJA_ENABLE_OPENMP)
    case LoopPattern::raja_omp:
    case LoopPattern::raja_fused_omp:
#endif
    {
      detail::getHostResource().wait();
    } break;
#if defined(RAJA_ENABLE_CUDA)
    case LoopPattern::raja_cuda:
    case LoopPattern::raja_fused_cuda:
    {
      detail::getCudaResource().wait();
    } break;
#endif
#if defined(RAJA_ENABLE_HIP)
    case LoopPattern::raja_hip:
    case LoopPattern::raja_fused_hip:
    {
      detail::getHipResource().wait();
    } break;
#endif
    default:
    {
      assert(0);
    } break;
  }
}

template < typename Index, typename LoopBody >
inline void loop(Index N, LoopBody&& body)
{
  switch (detail::getLoopPattern()) {
    case LoopPattern::seq:
    {
      for (Index i = static_cast<Index>(0); i < N; ++i) {
        body(i);
      }
    } break;
    case LoopPattern::raja_seq:
    case LoopPattern::raja_fused_seq:
    {
      RAJA::forall<loop_raja_forall_seq_policy>(
          detail::getHostResource(),
          RAJA::TypedRangeSegment<Index>(0, N),
          std::forward<LoopBody>(body));
    } break;
#if defined(RAJA_ENABLE_OPENMP)
    case LoopPattern::raja_omp:
    case LoopPattern::raja_fused_omp:
    {
      RAJA::forall<loop_raja_forall_omp_policy>(
          detail::getHostResource(),
          RAJA::TypedRangeSegment<Index>(0, N),
          std::forward<LoopBody>(body));
    } break;
#endif
#if defined(RAJA_ENABLE_CUDA)
    case LoopPattern::raja_cuda:
    case LoopPattern::raja_fused_cuda:
    {
      RAJA::forall<loop_raja_forall_cuda_policy>(
          detail::getCudaResource(),
          RAJA::TypedRangeSegment<Index>(0, N),
          std::forward<LoopBody>(body));
    } break;
#endif
#if defined(RAJA_ENABLE_HIP)
    case LoopPattern::raja_hip:
    case LoopPattern::raja_fused_hip:
    {
      RAJA::forall<loop_raja_forall_hip_policy>(
          detail::getHipResource(),
          RAJA::TypedRangeSegment<Index>(0, N),
          std::forward<LoopBody>(body));
    } break;
#endif
    default:
    {
      assert(0);
    } break;
  }
}

inline void* loop_allocate_buffer(size_t nbytes)
{
  switch (detail::getLoopPattern()) {
    case LoopPattern::seq:
    {
      return malloc(nbytes);
    } break;
    case LoopPattern::raja_seq:
    case LoopPattern::raja_fused_seq:
#if defined(RAJA_ENABLE_OPENMP)
    case LoopPattern::raja_omp:
    case LoopPattern::raja_fused_omp:
#endif
    {
      return detail::getHostBufferAllocator().allocate(nbytes);
    } break;
#if defined(RAJA_ENABLE_CUDA)
    case LoopPattern::raja_cuda:
    case LoopPattern::raja_fused_cuda:
    {
      return detail::getCudaBufferAllocator().allocate(nbytes);
    } break;
#endif
#if defined(RAJA_ENABLE_HIP)
    case LoopPattern::raja_hip:
    case LoopPattern::raja_fused_hip:
    {
      return detail::getHipBufferAllocator().allocate(nbytes);
    } break;
#endif
    default:
    {
      assert(0);
      return nullptr;
    } break;
  }
}

inline void loop_deallocate_buffer(void* ptr)
{
  switch (detail::getLoopPattern()) {
    case LoopPattern::seq:
    {
      free(ptr);
    } break;
    case LoopPattern::raja_seq:
    case LoopPattern::raja_fused_seq:
#if defined(RAJA_ENABLE_OPENMP)
    case LoopPattern::raja_omp:
    case LoopPattern::raja_fused_omp:
#endif
    {
      detail::getHostBufferAllocator().deallocate(static_cast<char*>(ptr), 0);
    } break;
#if defined(RAJA_ENABLE_CUDA)
    case LoopPattern::raja_cuda:
    case LoopPattern::raja_fused_cuda:
    {
      detail::getCudaBufferAllocator().deallocate(static_cast<char*>(ptr), 0);
    } break;
#endif
#if defined(RAJA_ENABLE_HIP)
    case LoopPattern::raja_hip:
    case LoopPattern::raja_fused_hip:
    {
      detail::getHipBufferAllocator().deallocate(static_cast<char*>(ptr), 0);
    } break;
#endif
    default:
    {
      assert(0);
    } break;
  }
}

#endif // RAJA_EXAMPLES_HALOEXCHANGE_LOOP_HPP
