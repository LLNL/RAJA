#ifndef _STORAGE_HPP_
#define _STORAGE_HPP_

#include <cstdio>

#include "RAJA/RAJA.hpp"
#include "RAJA/internal/type_traits.hpp"
#include "RAJA/util/defines.hpp"

namespace internal
{

template <typename ExecPolicy, typename T, bool gpu>
struct storage {
  using type = T;

  static T* alloc(int n)
  {
    return RAJA::allocate_aligned_type<T>(64, n * sizeof(T));
  }

  static void free(T* ptr) { ::free(ptr); }

  static void ready() {}
};

#ifdef RAJA_ENABLE_CUDA

template <typename Exec, typename T>
struct storage<Exec, T, true> {
  using type = T;

  static T* alloc(int n)
  {
    T* ptr;
    ::cudaMallocManaged(&ptr, n * sizeof(T));
    return ptr;
  }

  static void free(T* ptr) { ::cudaFree(ptr); }

  static void ready() { ::cudaDeviceSynchronize(); }
};

#endif
}

struct storage_base {
};

template <typename ExecPolicy, typename T, bool inplace>
struct storage : public storage_base {
};

template <typename ExecPolicy, typename T>
struct storage<ExecPolicy, T, true> : public storage_base {
  using type = T;

#ifdef RAJA_ENABLE_CUDA
  static constexpr bool UseGPU = RAJA::is_cuda_policy<ExecPolicy>::value;
  using StorageType = typename internal::storage<ExecPolicy, T, UseGPU>;
#else
  using StorageType = typename internal::storage<ExecPolicy, T, false>;
#endif

  storage(int n) : data(StorageType::alloc(n)), elems(n)
  {
    StorageType::ready();
  }

  ~storage() { StorageType::free(data); }
  T* ibegin() { return data; }
  T* iend() { return data + elems; }
  T* obegin() { return data; }
  T* oend() { return data + elems; }
  int size() { return elems; }
  void update() { StorageType::ready(); }

private:
  T* data;
  int elems;
};

template <typename ExecPolicy, typename T>
struct storage<ExecPolicy, T, false> : public storage_base {
  using type = T;

#ifdef RAJA_ENABLE_CUDA
  using StorageType = typename internal::
      storage<ExecPolicy, T, RAJA::is_cuda_policy<ExecPolicy>::value>;
#else
  using StorageType = typename internal::storage<ExecPolicy, T, false>;
#endif

  storage(int n)
      : in(StorageType::alloc(n)), out(StorageType::alloc(n)), elems(n)
  {
    StorageType::ready();
  }
  ~storage()
  {
    StorageType::free(in);
    StorageType::free(out);
  }
  T* ibegin() { return in; }
  T* iend() { return in + elems; }
  T* obegin() { return out; }
  T* oend() { return out + elems; }
  int size() { return elems; }
  void update() { StorageType::ready(); }

private:
  T* in;
  T* out;
  int elems;
};

#endif
