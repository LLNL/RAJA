#ifndef _STORAGE_HPP_
#define _STORAGE_HPP_

#include <cstdio>

#include <RAJA/RAJA.hxx>

namespace internal
{

template <typename ExecPolicy, typename T>
struct storage {
  using type = T;

  static T* alloc(int n)
  {
    T* ptr;
    ::posix_memalign((void**)&ptr, 64, n * sizeof(T));
    return ptr;
  }

  static void free(T* ptr) { free(ptr); }

  static void ready() {}
};

#ifdef RAJA_USE_CUDA

template <typename T>
struct storage<cuda_exec_base, T> {
  using type = T;

  static T* alloc(int n)
  {
    T* ptr;
    cudaMallocManaged(&ptr, n * sizeof(T));
    return ptr;
  }
  static void free(T* ptr) { cudaFree(ptr); }
  static void ready() { cudaDeviceSynchronize(); }
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
  storage(int n) : data{internal::storage<ExecPolicy, T>::alloc(n)}, elems{n}
  {
    internal::storage<ExecPolicy, T>::ready();
  }
  ~storage() { internal::storage<ExecPolicy, T>::free(data); }
  T* ibegin() { return data; }
  T* iend() { return data + elems; }
  T* obegin() { return data; }
  T* oend() { return data + elems; }
  int size() { return elems; }
  void update() { internal::storage<ExecPolicy, T>::ready(); }
private:
  T* data;
  int elems;
};

template <typename ExecPolicy, typename T>
struct storage<ExecPolicy, T, false> : public storage_base {
  using type = T;
  storage(int n)
      : in{internal::storage<ExecPolicy, T>::alloc(n)},
        out{internal::storage<ExecPolicy, T>::alloc(n)},
        elems{n}
  {
    internal::storage<ExecPolicy, T>::ready();
  }
  ~storage()
  {
    internal::storage<ExecPolicy, T>::free(in);
    internal::storage<ExecPolicy, T>::free(out);
  }
  T* ibegin() { return in; }
  T* iend() { return in + elems; }
  T* obegin() { return out; }
  T* oend() { return out + elems; }
  int size() { return elems; }
  void update() { internal::storage<ExecPolicy, T>::ready(); }
private:
  T* in;
  T* out;
  int elems;
};

#endif
