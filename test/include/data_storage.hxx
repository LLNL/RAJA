#ifndef _STORAGE_HPP_
#define _STORAGE_HPP_

#include <cstdio>

#include <RAJA/RAJA.hxx>

namespace internal
{

template <typename ExecPolicy, typename T, bool gpu = false>
struct storage {
  using type = T;

  static T* alloc(int n)
  {
    T* ptr;
    ::posix_memalign((void**)&ptr, 64, n * sizeof(T));
    return ptr;
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

template <typename T>
struct container {
  T* _begin;
  int _size;
  explicit container(T* begin, int size) : _begin{begin}, _size{size} {}
  T* begin() { return _begin; }
  T* end() { return _begin + _size; }
  int size() { return _size; }
};

struct storage_base {
};

template <typename ExecPolicy, typename T, bool inplace>
struct storage : public storage_base {
};

template <typename ExecPolicy, typename T>
struct storage<ExecPolicy, T, true> : public storage_base {
  using type = T;

#ifdef RAJA_ENABLE_CUDA
  using StorageType =
      typename internal::storage<ExecPolicy,
                                 T,
                                 std::is_base_of<RAJA::cuda_exec_base,
                                                 ExecPolicy>::value>;
#else
  using StorageType = typename internal::storage<ExecPolicy, T>;
#endif

  storage(int n)
      : data{StorageType::alloc(n)}, elems{n}, in{data, n}, out{data, n}
  {
    StorageType::ready();
  }

  ~storage() { StorageType::free(data); }
  int size() { return elems; }
  void update() { StorageType::ready(); }
private:
  T* data;
  int elems;

public:
  container<T> in;
  container<T> out;
};

template <typename ExecPolicy, typename T>
struct storage<ExecPolicy, T, false> : public storage_base {
  using type = T;

#ifdef RAJA_ENABLE_CUDA
  using StorageType =
      typename internal::storage<ExecPolicy,
                                 T,
                                 std::is_base_of<RAJA::cuda_exec_base,
                                                 ExecPolicy>::value>;
#else
  using StorageType = typename internal::storage<ExecPolicy, T>;
#endif

  storage(int n)
      : _in{StorageType::alloc(n)},
        _out{StorageType::alloc(n)},
        elems{n},
        in{_in, n},
        out{_out, n}
  {
    StorageType::ready();
  }

  ~storage()
  {
    StorageType::free(_in);
    StorageType::free(_out);
  }
  int size() { return elems; }
  void update() { StorageType::ready(); }
private:
  T* _in;
  T* _out;
  int elems;

public:
  container<T> in;
  container<T> out;
};

#endif
