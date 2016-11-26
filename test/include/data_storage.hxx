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
  using iterator = T*;
  T* _begin;
  int _size;
  explicit container(T* begin, int size) : _begin{begin}, _size{size} {}
  T* begin() { return _begin; }
  T* end() { return _begin + _size; }
  const T* begin() const { return _begin; }
  const T* end() const { return _begin + _size; }
  int size() const { return _size; }
};

struct storage_base {
};

template <typename ExecPolicy, typename T, bool inplace>
struct storage : public storage_base {
};

template <typename ExecPolicy, typename T>
struct storage<ExecPolicy, T, true> : public storage_base {
  using type = T;
  using data_type = container<T>;

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
      : data{StorageType::alloc(n)}, elems{n}, _in{data, n}, _out{data, n}
  {
    StorageType::ready();
  }

  ~storage() { StorageType::free(data); }
  int size() const { return elems; }
  void update() const { StorageType::ready(); }
private:
  T* data;
  int elems;
  container<T> _in;
  container<T> _out;

public:
  const container<T>& cin() const { return _in; }

  const container<T>& cout() const { return _out; }

  container<T>& in() { return _in; }

  container<T>& out() { return _out; }
};

template <typename ExecPolicy, typename T>
struct storage<ExecPolicy, T, false> : public storage_base {
  using type = T;
  using data_type = container<T>;

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
      : __in{StorageType::alloc(n)},
        __out{StorageType::alloc(n)},
        elems{n},
        _in{__in, n},
        _out{__out, n}
  {
    StorageType::ready();
  }

  ~storage()
  {
    StorageType::free(__in);
    StorageType::free(__out);
  }
  int size() const { return elems; }
  void update() const { StorageType::ready(); }
private:
  T* __in;
  T* __out;
  int elems;

  container<T> _in;
  container<T> _out;

public:
public:
  const container<T>& cin() const { return _in; }

  const container<T>& cout() const { return _out; }

  container<T>& in() { return _in; }

  container<T>& out() { return _out; }
};

#endif
