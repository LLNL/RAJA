//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-23, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

//
// Header defining "for one" unit test utility so that constructs can be
// tested outside of standard RAJA kernel launch utilities (forall, kernel).
//

#ifndef __RAJA_test_forone_HPP__
#define __RAJA_test_forone_HPP__

#include <cstdlib>
#include <cstring>
#include <utility>
#include <stdexcept>
#include <type_traits>

#include "RAJA_unit-test-policy.hpp"

// A class like unique_ptr but allows non-owning copies.
// Of all instances of origin_ptr with a certain value of ptr, at most one
// may own the resource and will releasing the resource based on object lifetimes.
// Note that non-owning instances may dangle.
// Note that all functions are host device so Deleter must have a host device
// call operator. If the release implementation only exists on the host then
// you will have to have to ifdef out release in device code.
template < typename T, typename Deleter >
struct original_ptr : private Deleter
{
  using pointer = T*;
  using element_type = T;
  using element_reference = T&;
  using deleter_type = Deleter;

  static_assert(std::is_same<std::remove_reference_t<Deleter>, Deleter>::value,
      "Does not accept reference Deleters");

  RAJA_HOST_DEVICE
  explicit original_ptr(pointer ptr = nullptr, Deleter const& aloc = Deleter{}) noexcept
    : Deleter(aloc)
    , m_ptr(ptr)
    , m_original(ptr != nullptr)
  {}
  RAJA_HOST_DEVICE
  explicit original_ptr(pointer ptr = nullptr, Deleter && aloc = Deleter{}) noexcept
    : Deleter(std::move(aloc))
    , m_ptr(ptr)
    , m_original(ptr != nullptr)
  {}

  RAJA_HOST_DEVICE
  original_ptr(original_ptr const& rhs)
    : Deleter(rhs.get_deleter())
    , m_ptr(rhs.m_ptr)
    , m_original(false)
  {}
  RAJA_HOST_DEVICE
  original_ptr& operator=(original_ptr const& rhs)
  {
    if (&rhs != this) {
      reset(nullptr);
      get_deleter() = rhs.get_deleter();
      m_ptr = rhs.m_ptr;
      m_original = false;
    }
    return *this;
  }

  RAJA_HOST_DEVICE
  original_ptr(original_ptr && rhs) noexcept
    : Deleter(std::move(rhs.get_deleter()))
    , m_ptr(rhs.m_ptr)
    , m_original(rhs.m_original)
  {
    rhs.m_original = false;
  }
  RAJA_HOST_DEVICE
  original_ptr& operator=(original_ptr && rhs) noexcept
  {
    reset(nullptr);
    get_deleter() = std::move(rhs.get_deleter());
    m_ptr = rhs.m_ptr;
    m_original = rhs.m_original;
    rhs.m_original = false;
    return *this;
  }

  RAJA_HOST_DEVICE
  ~original_ptr()
  {
    reset(nullptr);
  }

  // take ownership of ptr
  RAJA_HOST_DEVICE
  void reset(pointer ptr = nullptr) noexcept
  {
    if (m_original) {
      get_deleter()(m_ptr);
    }
    m_ptr = ptr;
    m_original = (ptr != nullptr);
  }

  // only releases the pointer if this is the original
  // if not nulls this pointer and returns nullptr
  RAJA_HOST_DEVICE
  pointer release() noexcept
  {
    pointer ptr = nullptr;
    if (m_original) {
      ptr = m_ptr;
      m_original = false;
    }
    m_ptr = nullptr;
    return ptr;
  }


  RAJA_HOST_DEVICE
  pointer get() const noexcept
  {
    return m_ptr;
  }

  RAJA_HOST_DEVICE
  Deleter& get_deleter() noexcept
  {
    return static_cast<Deleter&>(*this);
  }
  RAJA_HOST_DEVICE
  Deleter const& get_deleter() const noexcept
  {
    return static_cast<Deleter const&>(*this);
  }

  RAJA_HOST_DEVICE
  explicit operator bool() const noexcept
  {
    return (m_ptr != nullptr);
  }


  RAJA_HOST_DEVICE
  element_reference operator*() const
  {
    return *m_ptr;
  }
  RAJA_HOST_DEVICE
  pointer operator->() const
  {
    return m_ptr;
  }

  RAJA_HOST_DEVICE
  element_reference operator[](std::ptrdiff_t i) const
  {
    return m_ptr[i];
  }

private:
  pointer m_ptr = nullptr;
  bool m_original = false;
};


template < typename policy, typename T >
inline T* allocate(size_t size);

template < typename policy, typename T >
inline void deallocate(T* ptr);

template < typename dst_policy, typename src_policy, typename T >
inline T* memcpy(T* dst, T* src, size_t size);



// test_seq implementation
template < typename T >
inline T* allocate(test_seq, size_t size)
{
  void* ptr = std::malloc(size*sizeof(T));
  return static_cast<T*>(ptr);
}

template < typename T >
inline void deallocate(test_seq, T* ptr)
{
  std::free(ptr);
}

template < typename T >
inline T* memcpy(test_seq, test_seq, T* dst, T* src, size_t size)
{
  std::memcpy(dst, src, size*sizeof(T));
  return dst;
}


#if defined(RAJA_ENABLE_TARGET_OPENMP)

// test_openmp_target implementation
template < typename T >
inline T* allocate(test_openmp_target, size_t size)
{
  int did = omp_get_default_device();
  void* ptr = omp_target_alloc(size*sizeof(T), did);
  return static_cast<T*>(ptr);
}

template < typename T >
inline void deallocate(test_openmp_target, T* ptr)
{
  int did = omp_get_default_device();
  omp_target_free(ptr, did);
}

template < typename T >
inline T* memcpy(test_seq, test_openmp_target, T* dst, T* src, size_t size)
{
  int hid = omp_get_initial_device();
  int did = omp_get_default_device();
  omp_target_memcpy(dst, src, size*sizeof(T),
      0, 0, hid, did);
  return dst;
}

template < typename T >
inline T* memcpy(test_openmp_target, test_seq, T* dst, T* src, size_t size)
{
  int hid = omp_get_initial_device();
  int did = omp_get_default_device();
  omp_target_memcpy(dst, src, size*sizeof(T),
      0, 0, did, hid);
  return dst;
}

template < typename T >
inline T* memcpy(test_openmp_target, test_openmp_target, T* dst, T* src, size_t size)
{
  int did = omp_get_default_device();
  omp_target_memcpy(dst, src, size*sizeof(T),
      0, 0, did, did);
  return dst;
}

#endif


#if defined(RAJA_ENABLE_CUDA)

// test_cuda implementation
template < typename T >
inline T* allocate(test_cuda, size_t size)
{
  void* ptr = nullptr;
  cudaErrchk(cudaMallocManaged(&ptr, size*sizeof(T)));
  cudaErrchk(cudaDeviceSynchronize());
  return static_cast<T*>(ptr);
}

template < typename T >
inline void deallocate(test_cuda, T* ptr)
{
  cudaErrchk(cudaFree(ptr));
}

template < typename T >
inline T* memcpy(test_seq, test_cuda, T* dst, T* src, size_t size)
{
  cudaErrchk(cudaMemcpy(dst, src, size*sizeof(T), cudaMemcpyDeviceToHost));
  return dst;
}

template < typename T >
inline T* memcpy(test_cuda, test_seq, T* dst, T* src, size_t size)
{
  cudaErrchk(cudaMemcpy(dst, src, size*sizeof(T), cudaMemcpyHostToDevice));
  return dst;
}

template < typename T >
inline T* memcpy(test_cuda, test_cuda, T* dst, T* src, size_t size)
{
  cudaErrchk(cudaMemcpy(dst, src, size*sizeof(T), cudaMemcpyDeviceToDevice));
  return dst;
}

#endif

#if defined(RAJA_ENABLE_HIP)

// test_hip implementation
template < typename T >
inline T* allocate(test_hip, size_t size)
{
  void* ptr = nullptr;
  hipErrchk(hipMalloc(&ptr, size*sizeof(T)));
  hipErrchk(hipDeviceSynchronize());
  return static_cast<T*>(ptr);
}

template < typename T >
inline void deallocate(test_hip, T* ptr)
{
  hipErrchk(hipFree(ptr));
}

template < typename T >
inline T* memcpy(test_seq, test_hip, T* dst, T* src, size_t size)
{
  hipErrchk(hipMemcpy(dst, src, size*sizeof(T), hipMemcpyDeviceToHost));
  return dst;
}

template < typename T >
inline T* memcpy(test_hip, test_seq, T* dst, T* src, size_t size)
{
  hipErrchk(hipMemcpy(dst, src, size*sizeof(T), hipMemcpyHostToDevice));
  return dst;
}

template < typename T >
inline T* memcpy(test_hip, test_hip, T* dst, T* src, size_t size)
{
  hipErrchk(hipMemcpy(dst, src, size*sizeof(T), hipMemcpyDeviceToDevice));
  return dst;
}

#endif


template < typename policy, typename T >
inline T* allocate(size_t size)
{
  return allocate<T>(policy{}, size);
}

template < typename policy, typename T >
inline void deallocate(T* ptr)
{
  deallocate(policy{}, ptr);
}

template < typename dst_policy, typename src_policy, typename T >
inline T* memcpy(T* dst, T* src, size_t size)
{
  return memcpy(dst_policy{}, src_policy{}, dst, src, size);
}


template < typename T >
inline T* allocate(test_policy pol, size_t size)
{
  switch (pol) {
  case test_policy::seq:
    return allocate<T>(test_seq{}, size);
#if defined(RAJA_ENABLE_TARGET_OPENMP)
  case test_policy::openmp_target:
    return allocate<T>(test_openmp_target{}, size);
#endif
#if defined(RAJA_ENABLE_CUDA)
  case test_policy::cuda:
    return allocate<T>(test_cuda{}, size);
#endif
#if defined(RAJA_ENABLE_HIP)
  case test_policy::hip:
    return allocate<T>(test_hip{}, size);
#endif
  default:
    throw std::runtime_error("Policy is not valid");
  }
}

template < typename T >
inline void deallocate(test_policy pol, T* ptr)
{
  switch (pol) {
  case test_policy::seq:
    deallocate(test_seq{}, ptr); break;
#if defined(RAJA_ENABLE_TARGET_OPENMP)
  case test_policy::openmp_target:
    deallocate(test_openmp_target{}, ptr); break;
#endif
#if defined(RAJA_ENABLE_CUDA)
  case test_policy::cuda:
    deallocate(test_cuda{}, ptr); break;
#endif
#if defined(RAJA_ENABLE_HIP)
  case test_policy::hip:
    deallocate(test_hip{}, ptr); break;
#endif
  default:
    throw std::runtime_error("Policy is not valid");
  }
}

template < typename T >
inline T* memcpy(test_policy dst_pol, test_policy src_pol, T* dst, T* src, size_t size)
{
  switch (dst_pol) {
  case test_policy::seq:
    switch (src_pol) {
    case test_policy::seq:
      return memcpy(test_seq{}, test_seq{}, dst, src, size);
#if defined(RAJA_ENABLE_TARGET_OPENMP)
    case test_policy::openmp_target:
      return memcpy(test_seq{}, test_openmp_target{}, dst, src, size);
#endif
#if defined(RAJA_ENABLE_CUDA)
    case test_policy::cuda:
      return memcpy(test_seq{}, test_cuda{}, dst, src, size);
#endif
#if defined(RAJA_ENABLE_HIP)
    case test_policy::hip:
      return memcpy(test_seq{}, test_hip{}, dst, src, size);
#endif
    default:
      throw std::runtime_error("Source Policy is not valid");
    }
#if defined(RAJA_ENABLE_TARGET_OPENMP)
  case test_policy::openmp_target:
    switch (src_pol) {
    case test_policy::seq:
      return memcpy(test_openmp_target{}, test_seq{}, dst, src, size);
    case test_policy::openmp_target:
      return memcpy(test_openmp_target{}, test_openmp_target{}, dst, src, size);
    default:
      throw std::runtime_error("Source Policy is not valid");
    }
#endif
#if defined(RAJA_ENABLE_CUDA)
  case test_policy::cuda:
    switch (src_pol) {
    case test_policy::seq:
      return memcpy(test_cuda{}, test_seq{}, dst, src, size);
    case test_policy::cuda:
      return memcpy(test_cuda{}, test_cuda{}, dst, src, size);
    default:
      throw std::runtime_error("Source Policy is not valid");
    }
#endif
#if defined(RAJA_ENABLE_HIP)
  case test_policy::hip:
    switch (src_pol) {
    case test_policy::seq:
      return memcpy(test_hip{}, test_seq{}, dst, src, size);
    case test_policy::hip:
      return memcpy(test_hip{}, test_hip{}, dst, src, size);
    default:
      throw std::runtime_error("Source Policy is not valid");
    }
#endif
  default:
    throw std::runtime_error("Destination Policy is not valid");
  }
}


template < typename T >
struct test_delete
{
  using pointer = T*;

  test_policy m_pol = test_policy::undefined;

  RAJA_HOST_DEVICE
  void operator()(pointer ptr) const
  {
#if !defined(RAJA_GPU_DEVICE_COMPILE_PASS_ACTIVE)
    deallocate(m_pol, ptr);
#else
    RAJA_UNUSED_VAR(ptr);
#endif
  }

};


template < typename T >
using test_ptr = original_ptr<T, test_delete<T>>;

template < typename policy, typename T >
inline test_ptr<T> make_test_ptr(size_t size)
{
  return test_ptr<T>{allocate<policy, T>(size), test_delete<T>{test_policy_info<policy>::pol}};
}

template < typename dst_policy, typename T >
inline test_ptr<T> copy_test_ptr(
    original_ptr<T, test_delete<T>> const& src_ptr, size_t size)
{
  test_policy dst_pol = test_policy_info<dst_policy>::pol;
  test_policy src_pol = src_ptr.get_deleter().m_pol;
  auto dst_ptr = make_test_ptr<dst_policy, T>(size);
  memcpy(dst_pol, src_pol, dst_ptr.get(), src_ptr.get(), size);
  return dst_ptr;
}


#endif // RAJA_test_forone_HPP__
