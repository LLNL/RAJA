
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-19, Lawrence Livermore National Security, LLC.
//
// Produced at the Lawrence Livermore National Laboratory
//
// LLNL-CODE-689114
//
// All rights reserved.
//
// This file is part of RAJA.
//
// For details about use and distribution, please read RAJA/LICENSE.
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

///
/// Source file containing basic functional tests for non-arithmetic atomic operations using forall
///

#include <RAJA/RAJA.hpp>
#include "RAJA_gtest.hpp"
#include <type_traits>
#include "RAJA_value_params.hpp"

template < typename T >
RAJA_INLINE
RAJA_HOST_DEVICE
typename std::enable_if<sizeof(T) == 1, T>::type np2m1(T val)
{
  val |= val >> 1  ;
  val |= val >> 2  ;
  val |= val >> 4  ;
  return val;
}

template < typename T >
RAJA_INLINE
RAJA_HOST_DEVICE
typename std::enable_if<sizeof(T) == 2, T>::type np2m1(T val)
{
  val |= val >> 1  ;
  val |= val >> 2  ;
  val |= val >> 4  ;
  val |= val >> 8  ;
  return val;
}

template < typename T >
RAJA_INLINE
RAJA_HOST_DEVICE
typename std::enable_if<sizeof(T) == 4, T>::type np2m1(T val)
{
  val |= val >> 1  ;
  val |= val >> 2  ;
  val |= val >> 4  ;
  val |= val >> 8  ;
  val |= val >> 16 ;
  return val;
}

template < typename T >
RAJA_INLINE
RAJA_HOST_DEVICE
typename std::enable_if<sizeof(T) == 8, T>::type np2m1(T val)
{
  val |= val >> 1  ;
  val |= val >> 2  ;
  val |= val >> 4  ;
  val |= val >> 8  ;
  val |= val >> 16 ;
  val |= val >> 32 ;
  return val;
}

template < typename T >
RAJA_INLINE
RAJA_HOST_DEVICE
typename std::enable_if<sizeof(T) == 16, T>::type np2m1(T val)
{
  val |= val >> 1  ;
  val |= val >> 2  ;
  val |= val >> 4  ;
  val |= val >> 8  ;
  val |= val >> 16 ;
  val |= val >> 32 ;
  val |= val >> 64 ;
  return val;
}

template < typename T, typename AtomicPolicy >
struct AndEqOtherOp {
  AndEqOtherOp(T* count, RAJA::RangeSegment seg)
    : other(count), min(T(0)), max((T)seg.size()),
    final_min(min), final_max(min)
  { count[0] = np2m1((T)seg.size()); }
  RAJA_HOST_DEVICE
    T operator()(RAJA::Index_type i) const
    { return other &= (T)i; }
  RAJA::AtomicRef<T, AtomicPolicy> other;
  T min, max, final_min, final_max;
};

template < typename T, typename AtomicPolicy >
struct FetchAndOtherOp {
  FetchAndOtherOp(T* count, RAJA::RangeSegment seg)
    : other(count), min(T(0)), max(np2m1((T)seg.size())),
    final_min(min), final_max(min)
  { count[0] = max; }
  RAJA_HOST_DEVICE
    T operator()(RAJA::Index_type i) const
    { return other.fetch_and((T)i); }
  RAJA::AtomicRef<T, AtomicPolicy> other;
  T min, max, final_min, final_max;
};

template < typename T, typename AtomicPolicy >
struct OrEqOtherOp {
  OrEqOtherOp(T* count, RAJA::RangeSegment seg)
    : other(count), min(T(0)), max(np2m1((T)seg.size())),
    final_min(max), final_max(max)
  { count[0] = T(0); }
  RAJA_HOST_DEVICE
    T operator()(RAJA::Index_type i) const
    { return other |= (T)i; }
  RAJA::AtomicRef<T, AtomicPolicy> other;
  T min, max, final_min, final_max;
};

template < typename T, typename AtomicPolicy >
struct FetchOrOtherOp {
  FetchOrOtherOp(T* count, RAJA::RangeSegment seg)
    : other(count), min(T(0)), max(np2m1((T)seg.size())),
    final_min(max), final_max(max)
  { count[0] = T(0); }
  RAJA_HOST_DEVICE
    T operator()(RAJA::Index_type i) const
    { return other.fetch_or((T)i); }
  RAJA::AtomicRef<T, AtomicPolicy> other;
  T min, max, final_min, final_max;
};

template < typename T, typename AtomicPolicy >
struct XorEqOtherOp {
  XorEqOtherOp(T* count, RAJA::RangeSegment seg)
    : other(count), min(T(0)), max(np2m1((T)seg.size())),
    final_min(min), final_max(min)
  { count[0] = T(0);
    for (RAJA::Index_type i = 0; i < seg.size(); ++i) {
      final_min ^= (T)i; final_max ^= (T)i;
    } }
  RAJA_HOST_DEVICE
    T operator()(RAJA::Index_type i) const
    { return other ^= (T)i; }
  RAJA::AtomicRef<T, AtomicPolicy> other;
  T min, max, final_min, final_max;
};

template < typename T, typename AtomicPolicy >
struct FetchXorOtherOp {
  FetchXorOtherOp(T* count, RAJA::RangeSegment seg)
    : other(count), min(T(0)), max(np2m1((T)seg.size())),
    final_min(min), final_max(min)
  { count[0] = T(0);
    for (RAJA::Index_type i = 0; i < seg.size(); ++i) {
      final_min ^= (T)i; final_max ^= (T)i;
    } }
  RAJA_HOST_DEVICE
    T operator()(RAJA::Index_type i) const
    { return other.fetch_xor((T)i); }
  RAJA::AtomicRef<T, AtomicPolicy> other;
  T min, max, final_min, final_max;
};

template < typename T, typename AtomicPolicy >
struct LoadOtherOp {
  LoadOtherOp(T* count, RAJA::RangeSegment seg)
    : other(count), min((T)seg.size()), max(min),
    final_min(min), final_max(min)
  { count[0] = min; }
  RAJA_HOST_DEVICE
    T operator()(RAJA::Index_type RAJA_UNUSED_ARG(i)) const
    { return other.load(); }
  RAJA::AtomicRef<T, AtomicPolicy> other;
  T min, max, final_min, final_max;
};

template < typename T, typename AtomicPolicy >
struct OperatorTOtherOp {
  OperatorTOtherOp(T* count, RAJA::RangeSegment RAJA_UNUSED_ARG(seg))
    : other(count), min(T(0)), max(min),
    final_min(min), final_max(min)
  { count[0] = min; }
  RAJA_HOST_DEVICE
    T operator()(RAJA::Index_type RAJA_UNUSED_ARG(i)) const
    { return other; }
  RAJA::AtomicRef<T, AtomicPolicy> other;
  T min, max, final_min, final_max;
};

template < typename T, typename AtomicPolicy >
struct StoreOtherOp {
  StoreOtherOp(T* count, RAJA::RangeSegment seg)
    : other(count), min((T)0), max((T)seg.size() - (T)1),
    final_min(min), final_max(max)
  { count[0] = (T)seg.size(); }
  RAJA_HOST_DEVICE
    T operator()(RAJA::Index_type i) const
    { other.store((T)i); return (T)i; }
  RAJA::AtomicRef<T, AtomicPolicy> other;
  T min, max, final_min, final_max;
};

template < typename T, typename AtomicPolicy >
struct AssignOtherOp {
  AssignOtherOp(T* count, RAJA::RangeSegment seg)
    : other(count), min(T(0)), max((T)seg.size() - (T)1),
    final_min(min), final_max(max)
  { count[0] = (T)seg.size(); }
  RAJA_HOST_DEVICE
    T operator()(RAJA::Index_type i) const
    { return (other = (T)i); }
  RAJA::AtomicRef<T, AtomicPolicy> other;
  T min, max, final_min, final_max;
};

template < typename T, typename AtomicPolicy >
struct CASOtherOp {
  CASOtherOp(T* count, RAJA::RangeSegment seg)
    : other(count), min((T)0), max((T)seg.size() - (T)1),
    final_min(min), final_max(max)
  { count[0] = (T)0; }
  RAJA_HOST_DEVICE
    T operator()(RAJA::Index_type i) const
    {
      T received, expect = (T)0;
      while ((received = other.CAS(expect, (T)i)) != expect) {
        expect = received;
      }
      return received;
    }
  RAJA::AtomicRef<T, AtomicPolicy> other;
  T min, max, final_min, final_max;
};

template < typename T, typename AtomicPolicy >
struct CompareExchangeWeakOtherOp {
  CompareExchangeWeakOtherOp(T* count, RAJA::RangeSegment seg)
    : other(count), min((T)0), max((T)seg.size() - (T)1),
    final_min(min), final_max(max)
  { count[0] = (T)0; }
  RAJA_HOST_DEVICE
    T operator()(RAJA::Index_type i) const
    {
      T expect = (T)0;
      while (!other.compare_exchange_weak(expect, (T)i)) {}
      return expect;
    }
  RAJA::AtomicRef<T, AtomicPolicy> other;
  T min, max, final_min, final_max;
};

template < typename T, typename AtomicPolicy >
struct CompareExchangeStrongOtherOp {
  CompareExchangeStrongOtherOp(T* count, RAJA::RangeSegment seg)
    : other(count), min((T)0), max((T)seg.size() - (T)1),
    final_min(min), final_max(max)
  { count[0] = (T)0; }
  RAJA_HOST_DEVICE
    T operator()(RAJA::Index_type i) const
    {
      T expect = (T)0;
      while (!other.compare_exchange_strong(expect, (T)i)) {}
      return expect;
    }
  RAJA::AtomicRef<T, AtomicPolicy> other;
  T min, max, final_min, final_max;
};

template < typename T, typename AtomicPolicy >
struct MaxEqOtherOp {
  MaxEqOtherOp(T* count, RAJA::RangeSegment seg)
    : other(count), min(T(0)), max((T)seg.size() - (T)1),
    final_min(max), final_max(max)
  { count[0] = (T)0; }
  RAJA_HOST_DEVICE
    T operator()(RAJA::Index_type i) const
    { return other.max((T)i); }
  RAJA::AtomicRef<T, AtomicPolicy> other;
  T min, max, final_min, final_max;
};

template < typename T, typename AtomicPolicy >
struct FetchMaxOtherOp {
  FetchMaxOtherOp(T* count, RAJA::RangeSegment seg)
    : other(count), min(T(0)), max((T)seg.size() - (T)1),
    final_min(max), final_max(max)
  { count[0] = (T)0; }
  RAJA_HOST_DEVICE
    T operator()(RAJA::Index_type i) const
    { return other.fetch_max((T)i); }
  RAJA::AtomicRef<T, AtomicPolicy> other;
  T min, max, final_min, final_max;
};

template < typename T, typename AtomicPolicy >
struct MinEqOtherOp {
  MinEqOtherOp(T* count, RAJA::RangeSegment seg)
    : other(count), min(T(0)), max((T)seg.size() - (T)1),
    final_min(min), final_max(min)
  { count[0] = (T)seg.size(); }
  RAJA_HOST_DEVICE
    T operator()(RAJA::Index_type i) const
    { return other.min((T)i); }
  RAJA::AtomicRef<T, AtomicPolicy> other;
  T min, max, final_min, final_max;
};

template < typename T, typename AtomicPolicy >
struct FetchMinOtherOp {
  FetchMinOtherOp(T* count, RAJA::RangeSegment seg)
    : other(count), min(T(0)), max((T)seg.size()),
    final_min(min), final_max(min)
  { count[0] = (T)seg.size(); }
  RAJA_HOST_DEVICE
    T operator()(RAJA::Index_type i) const
    { return other.fetch_min((T)i); }
  RAJA::AtomicRef<T, AtomicPolicy> other;
  T min, max, final_min, final_max;
};


template <typename ExecPolicy,
         typename AtomicPolicy,
         typename T,
  template <typename, typename> class OtherOp>
void testAtomicRefOther(RAJA::RangeSegment seg, T* count, T* list)
{
  OtherOp<T, AtomicPolicy> otherop(count, seg);
  RAJA::forall<ExecPolicy>(seg, [=] RAJA_HOST_DEVICE(RAJA::Index_type i) {
      list[i] = otherop.max + (T)1;
      });
  RAJA::forall<ExecPolicy>(seg, [=] RAJA_HOST_DEVICE(RAJA::Index_type i) {
      T val = otherop(i);
      list[i] = val;
      });
#if defined(RAJA_ENABLE_CUDA)
  cudaErrchk(cudaDeviceSynchronize());
#endif
  EXPECT_LE(otherop.final_min, count[0]);
  EXPECT_GE(otherop.final_max, count[0]);
  for (RAJA::Index_type i = 0; i < seg.size(); i++) {
    EXPECT_LE(otherop.min, list[i]);
    EXPECT_GE(otherop.max, list[i]);
  }
}



template <typename ExecPolicy,
         typename AtomicPolicy,
         typename T,
  RAJA::Index_type N>
void testAtomicRefIntegral()
{
  RAJA::RangeSegment seg(0, N);

  // initialize an array
#if defined(RAJA_ENABLE_CUDA)
  T *count = nullptr;
  cudaErrchk(cudaMallocManaged((void **)&count, sizeof(T) * 1));
  T *list;
  cudaErrchk(cudaMallocManaged((void **)&list, sizeof(T) * N));
  bool *hit;
  cudaErrchk(cudaMallocManaged((void **)&hit, sizeof(bool) * N));
  cudaErrchk(cudaDeviceSynchronize());
#else
  T *count  = new T[1];
  T *list   = new T[N];
  bool *hit = new bool[N];
#endif

  testAtomicRefOther<ExecPolicy, AtomicPolicy, T, LoadOtherOp     >(seg, count, list);
  testAtomicRefOther<ExecPolicy, AtomicPolicy, T, OperatorTOtherOp>(seg, count, list);
  testAtomicRefOther<ExecPolicy, AtomicPolicy, T, StoreOtherOp    >(seg, count, list);
  testAtomicRefOther<ExecPolicy, AtomicPolicy, T, AssignOtherOp   >(seg, count, list);

  testAtomicRefOther<ExecPolicy, AtomicPolicy, T, CASOtherOp                  >(seg, count, list);
  testAtomicRefOther<ExecPolicy, AtomicPolicy, T, CompareExchangeWeakOtherOp  >(seg, count, list);
  testAtomicRefOther<ExecPolicy, AtomicPolicy, T, CompareExchangeStrongOtherOp>(seg, count, list);

  testAtomicRefOther<ExecPolicy, AtomicPolicy, T, MaxEqOtherOp   >(seg, count, list);
  testAtomicRefOther<ExecPolicy, AtomicPolicy, T, FetchMaxOtherOp>(seg, count, list);
  testAtomicRefOther<ExecPolicy, AtomicPolicy, T, MinEqOtherOp   >(seg, count, list);
  testAtomicRefOther<ExecPolicy, AtomicPolicy, T, FetchMinOtherOp>(seg, count, list);

  testAtomicRefOther<ExecPolicy, AtomicPolicy, T, AndEqOtherOp   >(seg, count, list);
  testAtomicRefOther<ExecPolicy, AtomicPolicy, T, FetchAndOtherOp>(seg, count, list);
  testAtomicRefOther<ExecPolicy, AtomicPolicy, T, OrEqOtherOp    >(seg, count, list);
  testAtomicRefOther<ExecPolicy, AtomicPolicy, T, FetchOrOtherOp >(seg, count, list);
  testAtomicRefOther<ExecPolicy, AtomicPolicy, T, XorEqOtherOp   >(seg, count, list);
  testAtomicRefOther<ExecPolicy, AtomicPolicy, T, FetchXorOtherOp>(seg, count, list);

#if defined(RAJA_ENABLE_CUDA)
  cudaErrchk(cudaFree(hit));
  cudaErrchk(cudaFree(list));
  cudaErrchk(cudaFree(count));
#else
  delete[] hit;
  delete[] list;
  delete[] count;
#endif
}

template <typename ExecPolicy,
         typename AtomicPolicy,
         typename T,
  RAJA::Index_type N>
void testAtomicRefFloating()
{
  RAJA::RangeSegment seg(0, N);

  // initialize an array
#if defined(RAJA_ENABLE_CUDA)
  T *count = nullptr;
  cudaErrchk(cudaMallocManaged((void **)&count, sizeof(T) * 1));
  T *list;
  cudaErrchk(cudaMallocManaged((void **)&list, sizeof(T) * N));
  bool *hit;
  cudaErrchk(cudaMallocManaged((void **)&hit, sizeof(bool) * N));
  cudaErrchk(cudaDeviceSynchronize());
#else
  T *count  = new T[1];
  T *list   = new T[N];
  bool *hit = new bool[N];
#endif

  testAtomicRefOther<ExecPolicy, AtomicPolicy, T, LoadOtherOp     >(seg, count, list);
  testAtomicRefOther<ExecPolicy, AtomicPolicy, T, OperatorTOtherOp>(seg, count, list);
  testAtomicRefOther<ExecPolicy, AtomicPolicy, T, StoreOtherOp    >(seg, count, list);
  testAtomicRefOther<ExecPolicy, AtomicPolicy, T, AssignOtherOp   >(seg, count, list);

  testAtomicRefOther<ExecPolicy, AtomicPolicy, T, CASOtherOp                  >(seg, count, list);
  testAtomicRefOther<ExecPolicy, AtomicPolicy, T, CompareExchangeWeakOtherOp  >(seg, count, list);
  testAtomicRefOther<ExecPolicy, AtomicPolicy, T, CompareExchangeStrongOtherOp>(seg, count, list);

  testAtomicRefOther<ExecPolicy, AtomicPolicy, T, MaxEqOtherOp   >(seg, count, list);
  testAtomicRefOther<ExecPolicy, AtomicPolicy, T, FetchMaxOtherOp>(seg, count, list);
  testAtomicRefOther<ExecPolicy, AtomicPolicy, T, MinEqOtherOp   >(seg, count, list);
  testAtomicRefOther<ExecPolicy, AtomicPolicy, T, FetchMinOtherOp>(seg, count, list);

#if defined(RAJA_ENABLE_CUDA)
  cudaErrchk(cudaFree(hit));
  cudaErrchk(cudaFree(list));
  cudaErrchk(cudaFree(count));
#else
  delete[] hit;
  delete[] list;
  delete[] count;
#endif
}

