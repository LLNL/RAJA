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
/// Source file containing tests for atomic operations
///

#include <RAJA/RAJA.hpp>
#include "RAJA_gtest.hpp"


template <typename ExecPolicy,
          typename AtomicPolicy,
          typename T,
          RAJA::Index_type N>
void testAtomicFunctionBasic()
{
  RAJA::RangeSegment seg(0, N);

// initialize an array
#if defined(RAJA_ENABLE_CUDA)
  T *dest = nullptr;
  cudaMallocManaged((void **)&dest, sizeof(T) * 8);

  cudaDeviceSynchronize();

#else
  T *dest = new T[8];
#endif


  // use atomic add to reduce the array
  dest[0] = (T)0;
  dest[1] = (T)N;
  dest[2] = (T)N;
  dest[3] = (T)0;
  dest[4] = (T)0;
  dest[5] = (T)0;
  dest[6] = (T)N + 1;
  dest[7] = (T)0;


  RAJA::forall<ExecPolicy>(seg, [=] RAJA_HOST_DEVICE(RAJA::Index_type i) {
    RAJA::atomic::atomicAdd<AtomicPolicy>(dest + 0, (T)1);
    RAJA::atomic::atomicSub<AtomicPolicy>(dest + 1, (T)1);

    RAJA::atomic::atomicMin<AtomicPolicy>(dest + 2, (T)i);
    RAJA::atomic::atomicMax<AtomicPolicy>(dest + 3, (T)i);
    RAJA::atomic::atomicInc<AtomicPolicy>(dest + 4);
    RAJA::atomic::atomicInc<AtomicPolicy>(dest + 5, (T)16);
    RAJA::atomic::atomicDec<AtomicPolicy>(dest + 6);
    RAJA::atomic::atomicDec<AtomicPolicy>(dest + 7, (T)16);
  });

#if defined(RAJA_ENABLE_CUDA)
  cudaDeviceSynchronize();
#endif

  EXPECT_EQ((T)N, dest[0]);
  EXPECT_EQ((T)0, dest[1]);
  EXPECT_EQ((T)0, dest[2]);
  EXPECT_EQ((T)N - 1, dest[3]);
  EXPECT_EQ((T)N, dest[4]);
  EXPECT_EQ((T)4, dest[5]);
  EXPECT_EQ((T)1, dest[6]);
  EXPECT_EQ((T)13, dest[7]);


#if defined(RAJA_ENABLE_CUDA)
  cudaFree(dest);
#else
  delete[] dest;
#endif
}


template <typename ExecPolicy, typename AtomicPolicy>
void testAtomicFunctionPol()
{
  testAtomicFunctionBasic<ExecPolicy, AtomicPolicy, int, 10000>();
  testAtomicFunctionBasic<ExecPolicy, AtomicPolicy, unsigned, 10000>();
  testAtomicFunctionBasic<ExecPolicy, AtomicPolicy, long long, 10000>();
  testAtomicFunctionBasic<ExecPolicy,
                          AtomicPolicy,
                          unsigned long long,
                          10000>();
  testAtomicFunctionBasic<ExecPolicy, AtomicPolicy, float, 10000>();
  testAtomicFunctionBasic<ExecPolicy, AtomicPolicy, double, 10000>();
}

template < typename T, typename AtomicPolicy >
struct PreIncCountOp {
  PreIncCountOp(T* count, RAJA::RangeSegment seg)
    : counter(count), min((T)0), max((T)seg.size()-(T)1), final((T)seg.size())
  { count[0] = (T)0; }
  RAJA_HOST_DEVICE
  T operator()(RAJA::Index_type RAJA_UNUSED_ARG(i)) const {
    return (++counter) - (T)1;
  }
  RAJA::atomic::AtomicRef<T, AtomicPolicy> counter;
  T min, max, final;
};

template < typename T, typename AtomicPolicy >
struct PostIncCountOp {
  PostIncCountOp(T* count, RAJA::RangeSegment seg)
    : counter(count), min((T)0), max((T)seg.size()-(T)1), final((T)seg.size())
  { count[0] = (T)0; }
  RAJA_HOST_DEVICE
  T operator()(RAJA::Index_type RAJA_UNUSED_ARG(i)) const {
    return (counter++);
  }
  RAJA::atomic::AtomicRef<T, AtomicPolicy> counter;
  T min, max, final;
};

template < typename T, typename AtomicPolicy >
struct AddEqCountOp {
  AddEqCountOp(T* count, RAJA::RangeSegment seg)
    : counter(count), min((T)0), max((T)seg.size()-(T)1), final((T)seg.size())
  { count[0] = (T)0; }
  RAJA_HOST_DEVICE
  T operator()(RAJA::Index_type RAJA_UNUSED_ARG(i)) const {
    return (counter += (T)1) - (T)1;
  }
  RAJA::atomic::AtomicRef<T, AtomicPolicy> counter;
  T min, max, final;
};

template < typename T, typename AtomicPolicy >
struct FetchAddCountOp {
  FetchAddCountOp(T* count, RAJA::RangeSegment seg)
    : counter(count), min((T)0), max((T)seg.size()-(T)1), final((T)seg.size())
  { count[0] = (T)0; }
  RAJA_HOST_DEVICE
  T operator()(RAJA::Index_type RAJA_UNUSED_ARG(i)) const {
    return counter.fetch_add((T)1);
  }
  RAJA::atomic::AtomicRef<T, AtomicPolicy> counter;
  T min, max, final;
};

template < typename T, typename AtomicPolicy >
struct PreDecCountOp {
  PreDecCountOp(T* count, RAJA::RangeSegment seg)
    : counter(count), min((T)0), max((T)seg.size()-(T)1), final((T)0)
  { count[0] = (T)seg.size(); }
  RAJA_HOST_DEVICE
  T operator()(RAJA::Index_type RAJA_UNUSED_ARG(i)) const {
    return (--counter);
  }
  RAJA::atomic::AtomicRef<T, AtomicPolicy> counter;
  T min, max, final;
};

template < typename T, typename AtomicPolicy >
struct PostDecCountOp {
  PostDecCountOp(T* count, RAJA::RangeSegment seg)
    : counter(count), min((T)0), max((T)seg.size()-(T)1), final((T)0)
  { count[0] = (T)seg.size(); }
  RAJA_HOST_DEVICE
  T operator()(RAJA::Index_type RAJA_UNUSED_ARG(i)) const {
    return (counter--) - (T)1;
  }
  RAJA::atomic::AtomicRef<T, AtomicPolicy> counter;
  T min, max, final;
};

template < typename T, typename AtomicPolicy >
struct SubEqCountOp {
  SubEqCountOp(T* count, RAJA::RangeSegment seg)
    : counter(count), min((T)0), max((T)seg.size()-(T)1), final((T)0)
  { count[0] = (T)seg.size(); }
  RAJA_HOST_DEVICE
  T operator()(RAJA::Index_type RAJA_UNUSED_ARG(i)) const {
    return (counter -= (T)1);
  }
  RAJA::atomic::AtomicRef<T, AtomicPolicy> counter;
  T min, max, final;
};

template < typename T, typename AtomicPolicy >
struct FetchSubCountOp {
  FetchSubCountOp(T* count, RAJA::RangeSegment seg)
    : counter(count), min((T)0), max((T)seg.size()-(T)1), final((T)0)
  { count[0] = (T)seg.size(); }
  RAJA_HOST_DEVICE
  T operator()(RAJA::Index_type RAJA_UNUSED_ARG(i)) const {
    return counter.fetch_sub((T)1) - (T)1;
  }
  RAJA::atomic::AtomicRef<T, AtomicPolicy> counter;
  T min, max, final;
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
  RAJA::atomic::AtomicRef<T, AtomicPolicy> other;
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
  RAJA::atomic::AtomicRef<T, AtomicPolicy> other;
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
  RAJA::atomic::AtomicRef<T, AtomicPolicy> other;
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
  RAJA::atomic::AtomicRef<T, AtomicPolicy> other;
  T min, max, final_min, final_max;
};

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
  RAJA::atomic::AtomicRef<T, AtomicPolicy> other;
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
  RAJA::atomic::AtomicRef<T, AtomicPolicy> other;
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
  RAJA::atomic::AtomicRef<T, AtomicPolicy> other;
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
  RAJA::atomic::AtomicRef<T, AtomicPolicy> other;
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
  RAJA::atomic::AtomicRef<T, AtomicPolicy> other;
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
  RAJA::atomic::AtomicRef<T, AtomicPolicy> other;
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
  RAJA::atomic::AtomicRef<T, AtomicPolicy> other;
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
  RAJA::atomic::AtomicRef<T, AtomicPolicy> other;
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
  RAJA::atomic::AtomicRef<T, AtomicPolicy> other;
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
  RAJA::atomic::AtomicRef<T, AtomicPolicy> other;
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
  RAJA::atomic::AtomicRef<T, AtomicPolicy> other;
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
  RAJA::atomic::AtomicRef<T, AtomicPolicy> other;
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
  RAJA::atomic::AtomicRef<T, AtomicPolicy> other;
  T min, max, final_min, final_max;
};

template <typename ExecPolicy,
          typename AtomicPolicy,
          typename T,
          template <typename, typename> class CountOp>
void testAtomicRefCount(RAJA::RangeSegment seg,
                         T* count, T* list, bool* hit)
{
  CountOp<T, AtomicPolicy> countop(count, seg);
  RAJA::forall<ExecPolicy>(seg, [=] RAJA_HOST_DEVICE(RAJA::Index_type i) {
    list[i] = countop.max + (T)1;
    hit[i] = false;
  });
  RAJA::forall<ExecPolicy>(seg, [=] RAJA_HOST_DEVICE(RAJA::Index_type i) {
    T val = countop(i);
    list[i] = val;
    hit[(RAJA::Index_type)val] = true;
  });
#if defined(RAJA_ENABLE_CUDA)
  cudaDeviceSynchronize();
#endif
  EXPECT_EQ(countop.final, count[0]);
  for (RAJA::Index_type i = 0; i < seg.size(); i++) {
    EXPECT_LE(countop.min, list[i]);
    EXPECT_GE(countop.max, list[i]);
    EXPECT_TRUE(hit[i]);
  }
}

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
  cudaDeviceSynchronize();
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
  cudaMallocManaged((void **)&count, sizeof(T) * 1);
  T *list;
  cudaMallocManaged((void **)&list, sizeof(T) * N);
  bool *hit;
  cudaMallocManaged((void **)&hit, sizeof(bool) * N);
  cudaDeviceSynchronize();
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

  testAtomicRefCount<ExecPolicy, AtomicPolicy, T, PreIncCountOp  >(seg, count, list, hit);
  testAtomicRefCount<ExecPolicy, AtomicPolicy, T, PostIncCountOp >(seg, count, list, hit);
  testAtomicRefCount<ExecPolicy, AtomicPolicy, T, AddEqCountOp   >(seg, count, list, hit);
  testAtomicRefCount<ExecPolicy, AtomicPolicy, T, FetchAddCountOp>(seg, count, list, hit);

  testAtomicRefCount<ExecPolicy, AtomicPolicy, T, PreDecCountOp  >(seg, count, list, hit);
  testAtomicRefCount<ExecPolicy, AtomicPolicy, T, PostDecCountOp >(seg, count, list, hit);
  testAtomicRefCount<ExecPolicy, AtomicPolicy, T, SubEqCountOp   >(seg, count, list, hit);
  testAtomicRefCount<ExecPolicy, AtomicPolicy, T, FetchSubCountOp>(seg, count, list, hit);

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
  cudaFree(hit);
  cudaFree(list);
  cudaFree(count);
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
  cudaMallocManaged((void **)&count, sizeof(T) * 1);
  T *list;
  cudaMallocManaged((void **)&list, sizeof(T) * N);
  bool *hit;
  cudaMallocManaged((void **)&hit, sizeof(bool) * N);
  cudaDeviceSynchronize();
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

  testAtomicRefCount<ExecPolicy, AtomicPolicy, T, PreIncCountOp  >(seg, count, list, hit);
  testAtomicRefCount<ExecPolicy, AtomicPolicy, T, PostIncCountOp >(seg, count, list, hit);
  testAtomicRefCount<ExecPolicy, AtomicPolicy, T, AddEqCountOp   >(seg, count, list, hit);
  testAtomicRefCount<ExecPolicy, AtomicPolicy, T, FetchAddCountOp>(seg, count, list, hit);

  testAtomicRefCount<ExecPolicy, AtomicPolicy, T, PreDecCountOp  >(seg, count, list, hit);
  testAtomicRefCount<ExecPolicy, AtomicPolicy, T, PostDecCountOp >(seg, count, list, hit);
  testAtomicRefCount<ExecPolicy, AtomicPolicy, T, SubEqCountOp   >(seg, count, list, hit);
  testAtomicRefCount<ExecPolicy, AtomicPolicy, T, FetchSubCountOp>(seg, count, list, hit);

  testAtomicRefOther<ExecPolicy, AtomicPolicy, T, MaxEqOtherOp   >(seg, count, list);
  testAtomicRefOther<ExecPolicy, AtomicPolicy, T, FetchMaxOtherOp>(seg, count, list);
  testAtomicRefOther<ExecPolicy, AtomicPolicy, T, MinEqOtherOp   >(seg, count, list);
  testAtomicRefOther<ExecPolicy, AtomicPolicy, T, FetchMinOtherOp>(seg, count, list);

#if defined(RAJA_ENABLE_CUDA)
  cudaFree(hit);
  cudaFree(list);
  cudaFree(count);
#else
  delete[] hit;
  delete[] list;
  delete[] count;
#endif
}


template <typename ExecPolicy, typename AtomicPolicy>
void testAtomicRefPol()
{
  testAtomicRefIntegral<ExecPolicy, AtomicPolicy, int, 10000>();
  testAtomicRefIntegral<ExecPolicy, AtomicPolicy, unsigned, 10000>();
  testAtomicRefIntegral<ExecPolicy, AtomicPolicy, long long, 10000>();
  testAtomicRefIntegral<ExecPolicy, AtomicPolicy, unsigned long long, 10000>();

  testAtomicRefFloating<ExecPolicy, AtomicPolicy, float, 10000>();
  testAtomicRefFloating<ExecPolicy, AtomicPolicy, double, 10000>();
}

template <typename ExecPolicy,
          typename AtomicPolicy,
          typename T,
          RAJA::Index_type N>
void testAtomicViewBasic()
{
  RAJA::RangeSegment seg(0, N);
  RAJA::RangeSegment seg_half(0, N / 2);

// initialize an array
#if defined(RAJA_ENABLE_CUDA)
  T *source = nullptr;
  cudaMallocManaged((void **)&source, sizeof(T) * N);

  T *dest = nullptr;
  cudaMallocManaged((void **)&dest, sizeof(T) * N / 2);

  cudaDeviceSynchronize();
#else
  T *source = new T[N];
  T *dest = new T[N / 2];
#endif

  RAJA::forall<RAJA::seq_exec>(seg,
                               [=](RAJA::Index_type i) { source[i] = (T)1; });

  // use atomic add to reduce the array
  RAJA::View<T, RAJA::Layout<1>> vec_view(source, N);

  RAJA::View<T, RAJA::Layout<1>> sum_view(dest, N);
  auto sum_atomic_view = RAJA::make_atomic_view<AtomicPolicy>(sum_view);


  // Zero out dest using atomic view
  RAJA::forall<ExecPolicy>(seg_half, [=] RAJA_HOST_DEVICE(RAJA::Index_type i) {
    sum_atomic_view(i) = (T)0;
  });

  // Assign values to dest using atomic view
  RAJA::forall<ExecPolicy>(seg, [=] RAJA_HOST_DEVICE(RAJA::Index_type i) {
    sum_atomic_view(i / 2) += vec_view(i);
  });

#if defined(RAJA_ENABLE_CUDA)
  cudaDeviceSynchronize();
#endif

  for (RAJA::Index_type i = 0; i < N / 2; ++i) {
    EXPECT_EQ((T)2, dest[i]);
  }

#if defined(RAJA_ENABLE_CUDA)
  cudaFree(source);
  cudaFree(dest);
#else
  delete[] source;
  delete[] dest;
#endif
}


template <typename ExecPolicy, typename AtomicPolicy>
void testAtomicViewPol()
{
  testAtomicViewBasic<ExecPolicy, AtomicPolicy, int, 100000>();
  testAtomicViewBasic<ExecPolicy, AtomicPolicy, unsigned, 100000>();
  testAtomicViewBasic<ExecPolicy, AtomicPolicy, long long, 100000>();
  testAtomicViewBasic<ExecPolicy, AtomicPolicy, unsigned long long, 100000>();
  testAtomicViewBasic<ExecPolicy, AtomicPolicy, float, 100000>();
  testAtomicViewBasic<ExecPolicy, AtomicPolicy, double, 100000>();
}


template <typename ExecPolicy,
          typename AtomicPolicy,
          typename T,
          RAJA::Index_type N>
void testAtomicLogical()
{
  RAJA::RangeSegment seg(0, N * 8);
  RAJA::RangeSegment seg_bytes(0, N);

// initialize an array
#if defined(RAJA_ENABLE_CUDA)
  T *dest_and = nullptr;
  cudaMallocManaged((void **)&dest_and, sizeof(T) * N);

  T *dest_or = nullptr;
  cudaMallocManaged((void **)&dest_or, sizeof(T) * N);

  T *dest_xor = nullptr;
  cudaMallocManaged((void **)&dest_xor, sizeof(T) * N);

  cudaDeviceSynchronize();
#else
  T *dest_and = new T[N];
  T *dest_or = new T[N];
  T *dest_xor = new T[N];
#endif

  RAJA::forall<RAJA::seq_exec>(seg_bytes, [=](RAJA::Index_type i) {
    dest_and[i] = (T)0;
    dest_or[i] = (T)0;
    dest_xor[i] = (T)0;
  });


  RAJA::forall<ExecPolicy>(seg, [=] RAJA_HOST_DEVICE(RAJA::Index_type i) {
    RAJA::Index_type offset = i / 8;
    RAJA::Index_type bit = i % 8;
    RAJA::atomic::atomicAnd<AtomicPolicy>(dest_and + offset,
                                          (T)(0xFF ^ (1 << bit)));
    RAJA::atomic::atomicOr<AtomicPolicy>(dest_or + offset, (T)(1 << bit));
    RAJA::atomic::atomicXor<AtomicPolicy>(dest_xor + offset, (T)(1 << bit));
  });

#if defined(RAJA_ENABLE_CUDA)
  cudaDeviceSynchronize();
#endif

  for (RAJA::Index_type i = 0; i < N; ++i) {
    EXPECT_EQ((T)0x00, dest_and[i]);
    EXPECT_EQ((T)0xFF, dest_or[i]);
    EXPECT_EQ((T)0xFF, dest_xor[i]);
  }

#if defined(RAJA_ENABLE_CUDA)
  cudaFree(dest_and);
  cudaFree(dest_or);
  cudaFree(dest_xor);
#else
  delete[] dest_and;
  delete[] dest_or;
  delete[] dest_xor;
#endif
}


template <typename ExecPolicy, typename AtomicPolicy>
void testAtomicLogicalPol()
{
  testAtomicLogical<ExecPolicy, AtomicPolicy, int, 100000>();
  testAtomicLogical<ExecPolicy, AtomicPolicy, unsigned, 100000>();
  testAtomicLogical<ExecPolicy, AtomicPolicy, long long, 100000>();
  testAtomicLogical<ExecPolicy, AtomicPolicy, unsigned long long, 100000>();
}


#if defined(RAJA_ENABLE_OPENMP)

TEST(Atomic, basic_OpenMP_AtomicFunction)
{
  testAtomicFunctionPol<RAJA::omp_for_exec, RAJA::atomic::auto_atomic>();
  testAtomicFunctionPol<RAJA::omp_for_exec, RAJA::atomic::omp_atomic>();
  testAtomicFunctionPol<RAJA::omp_for_exec, RAJA::atomic::builtin_atomic>();
}


TEST(Atomic, basic_OpenMP_AtomicRef)
{
  testAtomicRefPol<RAJA::omp_for_exec, RAJA::atomic::auto_atomic>();
  testAtomicRefPol<RAJA::omp_for_exec, RAJA::atomic::omp_atomic>();
  testAtomicRefPol<RAJA::omp_for_exec, RAJA::atomic::builtin_atomic>();
}


TEST(Atomic, basic_OpenMP_AtomicView)
{
  testAtomicViewPol<RAJA::omp_for_exec, RAJA::atomic::auto_atomic>();
  testAtomicViewPol<RAJA::omp_for_exec, RAJA::atomic::omp_atomic>();
  testAtomicViewPol<RAJA::omp_for_exec, RAJA::atomic::builtin_atomic>();
}


TEST(Atomic, basic_OpenMP_Logical)
{
  testAtomicLogicalPol<RAJA::omp_for_exec, RAJA::atomic::auto_atomic>();
  testAtomicLogicalPol<RAJA::omp_for_exec, RAJA::atomic::omp_atomic>();
  testAtomicLogicalPol<RAJA::omp_for_exec, RAJA::atomic::builtin_atomic>();
}

#endif

#if defined(RAJA_ENABLE_CUDA)

CUDA_TEST(Atomic, basic_CUDA_AtomicFunction)
{
  testAtomicFunctionPol<RAJA::cuda_exec<256>, RAJA::atomic::auto_atomic>();
  testAtomicFunctionPol<RAJA::cuda_exec<256>, RAJA::atomic::cuda_atomic>();
}

CUDA_TEST(Atomic, basic_CUDA_AtomicRef)
{
  testAtomicRefPol<RAJA::cuda_exec<256>, RAJA::atomic::auto_atomic>();
  testAtomicRefPol<RAJA::cuda_exec<256>, RAJA::atomic::cuda_atomic>();
}

CUDA_TEST(Atomic, basic_CUDA_AtomicView)
{
  testAtomicViewPol<RAJA::cuda_exec<256>, RAJA::atomic::auto_atomic>();
  testAtomicViewPol<RAJA::cuda_exec<256>, RAJA::atomic::cuda_atomic>();
}


CUDA_TEST(Atomic, basic_CUDA_Logical)
{
  testAtomicLogicalPol<RAJA::cuda_exec<256>, RAJA::atomic::auto_atomic>();
  testAtomicLogicalPol<RAJA::cuda_exec<256>, RAJA::atomic::cuda_atomic>();
}


#endif

TEST(Atomic, basic_seq_AtomicFunction)
{
  testAtomicFunctionPol<RAJA::seq_exec, RAJA::atomic::auto_atomic>();
  testAtomicFunctionPol<RAJA::seq_exec, RAJA::atomic::seq_atomic>();
  testAtomicFunctionPol<RAJA::seq_exec, RAJA::atomic::builtin_atomic>();
}

TEST(Atomic, basic_seq_AtomicRef)
{
  testAtomicRefPol<RAJA::seq_exec, RAJA::atomic::auto_atomic>();
  testAtomicRefPol<RAJA::seq_exec, RAJA::atomic::seq_atomic>();
  testAtomicRefPol<RAJA::seq_exec, RAJA::atomic::builtin_atomic>();
}

TEST(Atomic, basic_seq_AtomicView)
{
  testAtomicViewPol<RAJA::seq_exec, RAJA::atomic::auto_atomic>();
  testAtomicViewPol<RAJA::seq_exec, RAJA::atomic::seq_atomic>();
  testAtomicViewPol<RAJA::seq_exec, RAJA::atomic::builtin_atomic>();
}


TEST(Atomic, basic_seq_Logical)
{
  testAtomicLogicalPol<RAJA::seq_exec, RAJA::atomic::auto_atomic>();
  testAtomicLogicalPol<RAJA::seq_exec, RAJA::atomic::seq_atomic>();
  testAtomicLogicalPol<RAJA::seq_exec, RAJA::atomic::builtin_atomic>();
}
