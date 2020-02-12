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
/// Source file containing basic functional tests for arithmetic atomic operations using forall
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
struct PreIncCountOp {
  PreIncCountOp(T* count, RAJA::RangeSegment seg)
    : counter(count), min((T)0), max((T)seg.size()-(T)1), final((T)seg.size())
  { count[0] = (T)0; }
  RAJA_HOST_DEVICE
    T operator()(RAJA::Index_type RAJA_UNUSED_ARG(i)) const {
      return (++counter) - (T)1;
    }
  RAJA::AtomicRef<T, AtomicPolicy> counter;
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
  RAJA::AtomicRef<T, AtomicPolicy> counter;
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
  RAJA::AtomicRef<T, AtomicPolicy> counter;
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
  RAJA::AtomicRef<T, AtomicPolicy> counter;
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
  RAJA::AtomicRef<T, AtomicPolicy> counter;
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
  RAJA::AtomicRef<T, AtomicPolicy> counter;
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
  RAJA::AtomicRef<T, AtomicPolicy> counter;
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
  RAJA::AtomicRef<T, AtomicPolicy> counter;
  T min, max, final;
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
  cudaErrchk(cudaDeviceSynchronize());
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

  testAtomicRefCount<ExecPolicy, AtomicPolicy, T, PreIncCountOp  >(seg, count, list, hit);
  testAtomicRefCount<ExecPolicy, AtomicPolicy, T, PostIncCountOp >(seg, count, list, hit);
  testAtomicRefCount<ExecPolicy, AtomicPolicy, T, AddEqCountOp   >(seg, count, list, hit);
  testAtomicRefCount<ExecPolicy, AtomicPolicy, T, FetchAddCountOp>(seg, count, list, hit);

  testAtomicRefCount<ExecPolicy, AtomicPolicy, T, PreDecCountOp  >(seg, count, list, hit);
  testAtomicRefCount<ExecPolicy, AtomicPolicy, T, PostDecCountOp >(seg, count, list, hit);
  testAtomicRefCount<ExecPolicy, AtomicPolicy, T, SubEqCountOp   >(seg, count, list, hit);
  testAtomicRefCount<ExecPolicy, AtomicPolicy, T, FetchSubCountOp>(seg, count, list, hit);

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

  testAtomicRefCount<ExecPolicy, AtomicPolicy, T, PreIncCountOp  >(seg, count, list, hit);
  testAtomicRefCount<ExecPolicy, AtomicPolicy, T, PostIncCountOp >(seg, count, list, hit);
  testAtomicRefCount<ExecPolicy, AtomicPolicy, T, AddEqCountOp   >(seg, count, list, hit);
  testAtomicRefCount<ExecPolicy, AtomicPolicy, T, FetchAddCountOp>(seg, count, list, hit);

  testAtomicRefCount<ExecPolicy, AtomicPolicy, T, PreDecCountOp  >(seg, count, list, hit);
  testAtomicRefCount<ExecPolicy, AtomicPolicy, T, PostDecCountOp >(seg, count, list, hit);
  testAtomicRefCount<ExecPolicy, AtomicPolicy, T, SubEqCountOp   >(seg, count, list, hit);
  testAtomicRefCount<ExecPolicy, AtomicPolicy, T, FetchSubCountOp>(seg, count, list, hit);

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


