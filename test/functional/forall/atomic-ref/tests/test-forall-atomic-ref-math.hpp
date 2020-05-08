//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

///
/// Source file containing basic functional tests for arithmetic atomic operations using forall
///

#ifndef __TEST_FORALL_ATOMIC_REF_MATH_HPP__
#define __TEST_FORALL_ATOMIC_REF_MATH_HPP__

#include <RAJA/RAJA.hpp>
#include "RAJA_gtest.hpp"

template < typename T, typename AtomicPolicy >
struct PreIncCountOp {
  PreIncCountOp(T* count, RAJA::TypedRangeSegment<RAJA::Index_type> seg)
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
  PostIncCountOp(T* count, RAJA::TypedRangeSegment<RAJA::Index_type> seg)
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
  AddEqCountOp(T* count, RAJA::TypedRangeSegment<RAJA::Index_type> seg)
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
  FetchAddCountOp(T* count, RAJA::TypedRangeSegment<RAJA::Index_type> seg)
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
  PreDecCountOp(T* count, RAJA::TypedRangeSegment<RAJA::Index_type> seg)
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
  PostDecCountOp(T* count, RAJA::TypedRangeSegment<RAJA::Index_type> seg)
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
  SubEqCountOp(T* count, RAJA::TypedRangeSegment<RAJA::Index_type> seg)
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
  FetchSubCountOp(T* count, RAJA::TypedRangeSegment<RAJA::Index_type> seg)
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
void testAtomicRefCount(RAJA::TypedRangeSegment<RAJA::Index_type> seg,
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
#if defined(RAJA_ENABLE_HIP)
  hipErrchk(hipDeviceSynchronize());
#endif

  EXPECT_EQ(countop.final, count[0]);
  for (RAJA::Index_type i = 0; i < seg.size(); i++) {
    EXPECT_LE(countop.min, list[i]);
    EXPECT_GE(countop.max, list[i]);
    EXPECT_TRUE(hit[i]);
  }
}


TYPED_TEST_SUITE_P(ForallAtomicRefMathFunctionalTest);

template <typename T>
class ForallAtomicRefMathFunctionalTest : public ::testing::Test
{
};

template <typename ExecPolicy,
          typename AtomicPolicy,
          typename WORKINGRES,
          typename T>
void testAtomicFunctionRefMath( RAJA::Index_type N )
{
  RAJA::TypedRangeSegment<RAJA::Index_type> seg(0, N);

  camp::resources::Resource count_res{WORKINGRES()};
  camp::resources::Resource list_res{WORKINGRES()};
  camp::resources::Resource hit_res{WORKINGRES()};

  T * count   = count_res.allocate<T>(1);
  T * list    = list_res.allocate<T>(N);
  bool * hit  = hit_res.allocate<bool>(N);

#if defined(RAJA_ENABLE_CUDA)
  cudaErrchk(cudaDeviceSynchronize());
#endif

#if defined(RAJA_ENABLE_HIP)
  hipErrchk(hipDeviceSynchronize());
#endif

  testAtomicRefCount<ExecPolicy, AtomicPolicy, T, PreIncCountOp  >(seg, count, list, hit);
  testAtomicRefCount<ExecPolicy, AtomicPolicy, T, PostIncCountOp >(seg, count, list, hit);
  testAtomicRefCount<ExecPolicy, AtomicPolicy, T, AddEqCountOp   >(seg, count, list, hit);
  testAtomicRefCount<ExecPolicy, AtomicPolicy, T, FetchAddCountOp>(seg, count, list, hit);

  testAtomicRefCount<ExecPolicy, AtomicPolicy, T, PreDecCountOp  >(seg, count, list, hit);
  testAtomicRefCount<ExecPolicy, AtomicPolicy, T, PostDecCountOp >(seg, count, list, hit);
  testAtomicRefCount<ExecPolicy, AtomicPolicy, T, SubEqCountOp   >(seg, count, list, hit);
  testAtomicRefCount<ExecPolicy, AtomicPolicy, T, FetchSubCountOp>(seg, count, list, hit);

  count_res.deallocate( count );
  list_res.deallocate( list );
  hit_res.deallocate( hit );
}

TYPED_TEST_P(ForallAtomicRefMathFunctionalTest, AtomicRefMathFunctionalForall)
{
  using AExec   = typename camp::at<TypeParam, camp::num<0>>::type;
  using APol    = typename camp::at<TypeParam, camp::num<1>>::type;
  using ResType = typename camp::at<TypeParam, camp::num<2>>::type;
  using DType   = typename camp::at<TypeParam, camp::num<3>>::type;
  testAtomicFunctionRefMath<AExec, APol, ResType, DType>( 10000 );
}

REGISTER_TYPED_TEST_SUITE_P( ForallAtomicRefMathFunctionalTest,
                             AtomicRefMathFunctionalForall
                           );

#endif  //__TEST_FORALL_ATOMIC_REF_MATH_HPP__
