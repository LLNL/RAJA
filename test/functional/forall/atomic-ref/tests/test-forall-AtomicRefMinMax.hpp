//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-21, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

///
/// Source file containing basic functional tests for min/max atomic operations using forall
///

#ifndef __TEST_FORALL_ATOMICREF_MINMAX_HPP__
#define __TEST_FORALL_ATOMICREF_MINMAX_HPP__

template < typename T, typename AtomicPolicy, typename IdxType >
struct MaxEqOtherOp : all_op {
  MaxEqOtherOp(T* count, RAJA::TypedRangeSegment<IdxType> seg)
    : other(count), min(T(0)), max((T)seg.size() - (T)1),
    final_min(max), final_max(max)
  { count[0] = (T)0; }
  RAJA_HOST_DEVICE
    T operator()(IdxType i) const
    { return other.max((T)i); }
  RAJA::AtomicRef<T, AtomicPolicy> other;
  T min, max, final_min, final_max;
};

template < typename T, typename AtomicPolicy, typename IdxType >
struct FetchMaxOtherOp : all_op {
  FetchMaxOtherOp(T* count, RAJA::TypedRangeSegment<IdxType> seg)
    : other(count), min(T(0)), max((T)seg.size() - (T)1),
    final_min(max), final_max(max)
  { count[0] = (T)0; }
  RAJA_HOST_DEVICE
    T operator()(IdxType i) const
    { return other.fetch_max((T)i); }
  RAJA::AtomicRef<T, AtomicPolicy> other;
  T min, max, final_min, final_max;
};

template < typename T, typename AtomicPolicy, typename IdxType >
struct MinEqOtherOp : all_op {
  MinEqOtherOp(T* count, RAJA::TypedRangeSegment<IdxType> seg)
    : other(count), min(T(0)), max((T)seg.size() - (T)1),
    final_min(min), final_max(min)
  { count[0] = (T)seg.size(); }
  RAJA_HOST_DEVICE
    T operator()(IdxType i) const
    { return other.min((T)i); }
  RAJA::AtomicRef<T, AtomicPolicy> other;
  T min, max, final_min, final_max;
};

template < typename T, typename AtomicPolicy, typename IdxType >
struct FetchMinOtherOp : all_op {
  FetchMinOtherOp(T* count, RAJA::TypedRangeSegment<IdxType> seg)
    : other(count), min(T(0)), max((T)seg.size()),
    final_min(min), final_max(min)
  { count[0] = (T)seg.size(); }
  RAJA_HOST_DEVICE
    T operator()(IdxType i) const
    { return other.fetch_min((T)i); }
  RAJA::AtomicRef<T, AtomicPolicy> other;
  T min, max, final_min, final_max;
};

template  < typename ExecPolicy,
            typename AtomicPolicy,
            typename IdxType,
            typename T,
            template <typename, typename, typename> class OtherOp>
void
testAtomicRefMinMaxOp(RAJA::TypedRangeSegment<IdxType> seg, T* count, T* list)
{
  OtherOp<T, AtomicPolicy, IdxType> otherop(count, seg);
  RAJA::forall<ExecPolicy>(seg, [=] RAJA_HOST_DEVICE(IdxType i) {
      list[i] = otherop.max + (T)1;
  });
  RAJA::forall<ExecPolicy>(seg, [=] RAJA_HOST_DEVICE(IdxType i) {
      T val = otherop(i);
      list[i] = val;
  });
#if defined(RAJA_ENABLE_CUDA)
  cudaErrchk(cudaDeviceSynchronize());
#endif
#if defined(RAJA_ENABLE_HIP)
  hipErrchk(hipDeviceSynchronize());
#endif
  EXPECT_LE(otherop.final_min, count[0]);
  EXPECT_GE(otherop.final_max, count[0]);
  for (IdxType i = 0; i < seg.size(); i++) {
    EXPECT_LE(otherop.min, list[i]);
    EXPECT_GE(otherop.max, list[i]);
  }
}


template <typename ExecPolicy,
          typename AtomicPolicy,
          typename WORKINGRES,
          typename IdxType,
          typename T>
void ForallAtomicRefMinMaxTestImpl( IdxType N )
{
  RAJA::TypedRangeSegment<IdxType> seg(0, N);

  camp::resources::Resource count_res{WORKINGRES()};
  camp::resources::Resource list_res{WORKINGRES()};

  T * count   = count_res.allocate<T>(1);
  T * list    = list_res.allocate<T>(N);

#if defined(RAJA_ENABLE_CUDA)
  cudaErrchk(cudaDeviceSynchronize());
#endif

#if defined(RAJA_ENABLE_HIP)
  hipErrchk(hipDeviceSynchronize());
#endif

  testAtomicRefMinMaxOp<ExecPolicy, AtomicPolicy, IdxType, T, 
                       MaxEqOtherOp   >(seg, count, list);
  testAtomicRefMinMaxOp<ExecPolicy, AtomicPolicy, IdxType, T, 
                       FetchMaxOtherOp>(seg, count, list);
  testAtomicRefMinMaxOp<ExecPolicy, AtomicPolicy, IdxType, T, 
                       MinEqOtherOp   >(seg, count, list);
  testAtomicRefMinMaxOp<ExecPolicy, AtomicPolicy, IdxType, T, 
                       FetchMinOtherOp>(seg, count, list);

  count_res.deallocate( count );
  list_res.deallocate( list );
}


TYPED_TEST_SUITE_P(ForallAtomicRefMinMaxTest);
template <typename T>
class ForallAtomicRefMinMaxTest : public ::testing::Test
{
};

TYPED_TEST_P(ForallAtomicRefMinMaxTest, AtomicRefMinMaxForall)
{
  using AExec   = typename camp::at<TypeParam, camp::num<0>>::type;
  using APol    = typename camp::at<TypeParam, camp::num<1>>::type;
  using ResType = typename camp::at<TypeParam, camp::num<2>>::type;
  using IdxType = typename camp::at<TypeParam, camp::num<3>>::type;
  using DType   = typename camp::at<TypeParam, camp::num<4>>::type;

  ForallAtomicRefMinMaxTestImpl<AExec, APol, ResType, IdxType, DType>( 10000 );
}

REGISTER_TYPED_TEST_SUITE_P(ForallAtomicRefMinMaxTest,
                            AtomicRefMinMaxForall);

#endif  //__TEST_FORALL_ATOMICREF_MINMAX_HPP__
