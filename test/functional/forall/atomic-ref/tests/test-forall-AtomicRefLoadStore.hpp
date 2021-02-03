//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-21, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

///
/// Source file containing basic functional tests for load/store atomic operations using forall
///

#ifndef __TEST_FORALL_ATOMICREF_LOADSTORE_HPP__
#define __TEST_FORALL_ATOMICREF_LOADSTORE_HPP__

template < typename T, typename AtomicPolicy, typename IdxType >
struct LoadOtherOp : all_op {
  LoadOtherOp(T* count, RAJA::TypedRangeSegment<IdxType> seg)
    : other(count), min((T)seg.size()), max(min),
    final_min(min), final_max(min)
  { count[0] = min; }
  RAJA_HOST_DEVICE
    T operator()(IdxType RAJA_UNUSED_ARG(i)) const
    { return other.load(); }
  RAJA::AtomicRef<T, AtomicPolicy> other;
  T min, max, final_min, final_max;
};

template < typename T, typename AtomicPolicy, typename IdxType >
struct OperatorTOtherOp : all_op {
  OperatorTOtherOp(T* count, RAJA::TypedRangeSegment<IdxType> RAJA_UNUSED_ARG(seg))
    : other(count), min(T(0)), max(min),
    final_min(min), final_max(min)
  { count[0] = min; }
  RAJA_HOST_DEVICE
    T operator()(IdxType RAJA_UNUSED_ARG(i)) const
    { return other; }
  RAJA::AtomicRef<T, AtomicPolicy> other;
  T min, max, final_min, final_max;
};

template < typename T, typename AtomicPolicy, typename IdxType >
struct StoreOtherOp : all_op {
  StoreOtherOp(T* count, RAJA::TypedRangeSegment<IdxType> seg)
    : other(count), min((T)0), max((T)seg.size() - (T)1),
    final_min(min), final_max(max)
  { count[0] = (T)seg.size(); }
  RAJA_HOST_DEVICE
    T operator()(IdxType i) const
    { other.store((T)i); return (T)i; }
  RAJA::AtomicRef<T, AtomicPolicy> other;
  T min, max, final_min, final_max;
};

template < typename T, typename AtomicPolicy, typename IdxType >
struct AssignOtherOp : all_op {
  AssignOtherOp(T* count, RAJA::TypedRangeSegment<IdxType> seg)
    : other(count), min(T(0)), max((T)seg.size() - (T)1),
    final_min(min), final_max(max)
  { count[0] = (T)seg.size(); }
  RAJA_HOST_DEVICE
    T operator()(IdxType i) const
    { return (other = (T)i); }
  RAJA::AtomicRef<T, AtomicPolicy> other;
  T min, max, final_min, final_max;
};

template  < typename ExecPolicy,
            typename AtomicPolicy,
            typename IdxType,
            typename T,
            template <typename, typename, typename> class OtherOp>
void
testAtomicRefLoadStoreOp(RAJA::TypedRangeSegment<IdxType> seg, T* count, T* list)
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
void ForallAtomicRefLoadStoreTestImpl( IdxType N )
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

  testAtomicRefLoadStoreOp<ExecPolicy, AtomicPolicy, IdxType, T, 
                       LoadOtherOp     >(seg, count, list);
  testAtomicRefLoadStoreOp<ExecPolicy, AtomicPolicy, IdxType, T, 
                       OperatorTOtherOp>(seg, count, list);
  testAtomicRefLoadStoreOp<ExecPolicy, AtomicPolicy, IdxType, T, 
                       StoreOtherOp    >(seg, count, list);
  testAtomicRefLoadStoreOp<ExecPolicy, AtomicPolicy, IdxType, T, 
                       AssignOtherOp   >(seg, count, list);

  count_res.deallocate( count );
  list_res.deallocate( list );
}


TYPED_TEST_SUITE_P(ForallAtomicRefLoadStoreTest);
template <typename T>
class ForallAtomicRefLoadStoreTest : public ::testing::Test
{
};

TYPED_TEST_P(ForallAtomicRefLoadStoreTest, AtomicRefLoadStoreForall)
{
  using AExec   = typename camp::at<TypeParam, camp::num<0>>::type;
  using APol    = typename camp::at<TypeParam, camp::num<1>>::type;
  using ResType = typename camp::at<TypeParam, camp::num<2>>::type;
  using IdxType = typename camp::at<TypeParam, camp::num<3>>::type;
  using DType   = typename camp::at<TypeParam, camp::num<4>>::type;

  ForallAtomicRefLoadStoreTestImpl<AExec, APol, ResType, IdxType, DType>( 10000 );
}

REGISTER_TYPED_TEST_SUITE_P(ForallAtomicRefLoadStoreTest,
                            AtomicRefLoadStoreForall);

#endif  //__TEST_FORALL_ATOMICREF_LOADSTORE_HPP__
