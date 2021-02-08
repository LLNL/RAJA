//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-21, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

///
/// Source file containing basic functional tests for addition arithmetic atomic operations using forall
///

#ifndef __TEST_FORALL_ATOMICREF_ADD_HPP__
#define __TEST_FORALL_ATOMICREF_ADD_HPP__

template < typename T, typename AtomicPolicy, typename IdxType >
struct PreIncCountOp {
  PreIncCountOp(T* count, RAJA::TypedRangeSegment<IdxType> seg)
    : counter(count), min((T)0), max((T)seg.size()-(T)1), final((T)seg.size())
  { count[0] = (T)0; }
  RAJA_HOST_DEVICE
    T operator()(IdxType RAJA_UNUSED_ARG(i)) const {
      return (++counter) - (T)1;
    }
  RAJA::AtomicRef<T, AtomicPolicy> counter;
  T min, max, final;
};

template < typename T, typename AtomicPolicy, typename IdxType >
struct PostIncCountOp {
  PostIncCountOp(T* count, RAJA::TypedRangeSegment<IdxType> seg)
    : counter(count), min((T)0), max((T)seg.size()-(T)1), final((T)seg.size())
  { count[0] = (T)0; }
  RAJA_HOST_DEVICE
    T operator()(IdxType RAJA_UNUSED_ARG(i)) const {
      return (counter++);
    }
  RAJA::AtomicRef<T, AtomicPolicy> counter;
  T min, max, final;
};

template < typename T, typename AtomicPolicy, typename IdxType >
struct AddEqCountOp {
  AddEqCountOp(T* count, RAJA::TypedRangeSegment<IdxType> seg)
    : counter(count), min((T)0), max((T)seg.size()-(T)1), final((T)seg.size())
  { count[0] = (T)0; }
  RAJA_HOST_DEVICE
    T operator()(IdxType RAJA_UNUSED_ARG(i)) const {
      return (counter += (T)1) - (T)1;
    }
  RAJA::AtomicRef<T, AtomicPolicy> counter;
  T min, max, final;
};

template < typename T, typename AtomicPolicy, typename IdxType >
struct FetchAddCountOp {
  FetchAddCountOp(T* count, RAJA::TypedRangeSegment<IdxType> seg)
    : counter(count), min((T)0), max((T)seg.size()-(T)1), final((T)seg.size())
  { count[0] = (T)0; }
  RAJA_HOST_DEVICE
    T operator()(IdxType RAJA_UNUSED_ARG(i)) const {
      return counter.fetch_add((T)1);
    }
  RAJA::AtomicRef<T, AtomicPolicy> counter;
  T min, max, final;
};

template <typename ExecPolicy,
         typename AtomicPolicy,
         typename IdxType,
         typename T,
         template <typename, typename, typename> class CountOp>
void testAtomicRefAdd(RAJA::TypedRangeSegment<IdxType> seg,
    T* count, T* list, bool* hit)
{
  CountOp<T, AtomicPolicy, IdxType> countop(count, seg);
  RAJA::forall<ExecPolicy>(seg, [=] RAJA_HOST_DEVICE(IdxType i) {
      list[i] = countop.max + (T)1;
      hit[i] = false;
      });
  RAJA::forall<ExecPolicy>(seg, [=] RAJA_HOST_DEVICE(IdxType i) {
      T val = countop(i);
      list[i] = val;
      hit[(IdxType)val] = true;
      });
#if defined(RAJA_ENABLE_CUDA)
  cudaErrchk(cudaDeviceSynchronize());
#endif
#if defined(RAJA_ENABLE_HIP)
  hipErrchk(hipDeviceSynchronize());
#endif

  EXPECT_EQ(countop.final, count[0]);
  for (IdxType i = 0; i < seg.size(); i++) {
    EXPECT_LE(countop.min, list[i]);
    EXPECT_GE(countop.max, list[i]);
    EXPECT_TRUE(hit[i]);
  }
}


template <typename ExecPolicy,
          typename AtomicPolicy,
          typename WORKINGRES,
          typename IdxType,
          typename T>
void ForallAtomicRefAddTestImpl( IdxType N )
{
  RAJA::TypedRangeSegment<IdxType> seg(0, N);

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

  testAtomicRefAdd<ExecPolicy, AtomicPolicy, IdxType, T, 
                     PreIncCountOp  >(seg, count, list, hit);
  testAtomicRefAdd<ExecPolicy, AtomicPolicy, IdxType, T, 
                     PostIncCountOp >(seg, count, list, hit);
  testAtomicRefAdd<ExecPolicy, AtomicPolicy, IdxType, T, 
                     AddEqCountOp   >(seg, count, list, hit);
  testAtomicRefAdd<ExecPolicy, AtomicPolicy, IdxType, T, 
                     FetchAddCountOp>(seg, count, list, hit);

  count_res.deallocate( count );
  list_res.deallocate( list );
  hit_res.deallocate( hit );
}


TYPED_TEST_SUITE_P(ForallAtomicRefAddTest);
template <typename T>
class ForallAtomicRefAddTest : public ::testing::Test
{
};

TYPED_TEST_P(ForallAtomicRefAddTest, AtomicRefAddForall)
{
  using AExec   = typename camp::at<TypeParam, camp::num<0>>::type;
  using APol    = typename camp::at<TypeParam, camp::num<1>>::type;
  using ResType = typename camp::at<TypeParam, camp::num<2>>::type;
  using IdxType = typename camp::at<TypeParam, camp::num<3>>::type;
  using DType   = typename camp::at<TypeParam, camp::num<4>>::type;

  ForallAtomicRefAddTestImpl<AExec, APol, ResType, IdxType, DType>( 10000 );
}

REGISTER_TYPED_TEST_SUITE_P(ForallAtomicRefAddTest,
                            AtomicRefAddForall);

#endif  //__TEST_FORALL_ATOMICREF_ADD_HPP__
