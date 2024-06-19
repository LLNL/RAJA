//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-24, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef __TEST_FORALL_BASIC_REDUCESUM_HPP__
#define __TEST_FORALL_BASIC_REDUCESUM_HPP__

#include <cstdlib>
#include <ctime>
#include <numeric>
#include <vector>
#include <random>
#include <type_traits>

template <typename EXEC_POLICY, typename REDUCE_POLICY, typename ABSTRACTION,
          typename DATA_TYPE, typename IDX_TYPE,
          typename SEG_TYPE, typename Container,
          typename RandomGenerator,
          std::enable_if_t<!ABSTRACTION::template supports<DATA_TYPE>()>* = nullptr>
void ForallMultiReduceBasicTestImpl(const SEG_TYPE&,
                                    const Container&,
                                    const std::vector<IDX_TYPE>&,
                                    camp::resources::Resource,
                                    RandomGenerator&)
{ }
///
template <typename EXEC_POLICY, typename REDUCE_POLICY, typename ABSTRACTION,
          typename DATA_TYPE, typename IDX_TYPE,
          typename SEG_TYPE, typename Container,
          typename RandomGenerator,
          std::enable_if_t<ABSTRACTION::template supports<DATA_TYPE>()>* = nullptr>
void ForallMultiReduceBasicTestImpl(const SEG_TYPE& seg,
                                    const Container& multi_init,
                                    const std::vector<IDX_TYPE>& seg_idx,
                                    camp::resources::Resource working_res,
                                    RandomGenerator& rngen)
{
  using MULTIREDUCER = typename ABSTRACTION::template multi_reducer<REDUCE_POLICY, DATA_TYPE>;

  const IDX_TYPE data_len = seg_idx[seg_idx.size() - 1] + 1;
  const IDX_TYPE idx_len = static_cast<IDX_TYPE>( seg_idx.size() );

  DATA_TYPE* working_array;
  DATA_TYPE* check_array;
  DATA_TYPE* test_array;

  allocateForallTestData(data_len,
                         working_res,
                         &working_array,
                         &check_array,
                         &test_array);

  IDX_TYPE* working_bins;
  IDX_TYPE* check_bins;
  IDX_TYPE* test_bins;

  allocateForallTestData(data_len,
                         working_res,
                         &working_bins,
                         &check_bins,
                         &test_bins);


  const int modval = 100;
  const size_t num_bins = multi_init.size();


  // use ints to initialize array here to avoid floating point precision issues
  std::uniform_int_distribution<int> array_int_distribution(0, modval-1);
  std::uniform_int_distribution<IDX_TYPE> bin_distribution(0, num_bins-1);

  for (IDX_TYPE i = 0; i < data_len; ++i) {
    test_array[i] = DATA_TYPE(array_int_distribution(rngen));
    test_bins[i] = bin_distribution(rngen);
  }
  working_res.memcpy(working_array, test_array, sizeof(DATA_TYPE) * data_len);
  working_res.memcpy(working_bins, test_bins, sizeof(IDX_TYPE) * data_len);



  MULTIREDUCER red(num_bins);
  MULTIREDUCER red2(multi_init);

  {
    std::vector<DATA_TYPE> ref_vals(num_bins, ABSTRACTION::identity(red));

    for (IDX_TYPE i = 0; i < idx_len; ++i) {
      IDX_TYPE idx = seg_idx[i];
      ref_vals[test_bins[idx]] = ABSTRACTION::combine(ref_vals[test_bins[idx]], test_array[idx]);
    }

    RAJA::forall<EXEC_POLICY>(seg, [=] RAJA_HOST_DEVICE(IDX_TYPE idx) {
      ABSTRACTION::reduce(red[working_bins[idx]],  working_array[idx]);
      ABSTRACTION::reduce(red2[working_bins[idx]], working_array[idx]);
    });

    size_t bin = 0;
    for (auto init_val : multi_init) {
      ASSERT_EQ(DATA_TYPE(red[bin].get()), ref_vals[bin]);
      ASSERT_EQ(red2.get(bin), ABSTRACTION::combine(ref_vals[bin], init_val));
      ++bin;
    }
  }


  red.reset();

  {
    std::vector<DATA_TYPE> ref_vals(num_bins, ABSTRACTION::identity(red));

    const int nloops = 2;
    for (int j = 0; j < nloops; ++j) {

      for (IDX_TYPE i = 0; i < idx_len; ++i) {
        IDX_TYPE idx = seg_idx[i];
        ref_vals[test_bins[idx]] = ABSTRACTION::combine(ref_vals[test_bins[idx]], test_array[idx]);
      }

      RAJA::forall<EXEC_POLICY>(seg, [=] RAJA_HOST_DEVICE(IDX_TYPE idx) {
        ABSTRACTION::reduce(red[working_bins[idx]], working_array[idx]);
      });
    }

    for (size_t bin = 0; bin < num_bins; ++bin) {
      ASSERT_EQ(static_cast<DATA_TYPE>(red[bin].get()), ref_vals[bin]);
    }
  }

  if (ABSTRACTION::consistent(red)) {

    if /* constexpr */ (std::is_floating_point<DATA_TYPE>::value) {

      // use floating point values to accentuate floating point precision issues
      std::conditional_t<!std::is_floating_point<DATA_TYPE>::value,
          std::uniform_int_distribution<DATA_TYPE>,
          std::uniform_real_distribution<DATA_TYPE>> array_flt_distribution(0, modval-1);

      for (IDX_TYPE i = 0; i < data_len; ++i) {
        test_array[i] = DATA_TYPE(array_flt_distribution(rngen));
      }
      working_res.memcpy(working_array, test_array, sizeof(DATA_TYPE) * data_len);
    }


    std::vector<DATA_TYPE> ref_vals;

    const int nloops = 10;
    for (int j = 0; j < nloops; ++j) {
      red.reset();

      RAJA::forall<EXEC_POLICY>(seg, [=] RAJA_HOST_DEVICE(IDX_TYPE idx) {
        ABSTRACTION::reduce(red[working_bins[idx]], working_array[idx]);
      });

      if (ref_vals.empty()) {
        for (size_t bin = 0; bin < num_bins; ++bin) {
          ref_vals.emplace_back(red.get(bin));
        }
      } else {
        for (size_t bin = 0; bin < num_bins; ++bin) {
          ASSERT_EQ(red.get(bin), ref_vals[bin]);
        }
      }
    }
  }


  deallocateForallTestData(working_res,
                           working_bins,
                           check_bins,
                           test_bins);
  deallocateForallTestData(working_res,
                           working_array,
                           check_array,
                           test_array);
}


TYPED_TEST_SUITE_P(ForallMultiReduceBasicTest);
template <typename T>
class ForallMultiReduceBasicTest : public ::testing::Test
{
};

TYPED_TEST_P(ForallMultiReduceBasicTest, MultiReduceBasicForall)
{
  using IDX_TYPE      = typename camp::at<TypeParam, camp::num<0>>::type;
  using DATA_TYPE     = typename camp::at<TypeParam, camp::num<1>>::type;
  using WORKING_RES   = typename camp::at<TypeParam, camp::num<2>>::type;
  using EXEC_POLICY   = typename camp::at<TypeParam, camp::num<3>>::type;
  using REDUCE_POLICY = typename camp::at<TypeParam, camp::num<4>>::type;
  using ABSTRACTION   = typename camp::at<TypeParam, camp::num<5>>::type;

  // for setting random values in arrays
  auto random_seed = std::random_device{}();
  std::mt19937 rngen(random_seed);

  camp::resources::Resource working_res{WORKING_RES::get_default()};

  std::vector<IDX_TYPE> seg_idx;

  std::vector<DATA_TYPE> c3(3, DATA_TYPE(2));

// Range segment tests
  RAJA::TypedRangeSegment<IDX_TYPE> r1( 0, 28 );
  RAJA::getIndices(seg_idx, r1);
  ForallMultiReduceBasicTestImpl<EXEC_POLICY, REDUCE_POLICY, ABSTRACTION, DATA_TYPE>(
                                 r1, c3, seg_idx, working_res, rngen);

  seg_idx.clear();
  RAJA::TypedRangeSegment<IDX_TYPE> r2( 3, 642 );
  RAJA::getIndices(seg_idx, r2);
  ForallMultiReduceBasicTestImpl<EXEC_POLICY, REDUCE_POLICY, ABSTRACTION, DATA_TYPE>(
                                 r2, c3, seg_idx, working_res, rngen);

  seg_idx.clear();
  RAJA::TypedRangeSegment<IDX_TYPE> r3( 0, 2057 );
  RAJA::getIndices(seg_idx, r3);
  ForallMultiReduceBasicTestImpl<EXEC_POLICY, REDUCE_POLICY, ABSTRACTION, DATA_TYPE>(
                                 r3, c3, seg_idx, working_res, rngen);

// Range-stride segment tests
  seg_idx.clear();
  RAJA::TypedRangeStrideSegment<IDX_TYPE> r4( 0, 188, 2 );
  RAJA::getIndices(seg_idx, r4);
  ForallMultiReduceBasicTestImpl<EXEC_POLICY, REDUCE_POLICY, ABSTRACTION, DATA_TYPE>(
                                 r4, c3, seg_idx, working_res, rngen);

  seg_idx.clear();
  RAJA::TypedRangeStrideSegment<IDX_TYPE> r5( 3, 1029, 3 );
  RAJA::getIndices(seg_idx, r5);
  ForallMultiReduceBasicTestImpl<EXEC_POLICY, REDUCE_POLICY, ABSTRACTION, DATA_TYPE>(
                                 r5, c3, seg_idx, working_res, rngen);

// List segment tests
  seg_idx.clear(); 
  IDX_TYPE last = 10567;
  std::uniform_int_distribution<IDX_TYPE> dist(0, last-1);
  for (IDX_TYPE i = 0; i < last; ++i) {
    IDX_TYPE randval = dist(rngen);
    if ( i < randval ) {
      seg_idx.push_back(i);
    }
  }
  RAJA::TypedListSegment<IDX_TYPE> l1( &seg_idx[0], seg_idx.size(), 
                                       working_res );
  ForallMultiReduceBasicTestImpl<EXEC_POLICY, REDUCE_POLICY, ABSTRACTION, DATA_TYPE>(
                                 l1, c3, seg_idx, working_res, rngen);
}

REGISTER_TYPED_TEST_SUITE_P(ForallMultiReduceBasicTest,
                            MultiReduceBasicForall);

#endif  // __TEST_FORALL_BASIC_REDUCESUM_HPP__
