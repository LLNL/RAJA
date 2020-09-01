//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

///
/// Source file containing tests for RAJA CPU reduction operations.
///

#include "gtest/gtest.h"

#include <iostream>
#include "RAJA/RAJA.hpp"

#include <tuple>

template <typename TUPLE>
class ReductionCorrectnessTestTargetOMP : public ::testing::Test
{
protected:
  virtual void SetUp()
  {
    array_length = 102;

    array = RAJA::allocate_aligned_type<double>(RAJA::DATA_ALIGN,
                                                array_length * sizeof(double));

    for (int i = 1; i < array_length - 1; ++i) {
      array[i] = (RAJA::Real_type)i;
    }
    array[0] = 0.0;
    array[array_length - 1] = -1.0;
#pragma omp target enter data map(to : array[:array_length])

    sum = 0.0;
    min = array_length * 2;
    max = 0.0;
    minloc = -1;
    maxloc = -1;

    for (int i = 0; i < array_length; ++i) {
      RAJA::Real_type val = array[i];

      sum += val;

      if (val > max) {
        max = val;
        maxloc = i;
      }

      if (val < min) {
        min = val;
        minloc = i;
      }
    }
  }

  virtual void TearDown()
  {
#pragma omp target exit data map(release : array[:array_length])
    free(array);
  }

  RAJA::Real_ptr array;

  RAJA::Real_type max;
  RAJA::Real_type min;
  RAJA::Real_type sum;
  RAJA::Real_type maxloc;
  RAJA::Real_type minloc;

  RAJA::Index_type array_length;
};
TYPED_TEST_SUITE_P(ReductionCorrectnessTestTargetOMP);

TYPED_TEST_P(ReductionCorrectnessTestTargetOMP, ReduceMinLocGenericIndex)
{
  using ExecPolicy = typename std::tuple_element<0, TypeParam>::type;
  using ReducePolicy = typename std::tuple_element<1, TypeParam>::type;

  struct Index {
     RAJA::Index_type idx;
     Index() : idx(-1) {}
     Index(RAJA::Index_type idx) : idx(idx) {}
  };

  RAJA::ReduceMinLoc<ReducePolicy, double, Index> minloc_reducer(1024.0, Index(0));

  auto array = this->array;
  // TODO: remove this when compilers (clang-coral and IBM XLC) are no longer
  // broken for lambda capture
#pragma omp target data use_device_ptr(array)
  RAJA::forall<ExecPolicy>(RAJA::RangeSegment(0, this->array_length),
                           [=](int i) { minloc_reducer.minloc(array[i], Index(i)); });

  double raja_min = (double)minloc_reducer.get();
  Index raja_loc = minloc_reducer.getLoc();

  ASSERT_FLOAT_EQ(this->min, raja_min);
  ASSERT_EQ(this->minloc, raja_loc.idx);
}

TYPED_TEST_P(ReductionCorrectnessTestTargetOMP, ReduceMaxLocGenericIndex)
{
  using ExecPolicy = typename std::tuple_element<0, TypeParam>::type;
  using ReducePolicy = typename std::tuple_element<1, TypeParam>::type;

  struct Index {
     RAJA::Index_type idx;
     Index() : idx(-1) {}
     Index(RAJA::Index_type idx) : idx(idx) {}
  };

  RAJA::ReduceMaxLoc<ReducePolicy, double, Index> maxloc_reducer(0.0, Index());

  auto array = this->array;
  // TODO: remove this when compilers (clang-coral and IBM XLC) are no longer
  // broken for lambda capture
#pragma omp target data use_device_ptr(array)
  RAJA::forall<ExecPolicy>(RAJA::RangeSegment(0, this->array_length),
                           [=](int i) { maxloc_reducer.maxloc(array[i], Index(i)); });

  double raja_max = (double)maxloc_reducer.get();
  Index raja_loc = maxloc_reducer.getLoc();

  ASSERT_FLOAT_EQ(this->max, raja_max);
  ASSERT_EQ(this->maxloc, raja_loc.idx);
}

REGISTER_TYPED_TEST_SUITE_P(ReductionCorrectnessTestTargetOMP,
                            ReduceMinLocGenericIndex,
                            ReduceMaxLocGenericIndex);
using types =
    ::testing::Types<std::tuple<RAJA::omp_target_parallel_for_exec<16>,
                                RAJA::omp_target_reduce>,
                     std::tuple<RAJA::omp_target_parallel_for_exec<64>,
                                RAJA::omp_target_reduce>,
                     std::tuple<RAJA::omp_target_parallel_for_exec<256>,
                                RAJA::omp_target_reduce>>;

INSTANTIATE_TYPED_TEST_SUITE_P(Reduce, ReductionCorrectnessTestTargetOMP, types);

template <typename TUPLE>
class NestedReductionCorrectnessTestTargetOMP : public ::testing::Test
{
protected:
  virtual void SetUp()
  {
    x_size = 256;
    y_size = 256;
    z_size = 256;

    array = RAJA::allocate_aligned_type<double>(RAJA::DATA_ALIGN,
                                                x_size * y_size * z_size *
                                                    sizeof(double));

    const double val = 4.0 / (x_size * y_size * z_size);

    for (int i = 0; i < (x_size * y_size * z_size); ++i) {
      array[i] = (RAJA::Real_type)val;
    }

#pragma omp target enter data map(to : array[:x_size * y_size * z_size])

    sum = 4.0;
  }

  virtual void TearDown()
  {
#pragma omp target exit data map(release : array[:x_size * y_size * z_size])
    free(array);
  }

  RAJA::Real_ptr array;

  RAJA::Real_type sum;

  RAJA::Index_type x_size;
  RAJA::Index_type y_size;
  RAJA::Index_type z_size;
};


