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
#include "RAJA/internal/MemUtils_CPU.hpp"

#include <tuple>

#include <math.h>

template <typename T>
class ReductionConstructorTest : public ::testing::Test
{
};

template <typename T>
class ReductionConstructorTest2 : public ::testing::Test
{
};

TYPED_TEST_SUITE_P(ReductionConstructorTest);
TYPED_TEST_SUITE_P(ReductionConstructorTest2);

TYPED_TEST_P(ReductionConstructorTest, ReductionConstructor)
{
  using ReducePolicy = typename std::tuple_element<0, TypeParam>::type;
  using NumericType = typename std::tuple_element<1, TypeParam>::type;

  RAJA::ReduceSum<ReducePolicy, NumericType> reduce_sum(0.0);
  RAJA::ReduceMin<ReducePolicy, NumericType> reduce_min(0.0);
  RAJA::ReduceMax<ReducePolicy, NumericType> reduce_max(0.0);
  RAJA::ReduceMinLoc<ReducePolicy, NumericType> reduce_minloc(0.0, 1);
  RAJA::ReduceMaxLoc<ReducePolicy, NumericType> reduce_maxloc(0.0, 1);

  RAJA::tuple<RAJA::Index_type, RAJA::Index_type> LocTup(1, 1);
  RAJA::ReduceMinLoc<ReducePolicy, NumericType, RAJA::tuple<RAJA::Index_type, RAJA::Index_type>> reduce_minloctup(0.0, LocTup);
  RAJA::ReduceMaxLoc<ReducePolicy, NumericType, RAJA::tuple<RAJA::Index_type, RAJA::Index_type>> reduce_maxloctup(0.0, LocTup);

  ASSERT_EQ((NumericType)reduce_sum.get(), (NumericType)(0.0));
  ASSERT_EQ((NumericType)reduce_min.get(), (NumericType)(0.0));
  ASSERT_EQ((NumericType)reduce_max.get(), (NumericType)(0.0));
  ASSERT_EQ((RAJA::Index_type)reduce_minloc.getLoc(), (RAJA::Index_type)1);
  ASSERT_EQ((NumericType)reduce_minloc.get(), (NumericType)(0.0));
  ASSERT_EQ((NumericType)reduce_maxloc.get(), (NumericType)(0.0));
  ASSERT_EQ((RAJA::Index_type)reduce_maxloc.getLoc(), (RAJA::Index_type)1);

  ASSERT_EQ((NumericType)reduce_minloctup.get(), (NumericType)(0.0));
  ASSERT_EQ((NumericType)reduce_maxloctup.get(), (NumericType)(0.0));
  ASSERT_EQ((RAJA::Index_type)(RAJA::get<0>(reduce_minloctup.getLoc())), (RAJA::Index_type)1);
  ASSERT_EQ((RAJA::Index_type)(RAJA::get<1>(reduce_minloctup.getLoc())), (RAJA::Index_type)1);
  ASSERT_EQ((RAJA::Index_type)(RAJA::get<0>(reduce_maxloctup.getLoc())), (RAJA::Index_type)1);
  ASSERT_EQ((RAJA::Index_type)(RAJA::get<1>(reduce_maxloctup.getLoc())), (RAJA::Index_type)1);
}

TYPED_TEST_P(ReductionConstructorTest, ReductionConstructor2)
{
  using ReducePolicy = typename std::tuple_element<0, TypeParam>::type;
  using NumericType = typename std::tuple_element<1, TypeParam>::type;

  RAJA::ReduceSum<ReducePolicy, NumericType> reduce_sum;
  RAJA::ReduceMin<ReducePolicy, NumericType> reduce_min;
  RAJA::ReduceMax<ReducePolicy, NumericType> reduce_max;
  RAJA::ReduceMinLoc<ReducePolicy, NumericType> reduce_minloc;
  RAJA::ReduceMaxLoc<ReducePolicy, NumericType> reduce_maxloc;

  RAJA::ReduceMinLoc<ReducePolicy, NumericType, RAJA::tuple<RAJA::Index_type, RAJA::Index_type>> reduce_minloctup;
  RAJA::ReduceMaxLoc<ReducePolicy, NumericType, RAJA::tuple<RAJA::Index_type, RAJA::Index_type>> reduce_maxloctup;

  ASSERT_EQ((NumericType)reduce_sum.get(), NumericType());
  ASSERT_EQ((NumericType)reduce_min.get(), NumericType());
  ASSERT_EQ((NumericType)reduce_max.get(), NumericType());
  ASSERT_EQ((RAJA::Index_type)reduce_minloc.getLoc(), RAJA::Index_type());
  ASSERT_EQ((NumericType)reduce_minloc.get(), NumericType());
  ASSERT_EQ((NumericType)reduce_maxloc.get(), NumericType());
  ASSERT_EQ((RAJA::Index_type)reduce_maxloc.getLoc(), RAJA::Index_type());

  ASSERT_EQ((NumericType)reduce_minloctup.get(), NumericType());
  ASSERT_EQ((NumericType)reduce_maxloctup.get(), NumericType());
  ASSERT_EQ((RAJA::Index_type)(RAJA::get<0>(reduce_minloctup.getLoc())), RAJA::Index_type());
  ASSERT_EQ((RAJA::Index_type)(RAJA::get<1>(reduce_minloctup.getLoc())), RAJA::Index_type());
  ASSERT_EQ((RAJA::Index_type)(RAJA::get<0>(reduce_maxloctup.getLoc())), RAJA::Index_type());
  ASSERT_EQ((RAJA::Index_type)(RAJA::get<1>(reduce_maxloctup.getLoc())), RAJA::Index_type());
}

REGISTER_TYPED_TEST_SUITE_P(ReductionConstructorTest,
                           ReductionConstructor,
                           ReductionConstructor2);

using constructor_types =
    ::testing::Types<std::tuple<RAJA::seq_reduce, int>,
                     std::tuple<RAJA::seq_reduce, float>,
                     std::tuple<RAJA::seq_reduce, double>
#if defined(RAJA_ENABLE_TBB)
                     ,
                     std::tuple<RAJA::tbb_reduce, int>,
                     std::tuple<RAJA::tbb_reduce, float>,
                     std::tuple<RAJA::tbb_reduce, double>
#endif
#if defined(RAJA_ENABLE_OPENMP)
                     ,
                     std::tuple<RAJA::omp_reduce, int>,
                     std::tuple<RAJA::omp_reduce, float>,
                     std::tuple<RAJA::omp_reduce, double>,
                     std::tuple<RAJA::omp_reduce_ordered, int>,
                     std::tuple<RAJA::omp_reduce_ordered, float>,
                     std::tuple<RAJA::omp_reduce_ordered, double>
#endif
                     >;

INSTANTIATE_TYPED_TEST_SUITE_P(ReduceBasicTests,
                              ReductionConstructorTest,
                              constructor_types);

template <typename TUPLE>
class ReductionCorrectnessTest : public ::testing::Test
{
protected:
  virtual void SetUp()
  {
    array_length = 102;

    array = RAJA::allocate_aligned_type<RAJA::Real_type>(RAJA::DATA_ALIGN,
                                                         array_length * sizeof(RAJA::Real_type));
    for (int i = 1; i < array_length - 1; ++i) {
      array[i] = (RAJA::Real_type)i;
    }
    array[0] = 0.0;
    array[array_length - 1] = -1.0;

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

  virtual void TearDown() { RAJA::free_aligned(array); }

  RAJA::Real_ptr array;

  RAJA::Real_type max;
  RAJA::Real_type min;
  RAJA::Real_type sum;
  RAJA::Real_type maxloc;
  RAJA::Real_type minloc;

  RAJA::Index_type array_length;
};
TYPED_TEST_SUITE_P(ReductionCorrectnessTest);

template <typename TUPLE>
class ReductionGenericLocTest : public ::testing::Test
{
protected:
  virtual void SetUp()
  {
    // 2 dimensional, 10x10
    array_length = 100;
    xdim = 10;
    ydim = 10;

    array = RAJA::allocate_aligned_type<double *>(RAJA::DATA_ALIGN,
                                                  ydim * sizeof(double *));
    data = RAJA::allocate_aligned_type<double>(RAJA::DATA_ALIGN,
                                               array_length * sizeof(double));

    // set rows to point to data
    for ( int ii = 0; ii < ydim; ++ii ) {
      array[ii] = data + ii * ydim;
    }

    // setting data values
    int count = 0;
    for ( int ii = 0; ii < ydim; ++ii ) {
      for ( int jj = 0; jj < xdim; ++jj ) {
        array[ii][jj] = (RAJA::Real_type)(count++);
      }
    }

    array[ydim-1][xdim-1] = -1.0;

    sum = 0.0;
    min = array_length * 2;
    max = 0.0;
    minlocx = -1;
    minlocy = -1;
    maxlocx = -1;
    maxlocy = -1;

    for (int y = 0; y < ydim; ++y) {
      for ( int x = 0; x < xdim; ++x ) {
        RAJA::Real_type val = array[y][x];

        sum += val;

        if (val > max) {
          max = val;
          maxlocx = x;
          maxlocy = y;
        }

        if (val < min) {
          min = val;
          minlocx = x;
          minlocy = y;
        }
      }
    }
  }

  virtual void TearDown() {
    // NOTE: clang prefers cast to void * and fails compilation otherwise.
    // gcc prefers RAJA::Real_ptr * (with no compilation warnings).
    RAJA::free_aligned((void *)array);

    RAJA::free_aligned(data);
  }

  RAJA::Real_ptr * array;
  RAJA::Real_ptr data;

  RAJA::Real_type max;
  RAJA::Real_type min;
  RAJA::Real_type sum;
  RAJA::Real_type maxlocx;
  RAJA::Real_type maxlocy;
  RAJA::Real_type minlocx;
  RAJA::Real_type minlocy;

  RAJA::Index_type array_length;
  RAJA::Index_type xdim;
  RAJA::Index_type ydim;
};
TYPED_TEST_SUITE_P(ReductionGenericLocTest);

TYPED_TEST_P(ReductionCorrectnessTest, ReduceSum)
{
  using ExecPolicy = typename std::tuple_element<0, TypeParam>::type;
  using ReducePolicy = typename std::tuple_element<1, TypeParam>::type;
  // using NumericType = typename std::tuple_element<2, TypeParam>::type;

  RAJA::ReduceSum<ReducePolicy, double> sum_reducer(0.0);

  RAJA::forall<ExecPolicy>(RAJA::RangeSegment(0, this->array_length),
                           [=](int i) { sum_reducer += this->array[i]; });

  double raja_sum = (double)sum_reducer.get();

  ASSERT_FLOAT_EQ(this->sum, raja_sum);
}

TYPED_TEST_P(ReductionCorrectnessTest, ReduceSum2)
{
  using ExecPolicy = typename std::tuple_element<0, TypeParam>::type;
  using ReducePolicy = typename std::tuple_element<1, TypeParam>::type;
  // using NumericType = typename std::tuple_element<2, TypeParam>::type;

  RAJA::ReduceSum<ReducePolicy, double> sum_reducer;

  sum_reducer.reset(5.0);
  sum_reducer.reset(0.0);  // reset the value

  RAJA::forall<ExecPolicy>(RAJA::RangeSegment(0, this->array_length),
                           [=](int i) { sum_reducer += this->array[i]; });

  double raja_sum = (double)sum_reducer.get();

  ASSERT_FLOAT_EQ(this->sum, raja_sum);
}

TYPED_TEST_P(ReductionCorrectnessTest, ReduceMin)
{
  using ExecPolicy = typename std::tuple_element<0, TypeParam>::type;
  using ReducePolicy = typename std::tuple_element<1, TypeParam>::type;
  // using NumericType = typename std::tuple_element<2, TypeParam>::type;

  RAJA::ReduceMin<ReducePolicy, double> min_reducer(1024.0);

  RAJA::forall<ExecPolicy>(RAJA::RangeSegment(0, this->array_length),
                           [=](int i) { min_reducer.min(this->array[i]); });

  double raja_min = (double)min_reducer.get();

  ASSERT_FLOAT_EQ(this->min, raja_min);
}


TYPED_TEST_P(ReductionCorrectnessTest, ReduceMin2)
{
  using ExecPolicy = typename std::tuple_element<0, TypeParam>::type;
  using ReducePolicy = typename std::tuple_element<1, TypeParam>::type;
  // using NumericType = typename std::tuple_element<2, TypeParam>::type;

  RAJA::ReduceMin<ReducePolicy, double> min_reducer;

  min_reducer.reset(1024.0);
  RAJA::forall<ExecPolicy>(RAJA::RangeSegment(0, this->array_length),
                           [=](int i) { min_reducer.min(this->array[i]); });

  double raja_min = (double)min_reducer.get();

  ASSERT_FLOAT_EQ(this->min, raja_min);
}

TYPED_TEST_P(ReductionCorrectnessTest, ReduceMax)
{
  using ExecPolicy = typename std::tuple_element<0, TypeParam>::type;
  using ReducePolicy = typename std::tuple_element<1, TypeParam>::type;
  // using NumericType = typename std::tuple_element<2, TypeParam>::type;

  RAJA::ReduceMax<ReducePolicy, double> max_reducer(0.0);

  RAJA::forall<ExecPolicy>(RAJA::RangeSegment(0, this->array_length),
                           [=](int i) { max_reducer.max(this->array[i]); });

  double raja_max = (double)max_reducer.get();

  ASSERT_FLOAT_EQ(this->max, raja_max);
}

TYPED_TEST_P(ReductionCorrectnessTest, ReduceMax2)
{
  using ExecPolicy = typename std::tuple_element<0, TypeParam>::type;
  using ReducePolicy = typename std::tuple_element<1, TypeParam>::type;
  // using NumericType = typename std::tuple_element<2, TypeParam>::type;

  RAJA::ReduceMax<ReducePolicy, double> max_reducer;

  max_reducer.reset(5.0);
  max_reducer.reset(0.0);

  RAJA::forall<ExecPolicy>(RAJA::RangeSegment(0, this->array_length),
                           [=](int i) { max_reducer.max(this->array[i]); });

  double raja_max = (double)max_reducer.get();

  ASSERT_FLOAT_EQ(this->max, raja_max);
}

TYPED_TEST_P(ReductionCorrectnessTest, ReduceMinLoc)
{
  using ExecPolicy = typename std::tuple_element<0, TypeParam>::type;
  using ReducePolicy = typename std::tuple_element<1, TypeParam>::type;
  // using NumericType = typename std::tuple_element<2, TypeParam>::type;

  RAJA::ReduceMinLoc<ReducePolicy, double> minloc_reducer(1024.0, 0);

  RAJA::forall<ExecPolicy>(RAJA::RangeSegment(0, this->array_length),
                           [=](int i) {
                             minloc_reducer.minloc(this->array[i], i);
                           });

  RAJA::Index_type raja_loc = minloc_reducer.getLoc();
  double raja_min = (double)minloc_reducer.get();

  ASSERT_FLOAT_EQ(this->min, raja_min);
  ASSERT_EQ(this->minloc, raja_loc);
}

TYPED_TEST_P(ReductionCorrectnessTest, ReduceMinLocGenericIndex)
{
  using ExecPolicy = typename std::tuple_element<0, TypeParam>::type;
  using ReducePolicy = typename std::tuple_element<1, TypeParam>::type;
  // using NumericType = typename std::tuple_element<2, TypeParam>::type;

  struct Index {
     RAJA::Index_type idx;
     constexpr Index() : idx(-1) {}
     constexpr Index(RAJA::Index_type idx) : idx(idx) {}
  };

  RAJA::ReduceMinLoc<ReducePolicy, double, Index> minloc_reducer(1024.0, Index(0));

  RAJA::forall<ExecPolicy>(RAJA::RangeSegment(0, this->array_length),
                           [=](int i) {
                             minloc_reducer.minloc(this->array[i], Index(i));
                           });

  Index raja_loc = minloc_reducer.getLoc();
  double raja_min = (double)minloc_reducer.get();

  ASSERT_FLOAT_EQ(this->min, raja_min);
  ASSERT_EQ(this->minloc, raja_loc.idx);
}

TYPED_TEST_P(ReductionGenericLocTest, ReduceMinLoc2DIndex)
{
  using ExecPolicy = typename std::tuple_element<0, TypeParam>::type;
  using ReducePolicy = typename std::tuple_element<1, TypeParam>::type;

  struct Index2D {
    RAJA::Index_type idx, idy;
    RAJA::Index_type idarray;  // actual array index
    constexpr Index2D() : idx(-1), idy(-1), idarray(-1) {}

    // 2 dimensional array, 10 elements per row
    Index2D(RAJA::Index_type idx, RAJA::Index_type idy) : idx(idx), idy(idy)
    {
      idarray = idx % 10 + idy * 10;
    }
    Index2D(RAJA::Index_type idarray) : idarray(idarray)
    {
      idx = idarray % 10;
      idy = floor( idarray / 10 );
    }
  };

  RAJA::ReduceMinLoc<ReducePolicy, double, Index2D> minloc_reducer(1024.0, Index2D(0, 0));

  RAJA::forall<ExecPolicy>(RAJA::RangeSegment(0, this->array_length),
                           [=](int i) {
                             minloc_reducer.minloc(this->data[i], Index2D(i));
                           });

  Index2D raja_loc = minloc_reducer.getLoc();
  double raja_min = (double)minloc_reducer.get();

  ASSERT_FLOAT_EQ(this->min, raja_min);
  ASSERT_EQ(this->minlocx, raja_loc.idx);
  ASSERT_EQ(this->minlocy, raja_loc.idy);
}

TYPED_TEST_P(ReductionGenericLocTest, ReduceMinLoc2DIndexKernel)
{
  using ExecPolicy =
    RAJA::KernelPolicy<
      RAJA::statement::For<1, RAJA::loop_exec,  // row
        RAJA::statement::For<0, RAJA::loop_exec,  // col
          RAJA::statement::Lambda<0>
        >
      >
    >;

  using ReducePolicy = RAJA::seq_reduce;

  RAJA::RangeSegment colrange(0, 10);
  RAJA::RangeSegment rowrange(0, 10);

  struct Index2D {
     RAJA::Index_type idx, idy;
     constexpr Index2D() : idx(-1), idy(-1) {}
     constexpr Index2D(RAJA::Index_type idx, RAJA::Index_type idy) : idx(idx), idy(idy) {}
  };

  RAJA::ReduceMinLoc<ReducePolicy, double, Index2D> minloc_reducer(1024.0, Index2D(0, 0));

  RAJA::kernel<ExecPolicy>(RAJA::make_tuple(colrange, rowrange),
                           [=](int c, int r) {
                             minloc_reducer.minloc(this->array[r][c], Index2D(c, r));
                           });

  Index2D raja_loc = minloc_reducer.getLoc();
  double raja_min = (double)minloc_reducer.get();

  ASSERT_FLOAT_EQ(this->min, raja_min);
  ASSERT_EQ(this->minlocx, raja_loc.idx);
  ASSERT_EQ(this->minlocy, raja_loc.idy);
}

TYPED_TEST_P(ReductionGenericLocTest, ReduceMinLoc2DIndexViewKernel)
{
  using ExecPolicy =
    RAJA::KernelPolicy<
      RAJA::statement::For<1, RAJA::loop_exec,  // row
        RAJA::statement::For<0, RAJA::loop_exec,  // col
          RAJA::statement::Lambda<0>
        >
      >
    >;

  using ReducePolicy = RAJA::seq_reduce;

  RAJA::RangeSegment colrange(0, 10);
  RAJA::RangeSegment rowrange(0, 10);

  RAJA::View<double, RAJA::Layout<2>> ArrView(this->data, 10, 10);

  struct Index2D {
     RAJA::Index_type idx, idy;
     constexpr Index2D() : idx(-1), idy(-1) {}
     constexpr Index2D(RAJA::Index_type idx, RAJA::Index_type idy) : idx(idx), idy(idy) {}
  };

  RAJA::ReduceMinLoc<ReducePolicy, double, Index2D> minloc_reducer(1024.0, Index2D(0, 0));

  RAJA::kernel<ExecPolicy>(RAJA::make_tuple(colrange, rowrange),
                           [=](int c, int r) {
                             minloc_reducer.minloc(ArrView(r, c), Index2D(c, r));
                           });

  Index2D raja_loc = minloc_reducer.getLoc();
  double raja_min = (double)minloc_reducer.get();

  ASSERT_FLOAT_EQ(this->min, raja_min);
  ASSERT_EQ(this->minlocx, raja_loc.idx);
  ASSERT_EQ(this->minlocy, raja_loc.idy);
}

TYPED_TEST_P(ReductionGenericLocTest, ReduceMinLoc2DIndexTupleViewKernel)
{
  using ExecPolicy =
    RAJA::KernelPolicy<
      RAJA::statement::For<1, RAJA::loop_exec,  // row
        RAJA::statement::For<0, RAJA::loop_exec,  // col
          RAJA::statement::Lambda<0>
        >
      >
    >;

  using ReducePolicy = RAJA::seq_reduce;

  RAJA::RangeSegment colrange(0, 10);
  RAJA::RangeSegment rowrange(0, 10);

  RAJA::View<double, RAJA::Layout<2>> ArrView(this->data, 10, 10);

  RAJA::tuple<int, int> LocTup(0, 0);

  RAJA::ReduceMinLoc<ReducePolicy, double, RAJA::tuple<int, int>> minloc_reducer(1024.0, LocTup);

  RAJA::kernel<ExecPolicy>(RAJA::make_tuple(colrange, rowrange),
                           [=](int c, int r) {
                             minloc_reducer.minloc(ArrView(r, c), RAJA::make_tuple(c, r));
                           });

  RAJA::tuple<int, int> raja_loc = minloc_reducer.getLoc();
  double raja_min = (double)minloc_reducer.get();

  ASSERT_FLOAT_EQ(this->min, raja_min);
  ASSERT_EQ(this->minlocx, RAJA::get<0>(raja_loc));
  ASSERT_EQ(this->minlocy, RAJA::get<1>(raja_loc));
}

TYPED_TEST_P(ReductionGenericLocTest, ReduceMaxLoc2DIndex)
{
  using ExecPolicy = typename std::tuple_element<0, TypeParam>::type;
  using ReducePolicy = typename std::tuple_element<1, TypeParam>::type;

  struct Index2D {
    RAJA::Index_type idx, idy;
    RAJA::Index_type idarray;  // actual array index
    constexpr Index2D() : idx(-1), idy(-1), idarray(-1) {}

    // 2 dimensional array, 10 elements per row
    Index2D(RAJA::Index_type idx, RAJA::Index_type idy) : idx(idx), idy(idy)
    {
      idarray = idx % 10 + idy * 10;
    }
    Index2D(RAJA::Index_type idarray) : idarray(idarray)
    {
      idx = idarray % 10;
      idy = floor( idarray / 10 );
    }
  };

  RAJA::ReduceMaxLoc<ReducePolicy, double, Index2D> maxloc_reducer(-1024.0, Index2D(0, 0));

  RAJA::forall<ExecPolicy>(RAJA::RangeSegment(0, this->array_length),
                           [=](int i) {
                             maxloc_reducer.maxloc(this->data[i], Index2D(i));
                           });

  Index2D raja_loc = maxloc_reducer.getLoc();
  double raja_max = (double)maxloc_reducer.get();

  ASSERT_FLOAT_EQ(this->max, raja_max);
  ASSERT_EQ(this->maxlocx, raja_loc.idx);
  ASSERT_EQ(this->maxlocy, raja_loc.idy);
}

TYPED_TEST_P(ReductionGenericLocTest, ReduceMaxLoc2DIndexKernel)
{
  using ExecPolicy =
    RAJA::KernelPolicy<
      RAJA::statement::For<1, RAJA::loop_exec,  // row
        RAJA::statement::For<0, RAJA::loop_exec,  // col
          RAJA::statement::Lambda<0>
        >
      >
    >;

  using ReducePolicy = RAJA::seq_reduce;

  RAJA::RangeSegment colrange(0, 10);
  RAJA::RangeSegment rowrange(0, 10);

  struct Index2D {
     RAJA::Index_type idx, idy;
     constexpr Index2D() : idx(-1), idy(-1) {}
     constexpr Index2D(RAJA::Index_type idx, RAJA::Index_type idy) : idx(idx), idy(idy) {}
  };

  RAJA::ReduceMaxLoc<ReducePolicy, double, Index2D> maxloc_reducer(-1024.0, Index2D(0, 0));

  RAJA::kernel<ExecPolicy>(RAJA::make_tuple(colrange, rowrange),
                           [=](int c, int r) {
                             maxloc_reducer.maxloc(this->array[r][c], Index2D(c, r));
                           });

  Index2D raja_loc = maxloc_reducer.getLoc();
  double raja_max = (double)maxloc_reducer.get();

  ASSERT_FLOAT_EQ(this->max, raja_max);
  ASSERT_EQ(this->maxlocx, raja_loc.idx);
  ASSERT_EQ(this->maxlocy, raja_loc.idy);
}

TYPED_TEST_P(ReductionGenericLocTest, ReduceMaxLoc2DIndexViewKernel)
{
  using ExecPolicy =
    RAJA::KernelPolicy<
      RAJA::statement::For<1, RAJA::loop_exec,  // row
        RAJA::statement::For<0, RAJA::loop_exec,  // col
          RAJA::statement::Lambda<0>
        >
      >
    >;

  using ReducePolicy = RAJA::seq_reduce;

  RAJA::RangeSegment colrange(0, 10);
  RAJA::RangeSegment rowrange(0, 10);

  RAJA::View<double, RAJA::Layout<2>> ArrView(this->data, 10, 10);

  struct Index2D {
     RAJA::Index_type idx, idy;
     constexpr Index2D() : idx(-1), idy(-1) {}
     constexpr Index2D(RAJA::Index_type idx, RAJA::Index_type idy) : idx(idx), idy(idy) {}
  };

  RAJA::ReduceMaxLoc<ReducePolicy, double, Index2D> maxloc_reducer(-1024.0, Index2D(0, 0));

  RAJA::kernel<ExecPolicy>(RAJA::make_tuple(colrange, rowrange),
                           [=](int c, int r) {
                             maxloc_reducer.maxloc(ArrView(r, c), Index2D(c, r));
                           });

  Index2D raja_loc = maxloc_reducer.getLoc();
  double raja_max = (double)maxloc_reducer.get();

  ASSERT_FLOAT_EQ(this->max, raja_max);
  ASSERT_EQ(this->maxlocx, raja_loc.idx);
  ASSERT_EQ(this->maxlocy, raja_loc.idy);
}

TYPED_TEST_P(ReductionGenericLocTest, ReduceMaxLoc2DIndexTupleViewKernel)
{
  using ExecPolicy =
    RAJA::KernelPolicy<
      RAJA::statement::For<1, RAJA::loop_exec,  // row
        RAJA::statement::For<0, RAJA::loop_exec,  // col
          RAJA::statement::Lambda<0>
        >
      >
    >;

  using ReducePolicy = RAJA::seq_reduce;

  RAJA::RangeSegment colrange(0, 10);
  RAJA::RangeSegment rowrange(0, 10);

  RAJA::View<double, RAJA::Layout<2>> ArrView(this->data, 10, 10);

  RAJA::tuple<int, int> LocTup(0, 0);

  RAJA::ReduceMaxLoc<ReducePolicy, double, RAJA::tuple<int, int>> maxloc_reducer(-1024.0, LocTup);

  RAJA::kernel<ExecPolicy>(RAJA::make_tuple(colrange, rowrange),
                           [=](int c, int r) {
                             maxloc_reducer.maxloc(ArrView(r, c), RAJA::make_tuple(c, r));
                           });

  RAJA::tuple<int, int> raja_loc = maxloc_reducer.getLoc();
  double raja_max = (double)maxloc_reducer.get();

  ASSERT_FLOAT_EQ(this->max, raja_max);
  ASSERT_EQ(this->maxlocx, RAJA::get<0>(raja_loc));
  ASSERT_EQ(this->maxlocy, RAJA::get<1>(raja_loc));
}

TYPED_TEST_P(ReductionCorrectnessTest, ReduceMinLoc2)
{
  using ExecPolicy =
      RAJA::seq_exec;  // typename std::tuple_element<0, TypeParam>::type;
  using ReducePolicy =
      RAJA::seq_reduce;  // typename std::tuple_element<1, TypeParam>::type;
  // using NumericType = typename std::tuple_element<2, TypeParam>::type;


  RAJA::ReduceMinLoc<ReducePolicy, double> minloc_reducer;

  minloc_reducer.reset(1024.0, 0);

  RAJA::forall<ExecPolicy>(RAJA::RangeSegment(0, this->array_length),
                           [=](int i) {
                             minloc_reducer.minloc(this->array[i], i);
                           });

  RAJA::Index_type raja_loc = minloc_reducer.getLoc();
  double raja_min = (double)minloc_reducer.get();

  ASSERT_FLOAT_EQ(this->min, raja_min);
  ASSERT_EQ(this->minloc, raja_loc);
}

TYPED_TEST_P(ReductionCorrectnessTest, ReduceMinLocGenericIndex2)
{
  using ExecPolicy =
      RAJA::seq_exec;  // typename std::tuple_element<0, TypeParam>::type;
  using ReducePolicy =
      RAJA::seq_reduce;  // typename std::tuple_element<1, TypeParam>::type;
  // using NumericType = typename std::tuple_element<2, TypeParam>::type;

  struct Index {
     RAJA::Index_type idx;
     constexpr Index() : idx(-1) {}
     constexpr Index(RAJA::Index_type idx) : idx(idx) {}
  };

  RAJA::ReduceMinLoc<ReducePolicy, double, Index> minloc_reducer;

  minloc_reducer.reset({1024.0, Index(0)});

  RAJA::forall<ExecPolicy>(RAJA::RangeSegment(0, this->array_length),
                           [=](int i) {
                             minloc_reducer.minloc(this->array[i], Index(i));
                           });

  Index raja_loc = minloc_reducer.getLoc();
  double raja_min = (double)minloc_reducer.get();

  ASSERT_FLOAT_EQ(this->min, raja_min);
  ASSERT_EQ(this->minloc, raja_loc.idx);
}

TYPED_TEST_P(ReductionCorrectnessTest, ReduceMaxLoc)
{
  using ExecPolicy = typename std::tuple_element<0, TypeParam>::type;
  using ReducePolicy = typename std::tuple_element<1, TypeParam>::type;
  // using NumericType = typename std::tuple_element<2, TypeParam>::type;

  RAJA::ReduceMaxLoc<ReducePolicy, double> maxloc_reducer(0.0, -1);

  RAJA::forall<ExecPolicy>(RAJA::RangeSegment(0, this->array_length),
                           [=](int i) {
                             maxloc_reducer.maxloc(this->array[i], i);
                           });

  RAJA::Index_type raja_loc = maxloc_reducer.getLoc();
  double raja_max = (double)maxloc_reducer.get();

  ASSERT_FLOAT_EQ(this->max, raja_max);
  ASSERT_EQ(this->maxloc, raja_loc);
}

TYPED_TEST_P(ReductionCorrectnessTest, ReduceMaxLocGenericIndex)
{
  using ExecPolicy = typename std::tuple_element<0, TypeParam>::type;
  using ReducePolicy = typename std::tuple_element<1, TypeParam>::type;
  // using NumericType = typename std::tuple_element<2, TypeParam>::type;

  struct Index {
     RAJA::Index_type idx;
     constexpr Index() : idx(-1) {}
     constexpr Index(RAJA::Index_type idx) : idx(idx) {}
  };

  RAJA::ReduceMaxLoc<ReducePolicy, double, Index> maxloc_reducer(0.0, Index());

  RAJA::forall<ExecPolicy>(RAJA::RangeSegment(0, this->array_length),
                           [=](int i) {
                             maxloc_reducer.maxloc(this->array[i], Index(i));
                           });

  Index raja_loc = maxloc_reducer.getLoc();
  double raja_max = (double)maxloc_reducer.get();

  ASSERT_FLOAT_EQ(this->max, raja_max);
  ASSERT_EQ(this->maxloc, raja_loc.idx);
}

TYPED_TEST_P(ReductionCorrectnessTest, ReduceMaxLoc2)
{
  using ExecPolicy = typename std::tuple_element<0, TypeParam>::type;
  using ReducePolicy = typename std::tuple_element<1, TypeParam>::type;
  // using NumericType = typename std::tuple_element<2, TypeParam>::type;

  RAJA::ReduceMaxLoc<ReducePolicy, double> maxloc_reducer;

  maxloc_reducer.reset(0.0, -1);

  RAJA::forall<ExecPolicy>(RAJA::RangeSegment(0, this->array_length),
                           [=](int i) {
                             maxloc_reducer.maxloc(this->array[i], i);
                           });

  RAJA::Index_type raja_loc = maxloc_reducer.getLoc();
  double raja_max = (double)maxloc_reducer.get();

  ASSERT_FLOAT_EQ(this->max, raja_max);
  ASSERT_EQ(this->maxloc, raja_loc);
}

TYPED_TEST_P(ReductionCorrectnessTest, ReduceMaxLocGenericIndex2)
{
  using ExecPolicy = typename std::tuple_element<0, TypeParam>::type;
  using ReducePolicy = typename std::tuple_element<1, TypeParam>::type;
  // using NumericType = typename std::tuple_element<2, TypeParam>::type;

  struct Index {
     RAJA::Index_type idx;
     constexpr Index() : idx(-1) {}
     constexpr Index(RAJA::Index_type idx) : idx(idx) {}
  };

  RAJA::ReduceMaxLoc<ReducePolicy, double, Index> maxloc_reducer;

  maxloc_reducer.reset({0.0, Index()});

  RAJA::forall<ExecPolicy>(RAJA::RangeSegment(0, this->array_length),
                           [=](int i) {
                             maxloc_reducer.maxloc(this->array[i], Index(i));
                           });

  Index raja_loc = maxloc_reducer.getLoc();
  double raja_max = (double)maxloc_reducer.get();

  ASSERT_FLOAT_EQ(this->max, raja_max);
  ASSERT_EQ(this->maxloc, raja_loc.idx);
}

REGISTER_TYPED_TEST_SUITE_P(ReductionCorrectnessTest,
                           ReduceSum,
                           ReduceSum2,
                           ReduceMin,
                           ReduceMin2,
                           ReduceMax,
                           ReduceMax2,
                           ReduceMinLoc,
                           ReduceMinLocGenericIndex,
                           ReduceMinLoc2,
                           ReduceMinLocGenericIndex2,
                           ReduceMaxLoc,
                           ReduceMaxLocGenericIndex,
                           ReduceMaxLoc2,
                           ReduceMaxLocGenericIndex2);

REGISTER_TYPED_TEST_SUITE_P(ReductionGenericLocTest,
                           ReduceMinLoc2DIndex,
                           ReduceMinLoc2DIndexKernel,
                           ReduceMinLoc2DIndexViewKernel,
                           ReduceMinLoc2DIndexTupleViewKernel,
                           ReduceMaxLoc2DIndex,
                           ReduceMaxLoc2DIndexKernel,
                           ReduceMaxLoc2DIndexViewKernel,
                           ReduceMaxLoc2DIndexTupleViewKernel);

using types = ::testing::Types<
    std::tuple<RAJA::seq_exec, RAJA::seq_reduce>,
    std::tuple<RAJA::loop_exec, RAJA::seq_reduce>
#if defined(RAJA_ENABLE_OPENMP)
    ,
    std::tuple<RAJA::omp_parallel_for_exec, RAJA::omp_reduce>,
    std::tuple<RAJA::omp_parallel_for_exec, RAJA::omp_reduce_ordered>
#endif
#if defined(RAJA_ENABLE_TBB)
    ,
    std::tuple<RAJA::tbb_for_exec, RAJA::tbb_reduce>
#endif
    >;

INSTANTIATE_TYPED_TEST_SUITE_P(Reduce, ReductionCorrectnessTest, types);
INSTANTIATE_TYPED_TEST_SUITE_P(Reduce, ReductionGenericLocTest, types);



