//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

///
/// Source file containing tests for RAJA GPU tuple min-loc reductions.
///

#include <iostream>
#include <random>

#include "RAJA/RAJA.hpp"
#include "RAJA_gtest.hpp"

using namespace RAJA;

// array sizing
static constexpr RAJA::Index_type xdim = 1024;
static constexpr RAJA::Index_type ydim = 1024;
static constexpr RAJA::Index_type array_length = xdim * ydim;

// for setting random values in arrays
static std::random_device rd;
static std::mt19937 prng(rd());
static std::uniform_real_distribution<double> valuedist(-10, 10);
static std::uniform_real_distribution<double> indexdist(0, array_length - 1);

// GPU applier
template <typename T>
struct funcapplier;

template <typename NumType, typename Indexer>
struct funcapplier<ReduceMinLoc<hip_reduce, NumType, Indexer>>   // GPU minloc
{
  static NumType extremeval()
  {
    return (NumType)1024;
  }

  RAJA_HOST_DEVICE static void apply(ReduceMinLoc<hip_reduce, NumType, Indexer> const & r,
                                     NumType const & val,
                                     Indexer ii)
  {
    r.minloc(val, ii);
  }

  // helps gtest hip_typed_test_p determine minloc (min == false)
  static bool minormax()
  {
    return false;
  }
};

template <typename NumType, typename Indexer>
struct funcapplier<ReduceMinLoc<seq_reduce, NumType, Indexer>>    // CPU minloc
{
  static NumType extremeval()
  {
    return (NumType)1024;
  }

  static void apply(ReduceMinLoc<seq_reduce, NumType, Indexer> const & r,
                                     NumType const & val,
                                     Indexer ii)
  {
    r.minloc(val, ii);
  }
};

template <typename NumType, typename Indexer>
struct funcapplier<ReduceMaxLoc<hip_reduce, NumType, Indexer>>   // GPU maxloc
{
  static NumType extremeval()
  {
    return (NumType)(-1024);
  }

  RAJA_HOST_DEVICE static void apply(ReduceMaxLoc<hip_reduce, NumType, Indexer> const & r,
                                     NumType const & val,
                                     Indexer ii)
  {
    r.maxloc(val, ii);
  }

  // helps gtest hip_typed_test_p determine maxloc (max == true)
  static bool minormax()
  {
    return true;
  }
};

template <typename NumType, typename Indexer>
struct funcapplier<ReduceMaxLoc<seq_reduce, NumType, Indexer>>    // CPU maxloc
{
  static NumType extremeval()
  {
    return (NumType)(-1024);
  }

  static void apply(ReduceMaxLoc<seq_reduce, NumType, Indexer> const & r,
                                     NumType const & val,
                                     Indexer ii)
  {
    r.maxloc(val, ii);
  }
};

// base test
template <typename T>
struct HIPReduceLocTest : public ::testing::Test
{
  public:
  virtual void SetUp()
  {
    array = RAJA::allocate_aligned_type<double *>(RAJA::DATA_ALIGN,
                                                  ydim * sizeof(double *));
    d_array = RAJA::allocate_aligned_type<double *>(RAJA::DATA_ALIGN,
                                                  ydim * sizeof(double *));

    data = new double[array_length];
    hipMalloc( &d_data, sizeof(double) * array_length );

    dataview.set_data( data );
    d_dataview.set_data( d_data );

    // set rows to point to data
    for ( int ii = 0; ii < ydim; ++ii ) {
      array[ii] = data + ii * ydim;
      d_array[ii] = d_data + ii * ydim;
    }

    // setting data values
    int count = 0;
    for ( int ii = 0; ii < ydim; ++ii ) {
      for ( int jj = 0; jj < xdim; ++jj ) {
        array[ii][jj] = (RAJA::Real_type)(count++);
      }
    }

    array[ydim-1][xdim-1] = -1.0;

    for ( int ii = 0; ii < ydim; ++ii ) 
      hipMemcpy(d_array[ii], array[ii], sizeof(double) * xdim, hipMemcpyHostToDevice);

    reducelocs();
  }

  virtual void TearDown() {
    RAJA::free_aligned(array);
    RAJA::free_aligned(d_array);
    hipFree( d_data );
    delete[] data;
  }

  // make all values equal
  void equalize()
  {
    for ( int ii = 0; ii < array_length; ++ii )
    {
      data[ii] = 5.0;
    }
    hipMemcpy(d_data, data, sizeof(double) * array_length, hipMemcpyHostToDevice);
  }

  // change a random value in data
  void randompoke( int index, double val )
  {
    data[index] = val;
    hipMemcpy(d_data+index, &val, sizeof(double), hipMemcpyHostToDevice);    
  }

  // cpu reduction, for recalculating solutions
  void reducelocs()
  {
    min = array_length * 2;
    max = 0.0;
    minlocx = -1;
    minlocy = -1;
    maxlocx = -1;
    maxlocy = -1;

    for (int y = 0; y < ydim; ++y) {
      for ( int x = 0; x < xdim; ++x ) {
        RAJA::Real_type val = array[y][x];

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


  // accessors based on type
  RAJA::Real_type getminormax( bool actual )
  {
    if ( !actual )
    {
      return min;
    }
    else
    {
      return max;
    }
  }

  RAJA::Real_type getlocX( bool actual )
  {
    if ( !actual )
    {
      return minlocx;
    }
    else
    {
      return maxlocx;
    }
  }

  RAJA::Real_type getlocY( bool actual )
  {
    if ( !actual )
    {
      return minlocy;
    }
    else
    {
      return maxlocy;
    }
  }

  double ** array;
  double ** d_array;
  double * data;
  double * d_data;
  RAJA::View<double, RAJA::Layout<2>> dataview{nullptr, xdim, ydim}; 
  RAJA::View<double, RAJA::Layout<2>> d_dataview{nullptr, xdim, ydim}; 

  RAJA::Real_type max;
  RAJA::Real_type min;
  RAJA::Real_type maxlocx;
  RAJA::Real_type maxlocy;
  RAJA::Real_type minlocx;
  RAJA::Real_type minlocy;
};

TYPED_TEST_SUITE_P(HIPReduceLocTest);

GPU_TYPED_TEST_P(HIPReduceLocTest, ReduceLoc2DIndexTupleViewKernel)
{
  using applygpu = funcapplier<at_v<TypeParam, 0>>;
  using applycpu = funcapplier<at_v<TypeParam, 1>>;

  using ExecutionPol =
    RAJA::KernelPolicy<
      RAJA::statement::HipKernel<
      RAJA::statement::For<1, RAJA::hip_thread_x_loop,  // row
        RAJA::statement::For<0, RAJA::hip_thread_y_loop,  // col
          RAJA::statement::Lambda<0>
        >
      >
      > // end HipKernel
    >;

  RAJA::RangeSegment colrange(0, ydim);
  RAJA::RangeSegment rowrange(0, xdim);

  // FIRST TEST: original unequal values
  at_v<TypeParam, 0> minmaxloc_reducer(applygpu::extremeval(), RAJA::make_tuple(0, 0));

  RAJA::View<double, RAJA::Layout<2>> d_dataview(this->d_data, xdim, ydim);

  RAJA::kernel<ExecutionPol>(RAJA::make_tuple(colrange, rowrange),
                           [=] RAJA_DEVICE (int c, int r) {
                             auto ii = RAJA::make_tuple(c, r);
                             applygpu::apply(minmaxloc_reducer, d_dataview(r, c), RAJA::make_tuple(c, r));
                           });

  RAJA::tuple<int, int> raja_loc = minmaxloc_reducer.getLoc();

  ASSERT_FLOAT_EQ(this->getminormax(applygpu::minormax()), (double)minmaxloc_reducer.get());
  ASSERT_EQ(this->getlocX(applygpu::minormax()), RAJA::get<0>(raja_loc));
  ASSERT_EQ(this->getlocY(applygpu::minormax()), RAJA::get<1>(raja_loc));

  // SECOND TEST: equal values
  this->equalize();
  this->reducelocs();  // calculating new solution

  at_v<TypeParam, 0> minmaxloc_reducer2(applygpu::extremeval(), RAJA::make_tuple(0, 0));

  RAJA::kernel<ExecutionPol>(RAJA::make_tuple(colrange, rowrange),
                           [=] RAJA_DEVICE (int c, int r) {
                             applygpu::apply(minmaxloc_reducer2, d_dataview(r, c), RAJA::make_tuple(c, r));
                           });

  ASSERT_FLOAT_EQ(this->getminormax(applygpu::minormax()), (double)minmaxloc_reducer2.get());
  ASSERT_EQ(this->getlocX(applygpu::minormax()), RAJA::get<0>(minmaxloc_reducer2.getLoc()));
  ASSERT_EQ(this->getlocY(applygpu::minormax()), RAJA::get<1>(minmaxloc_reducer2.getLoc()));

  // THIRD TEST: match with RAJA CPU reduction
  at_v<TypeParam, 1> cpuloc_reducer(applycpu::extremeval(), 0);

  RAJA::RangeSegment wholerange(0, array_length);

  RAJA::forall<RAJA::seq_exec>(wholerange, [=] (int ii) {
                            applycpu::apply(cpuloc_reducer, this->data[ii], ii);
                          });
  
  ASSERT_FLOAT_EQ((double)cpuloc_reducer.get(), (double)minmaxloc_reducer2.get());
  ASSERT_EQ(cpuloc_reducer.getLoc(), RAJA::get<0>(minmaxloc_reducer2.getLoc()) + RAJA::get<1>(minmaxloc_reducer2.getLoc()) * ydim);
}

GPU_TYPED_TEST_P(HIPReduceLocTest, ReduceLoc2DIndexTupleViewKernelRandom)
{
  using applygpu = funcapplier<at_v<TypeParam, 0>>;
  using applycpu = funcapplier<at_v<TypeParam, 1>>;

  using ExecutionPol =
    RAJA::KernelPolicy<
      RAJA::statement::HipKernel<
      RAJA::statement::For<1, RAJA::hip_thread_x_loop,  // row
        RAJA::statement::For<0, RAJA::hip_thread_y_loop,  // col
          RAJA::statement::Lambda<0>
        >
      >
      > // end HipKernel
    >;

  RAJA::RangeSegment colrange(0, ydim);
  RAJA::RangeSegment rowrange(0, xdim);

  this->equalize();
  constexpr int reps = 25;

  for ( int jj = 0; jj < reps; ++jj )
  {
    this->randompoke( (int)indexdist(prng), (double)valuedist(prng) ); // perturb array

    // compare naive CPU reduceloc with GPU reduceloc
    this->reducelocs(); // recalculate solution

    at_v<TypeParam, 0> minmaxloc_reducer(applygpu::extremeval(), RAJA::make_tuple(0, 0));

    RAJA::View<double, RAJA::Layout<2>> d_dataview(this->d_data, xdim, ydim);
    RAJA::kernel<ExecutionPol>(RAJA::make_tuple(colrange, rowrange),
                             [=] RAJA_DEVICE (int c, int r) {
                               applygpu::apply(minmaxloc_reducer, d_dataview(r, c), RAJA::make_tuple(c, r));
                             });

    RAJA::tuple<int, int> raja_loc = minmaxloc_reducer.getLoc();

    ASSERT_FLOAT_EQ(this->getminormax(applygpu::minormax()), (double)minmaxloc_reducer.get());
    ASSERT_EQ(this->getlocX(applygpu::minormax()), RAJA::get<0>(raja_loc));
    ASSERT_EQ(this->getlocY(applygpu::minormax()), RAJA::get<1>(raja_loc));

    // compare RAJA CPU reduceloc with GPU reduceloc
    at_v<TypeParam, 1> cpuloc_reducer(applycpu::extremeval(), 0);

    RAJA::RangeSegment wholerange(0, array_length);

    RAJA::forall<RAJA::seq_exec>(wholerange, [=] (int ii) {
                              applycpu::apply(cpuloc_reducer, this->data[ii], ii);
                           });

    ASSERT_FLOAT_EQ((double)cpuloc_reducer.get(), (double)minmaxloc_reducer.get());
    ASSERT_EQ(cpuloc_reducer.getLoc(), RAJA::get<0>(raja_loc) + RAJA::get<1>(raja_loc) * ydim);
  }
}

REGISTER_TYPED_TEST_SUITE_P( HIPReduceLocTest,
                             ReduceLoc2DIndexTupleViewKernel,
                             ReduceLoc2DIndexTupleViewKernelRandom
                          );

using MinLocTypeTuple = ::testing::Types<
                          list<ReduceMinLoc<RAJA::hip_reduce, double, RAJA::tuple<int, int>>,
                               ReduceMinLoc<RAJA::seq_reduce, double, int>>
                        >;
INSTANTIATE_TYPED_TEST_SUITE_P(ReduceMin2DTuple, HIPReduceLocTest, MinLocTypeTuple);

using MaxLocTypeTuple = ::testing::Types<
                          list<ReduceMaxLoc<RAJA::hip_reduce, double, RAJA::tuple<int, int>>,
                               ReduceMaxLoc<RAJA::seq_reduce, double, int>>
                        >;
INSTANTIATE_TYPED_TEST_SUITE_P(ReduceMax2DTuple, HIPReduceLocTest, MaxLocTypeTuple);

