//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

///
/// Source file containing tests for RAJA GPU tuple min-loc reductions
/// with random perturbations. Temporarily duplicates functionality of
/// test/unit/cuda/test-reduce-loc.cpp, but is more clear.
///

#include <iostream>
#include <random>

#include "RAJA/RAJA.hpp"
#include "RAJA_gtest.hpp"

using namespace RAJA;

static constexpr int block_size = 1024;

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
struct funcapplier<ReduceMinLoc<cuda_reduce, NumType, Indexer>>
{
  static NumType extremeval()
  {
    return (NumType)1024;
  }

  RAJA_HOST_DEVICE static void apply(ReduceMinLoc<cuda_reduce, NumType, Indexer> const & r,
                                     NumType const & val,
                                     Indexer ii)
  {
    r.minloc(val, ii);
  }

  // helps gtest cuda_typed_test_p determine minloc (min == false)
  static bool minormax()
  {
    return false;
  }
};

template <typename NumType, typename Indexer>
struct funcapplier<ReduceMinLoc<seq_reduce, NumType, Indexer>>
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
struct funcapplier<ReduceMaxLoc<cuda_reduce, NumType, Indexer>>
{
  static NumType extremeval()
  {
    return (NumType)(-1024);
  }

  RAJA_HOST_DEVICE static void apply(ReduceMaxLoc<cuda_reduce, NumType, Indexer> const & r,
                                     NumType const & val,
                                     Indexer ii)
  {
    r.maxloc(val, ii);
  }

  // helps gtest cuda_typed_test_p determine maxloc (max == true)
  static bool minormax()
  {
    return true;
  }
};

template <typename NumType, typename Indexer>
struct funcapplier<ReduceMaxLoc<seq_reduce, NumType, Indexer>>
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
struct CUDAReduceLocRandTest : public ::testing::Test
{
  public:
  virtual void SetUp()
  {
    cudaErrchk(cudaMallocManaged(&data, sizeof(int) * array_length, cudaMemAttachGlobal));

    // setting data values
    int count = 0;
    for ( int ii = 0; ii < array_length; ++ii ) {
      data[ii] = (RAJA::Real_type)(count++);
    }

    data[array_length-1] = -1.0;

    reducelocs( 0, array_length );
  }

  virtual void TearDown() {
    cudaErrchk(cudaFree(data));
  }

  // make all values equal
  void equalize()
  {
    for ( int ii = 0; ii < array_length; ++ii )
    {
      data[ii] = 5.0;
    }
  }

  // change a random value in data
  void randompoke( int index, int val )
  {
    data[index] = val;
  }

  // cpu reduction, for recalculating solutions
  void reducelocs( int start, int finish )
  {
    min = array_length * 2;
    max = 0.0;
    minloc = -1;
    maxloc = -1;

    for (int yy = 0; yy < ydim; ++yy) {
        RAJA::Real_type val = data[yy];

        if (val > max) {
          max = val;
          maxloc = yy;
        }

        if (val < min) {
          min = val;
          minloc = yy;
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

  RAJA::Real_type getloc( bool actual )
  {
    if ( !actual )
    {
      return minloc;
    }
    else
    {
      return maxloc;
    }
  }

  int * data;

  int max;
  int min;
  int maxloc;
  int minloc;
};

TYPED_TEST_SUITE_P(CUDAReduceLocRandTest);

// Tests CUDA reduce loc on array over one range.
// Each iteration introduces a random value into the array.
GPU_TYPED_TEST_P(CUDAReduceLocRandTest, ReduceLocRandom)
{
  using applygpu = funcapplier<at_v<TypeParam, 0>>;
  using applycpu = funcapplier<at_v<TypeParam, 1>>;

  int * data2 = this->data;

  this->equalize();
  constexpr int reps = 25;

  for ( int jj = 0; jj < reps; ++jj )
  {
    this->randompoke( (int)indexdist(prng), (int)valuedist(prng) ); // perturb array
    // RAJA CPU reduce loc
    at_v<TypeParam, 1> cpuloc_reducer(applycpu::extremeval(), 0);
    RAJA::forall<seq_exec>(RAJA::RangeSegment(0, array_length),
                            [=] (int ii) {
                              applycpu::apply(cpuloc_reducer, this->data[ii], ii);
                            });

    // RAJA GPU reduce loc
    at_v<TypeParam, 0> minmaxloc_reducer(applygpu::extremeval(), 0);

    RAJA::forall<cuda_exec<block_size>>(RAJA::RangeSegment(0, array_length),
                             [=] RAJA_DEVICE (int ii) {
                               applygpu::apply(minmaxloc_reducer, data2[ii], ii);
                             });

    int raja_loc = minmaxloc_reducer.getLoc();

    ASSERT_EQ(cpuloc_reducer.get(), minmaxloc_reducer.get());
    ASSERT_EQ(cpuloc_reducer.getLoc(), raja_loc);
  }
}

// Tests CUDA reduce loc on array with all same values, over segments.
// CUDA finds location in the last segment, 
// while CPU seq_reduce finds location in first segment.
GPU_TYPED_TEST_P(CUDAReduceLocRandTest, ReduceLocSameHalves)
{
  using applygpu = funcapplier<at_v<TypeParam, 0>>;
  using applycpu = funcapplier<at_v<TypeParam, 1>>;

  RAJA::RangeSegment colrange0(0, array_length/2);
  RAJA::RangeSegment colrange1(array_length/2, array_length);

  RAJA::TypedIndexSet<RAJA::RangeSegment> cset;

  cset.push_back( colrange0 );
  cset.push_back( colrange1 );

  int * data2 = this->data;

  this->equalize();

  // CPU reduce loc
  at_v<TypeParam, 1> cpureduce(applygpu::extremeval(), 0);
  RAJA::forall<ExecPolicy<seq_segit, seq_exec>>(cset, [=] (int ii) {
                            applycpu::apply(cpureduce, this->data[ii], ii);
                          });

  // GPU reduce loc
  at_v<TypeParam, 0> minmaxloc_reducer(applygpu::extremeval(), 0);

  RAJA::forall<ExecPolicy<seq_segit, cuda_exec<block_size>>>(
    cset, [=] RAJA_DEVICE (int ii) {
      applygpu::apply(minmaxloc_reducer, data2[ii], ii);
    });

  int raja_loc = minmaxloc_reducer.getLoc();

  ASSERT_EQ((int)cpureduce.get(), (int)minmaxloc_reducer.get());

  // CUDA reduce loc over a TypedIndexSet finds the loc of the 
  // last Segment. CPU reduce loc treats TypedIndexSet as one Segment.
  ASSERT_EQ(cpureduce.getLoc(), 0);
  ASSERT_EQ(array_length / 2, raja_loc);
}

// Tests CUDA reduce loc on array with unique values, over segments.
GPU_TYPED_TEST_P(CUDAReduceLocRandTest, ReduceLocAscendingHalves)
{
  using applygpu = funcapplier<at_v<TypeParam, 0>>;
  using applycpu = funcapplier<at_v<TypeParam, 1>>;

  RAJA::RangeSegment colrange0(0, array_length/2);
  RAJA::RangeSegment colrange1(array_length/2, array_length);

  RAJA::TypedIndexSet<RAJA::RangeSegment> cset;

  cset.push_back( colrange0 );
  cset.push_back( colrange1 );

  int * data2 = this->data;

  // create ascending array
  for ( int zz = 0; zz < array_length; ++zz )
  {
    this->data[zz] = zz;
  }

  // CPU reduce loc
  at_v<TypeParam, 1> cpureduce(applycpu::extremeval(), 0);
  RAJA::forall<ExecPolicy<seq_segit, seq_exec>>(cset, [=] (int ii) {
                            applycpu::apply(cpureduce, this->data[ii], ii);
                          });

  // GPU reduce loc
  at_v<TypeParam, 0> minmaxloc_reducer(applygpu::extremeval(), 0);

  RAJA::forall<ExecPolicy<seq_segit, cuda_exec<block_size>>>(
    cset, [=] RAJA_DEVICE (int ii) {
      applygpu::apply(minmaxloc_reducer, data2[ii], ii);
    });

  int raja_loc = minmaxloc_reducer.getLoc();

  ASSERT_EQ(cpureduce.get(), minmaxloc_reducer.get());

  ASSERT_EQ(cpureduce.getLoc(), raja_loc);
}

// Tests CUDA reduce loc on two segment halves of array.
// Each test iteration introduces a random value within the segments.
// Compare scaled CUDA reduce loc vs. un-scaled CUDA reduce loc.
GPU_TYPED_TEST_P(CUDAReduceLocRandTest, ReduceLocRandomHalves)
{
  using applygpu = funcapplier<at_v<TypeParam, 0>>;

  RAJA::RangeSegment colrange0(0, array_length/2);
  RAJA::RangeSegment colrange1(array_length/2, array_length);

  RAJA::TypedIndexSet<RAJA::RangeSegment> cset;

  cset.push_back( colrange0 );
  cset.push_back( colrange1 );

  int * data2 = this->data;

  this->equalize();
  constexpr int reps = 25;

  for ( int jj = 0; jj < reps; ++jj )
  {
    int index = (int)indexdist(prng);
    int value = (int)valuedist(prng);
    this->randompoke( index, value ); // perturb array

    // scaled GPU reduce loc
    at_v<TypeParam, 0> gpureducescaled(applygpu::extremeval(), 0);
    RAJA::forall<ExecPolicy<seq_segit, cuda_exec<block_size>>>(
      cset, [=] RAJA_DEVICE (int ii) {
        applygpu::apply(gpureducescaled, 2*(data2[ii]), ii);
      });

    int scaled_loc = gpureducescaled.getLoc();

    // normal GPU reduce loc
    at_v<TypeParam, 0> minmaxloc_reducer(applygpu::extremeval(), 0);

    RAJA::forall<ExecPolicy<seq_segit, cuda_exec<block_size>>>(
      cset, [=] RAJA_DEVICE (int ii) {
        applygpu::apply(minmaxloc_reducer, data2[ii], ii);
      });

    int raja_loc = minmaxloc_reducer.getLoc();

    ASSERT_EQ(gpureducescaled.get(), 2*minmaxloc_reducer.get());
    ASSERT_EQ(scaled_loc, raja_loc);
  }
}

// Tests whether CUDA reduce loc works over non-block-sized boundaries.
// Segments being reduced are non-contiguous.
// Each test iteration introduces a random value within the segments.
// Compare scaled CUDA reduce loc vs. un-scaled CUDA reduce loc.
GPU_TYPED_TEST_P(CUDAReduceLocRandTest, ReduceLocRandomDisjoint)
{
  using applygpu = funcapplier<at_v<TypeParam, 0>>;

  RAJA::RangeSegment colrange0(1, 230);
  RAJA::RangeSegment colrange1(237, 385);
  RAJA::RangeSegment colrange2(410, 687);
  RAJA::RangeSegment colrange3(857, 999);

  RAJA::TypedIndexSet<RAJA::RangeSegment> cset;

  cset.push_back( colrange0 );
  cset.push_back( colrange1 );
  cset.push_back( colrange2 );
  cset.push_back( colrange3 );

  int * data2 = this->data;

  this->equalize();
  constexpr int reps = 25;

  for ( int jj = 0; jj < reps; ++jj )
  {
    // choose index
    int index = 101;
    if ( jj % 2 == 0 ) index = 299;
    if ( jj % 3 == 0 ) index = 511;
    if ( jj % 4 == 0 ) index = 913;
    this->randompoke( index, (int)valuedist(prng) ); // perturb array

    // scaled GPU reduce loc
    at_v<TypeParam, 0> gpureducescaled(applygpu::extremeval(), 0);
    RAJA::forall<ExecPolicy<seq_segit, cuda_exec<block_size>>>(
      cset, [=] RAJA_DEVICE (int ii) {
        applygpu::apply(gpureducescaled, 2*(data2[ii]), ii);
      });

    int scaled_loc = gpureducescaled.getLoc();

    // normal GPU reduce loc
    at_v<TypeParam, 0> minmaxloc_reducer(applygpu::extremeval(), 0);

    RAJA::forall<ExecPolicy<seq_segit, cuda_exec<block_size>>>(
      cset, [=] RAJA_DEVICE (int ii) {
        applygpu::apply(minmaxloc_reducer, data2[ii], ii);
      });

    int raja_loc = minmaxloc_reducer.getLoc();

    ASSERT_EQ(gpureducescaled.get(), 2*minmaxloc_reducer.get());
    ASSERT_EQ(scaled_loc, raja_loc);
  }
}

REGISTER_TYPED_TEST_SUITE_P( CUDAReduceLocRandTest,
                             ReduceLocRandom,
                             ReduceLocSameHalves,
                             ReduceLocAscendingHalves,
                             ReduceLocRandomHalves,
                             ReduceLocRandomDisjoint
                          );

using MinLocType = ::testing::Types<
                     list<ReduceMinLoc<RAJA::cuda_reduce, int, int>,
                          ReduceMinLoc<RAJA::seq_reduce, int, int>>
                   >;
INSTANTIATE_TYPED_TEST_SUITE_P(ReduceMin, CUDAReduceLocRandTest, MinLocType);

using MaxLocType = ::testing::Types<
                     list<ReduceMaxLoc<RAJA::cuda_reduce, int, int>,
                          ReduceMaxLoc<RAJA::seq_reduce, int, int>>
                   >;
INSTANTIATE_TYPED_TEST_SUITE_P(ReduceMax, CUDAReduceLocRandTest, MaxLocType);
