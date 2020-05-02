//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef __TEST_FORALL_ATOMIC_UTILS_HPP__
#define __TEST_FORALL_ATOMIC_UTILS_HPP__

#include "RAJA/RAJA.hpp"

#include "test-forall-utils.hpp"

#include <numeric>

using SequentialForallAtomicExecPols =
  camp::list<
              RAJA::seq_exec,
              RAJA::loop_exec
              //RAJA::simd_exec not expected to work with atomics
            >;

using SequentialAtomicPols =
  camp::list<
#if defined(RAJA_TEST_EXHAUSTIVE)
              RAJA::auto_atomic,
              RAJA::builtin_atomic,
#endif
              RAJA::seq_atomic
            >;

#if defined(RAJA_ENABLE_OPENMP)
using OpenMPForallAtomicExecPols =
  camp::list<
#if defined(RAJA_TEST_EXHAUSTIVE)
              RAJA::omp_for_nowait_exec,
              RAJA::omp_parallel_for_exec,
#endif
              RAJA::omp_for_exec
              //RAJA::omp_parallel_exec<RAJA::seq_exec>
              //can work with atomics but tests not suited to omp parallel region
            >;

using OpenMPAtomicPols =
  camp::list<
#if defined(RAJA_TEST_EXHAUSTIVE)
              RAJA::omp_atomic,
              RAJA::builtin_atomic,
#endif
              RAJA::auto_atomic
            >;
#endif  // RAJA_ENABLE_OPENMP

#if defined(RAJA_ENABLE_CUDA)
using CudaAtomicPols =
  camp::list<
#if defined(RAJA_TEST_EXHAUSTIVE)
              RAJA::auto_atomic,
#endif
              RAJA::cuda_atomic
            >;
#endif  // RAJA_ENABLE_CUDA

#if defined(RAJA_ENABLE_HIP)
using HipAtomicPols =
  camp::list<
#if defined(RAJA_TEST_EXHAUSTIVE)
               RAJA::auto_atomic,
#endif
               RAJA::hip_atomic
            >;
#endif  // RAJA_ENABLE_HIP

//
// Atomic data types
//
using AtomicDataTypeList =
  camp::list<
              RAJA::Index_type,
              int,
#if defined(RAJA_TEST_EXHAUSTIVE)
              unsigned,
              long long,
              unsigned long long,
              float,
#endif
              double
           >;


// range segment multiplexer
template< typename Index, typename SegType >
struct RSMultiplexer {};

template< typename Index >
struct RSMultiplexer < Index, RAJA::TypedRangeSegment<Index> >
{
  RAJA::TypedRangeSegment<Index>
  makeseg( Index N )
  {
    return RAJA::TypedRangeSegment<Index>( 0, N );
  }
};

template< typename Index >
struct RSMultiplexer < Index, RAJA::TypedRangeStrideSegment<Index> >
{
  RAJA::TypedRangeStrideSegment<Index>
  makeseg( Index N )
  {
    return RAJA::TypedRangeStrideSegment<Index>( 0, N, 1 );
  }
};

template< typename Index >
struct RSMultiplexer < Index, RAJA::TypedListSegment<Index> >
{
  RAJA::TypedListSegment<Index>
  makeseg( Index N )
  {
    std::vector<Index> temp(N);
    std::iota( std::begin(temp), std::end(temp), 0 );
    return RAJA::TypedListSegment<Index>( &temp[0], static_cast<size_t>(temp.size()) );
  }
};

template< typename Index >
struct RSMultiplexer < Index, RAJA::TypedIndexSet<RAJA::TypedListSegment<Index>, RAJA::TypedRangeSegment<Index>, RAJA::TypedRangeStrideSegment<Index>> >
{
  RAJA::TypedIndexSet<RAJA::TypedListSegment<Index>, RAJA::TypedRangeSegment<Index>, RAJA::TypedRangeStrideSegment<Index>>
  makeseg( Index N )
  {
    RAJA::Index_type chunk = N/3;
    RAJA::TypedIndexSet<RAJA::TypedListSegment<Index>, RAJA::TypedRangeSegment<Index>, RAJA::TypedRangeStrideSegment<Index>> Iset;

    // create ListSegment for first 1/3rd
    std::vector<Index> temp(chunk);
    std::iota( std::begin(temp), std::begin(temp) + chunk, 0 );
    Iset.push_back( RAJA::TypedListSegment<Index>( &temp[0], static_cast<size_t>(temp.size()) ) );

    // create RangeSegment for second 1/3rd
    Iset.push_back( RAJA::TypedRangeSegment<Index>( chunk, 2*chunk ) );

    // create RangeStrideSegment for last 1/3rd
    Iset.push_back( RAJA::TypedRangeStrideSegment<Index>( 2*chunk, N, 1 ) );

    return Iset;
  }
};

using AtomicSegmentList = 
  camp::list<
              RAJA::TypedRangeSegment<RAJA::Index_type>,
              RAJA::TypedRangeStrideSegment<RAJA::Index_type>,
              RAJA::TypedListSegment<RAJA::Index_type>
              //RAJA::TypedIndexSet<RAJA::TypedListSegment<RAJA::Index_type>, RAJA::TypedRangeSegment<RAJA::Index_type>, RAJA::TypedRangeStrideSegment<RAJA::Index_type>>
            >;

#endif  // __TEST_FORALL_ATOMIC_UTILS_HPP__
