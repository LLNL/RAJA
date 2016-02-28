/*
 * Copyright (c) 2016, Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 *
 * All rights reserved.
 *
 * This source code cannot be distributed without permission and
 * further review from Lawrence Livermore National Laboratory.
 */

/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Header file defining RAJA loop execution policies.
 * 
 *          Note: availability of some policies depends on compiler choice.
 *
 ******************************************************************************
 */

#ifndef RAJA_execpolicy_HXX
#define RAJA_execpolicy_HXX


#include "config.hxx"


namespace RAJA {


#if defined(RAJA_COMPILER_ICC)

//
// Segment execution policies
//
struct seq_exec {};
struct simd_exec {};
struct omp_parallel_for_exec {};
//struct omp_for_nowait_exec {};

//
// Index set segment iteration policies
// 
struct seq_segit {};
struct omp_parallel_for_segit {};
struct omp_parallel_segit {};
struct omp_taskgraph_segit {};
struct omp_taskgraph_interval_segit {};

//
// Reduction policies
//
struct seq_reduce {};
struct omp_reduce {};

#endif   // end  Intel compilers.....


#if defined(RAJA_COMPILER_GNU) 

//
// Segment execution policies
//
struct seq_exec {};
struct simd_exec {};
struct omp_parallel_for_exec {};
//struct omp_for_nowait_exec {};

//
// Index set segment iteration policies
//
struct seq_segit {};
struct omp_parallel_for_segit {};
struct omp_parallel_segit {};
struct omp_taskgraph_segit {};
struct omp_taskgraph_interval_segit {};

//
// Reduction policies
//
struct seq_reduce {};
struct omp_reduce {};

#endif   // end  GNU compilers.....


#if defined(RAJA_COMPILER_XLC12)

//
// Segment execution policies
//
struct seq_exec {};
struct simd_exec {};
struct omp_parallel_for_exec {};
//struct omp_for_nowait_exec {};

//
// Index set segment iteration policies
//
struct seq_segit {};
struct omp_parallel_for_segit {};
struct omp_parallel_segit {};
struct omp_taskgraph_segit {};
struct omp_taskgraph_interval_segit {};

//
// Reduction policies
//
struct seq_reduce {};
struct omp_reduce {};

#endif   // end  xlc v12 compiler on bgq


#if defined(RAJA_COMPILER_CLANG)

//
// Segment exec policies
//
struct seq_exec {};
struct simd_exec {};
struct omp_parallel_for_exec {};

//
// Index set segment iteration policies
// 
struct seq_segit {};
struct omp_parallel_for_segit {};
struct omp_parallel_segit {};
struct omp_taskgraph_segit {};
struct omp_taskgraph_interval_segit {};

//
// Reduction policies
//
struct seq_reduce {};
struct omp_reduce {};

#endif   // end  CLANG compilers.....


#if defined(RAJA_USE_CUDA)

//
// Segment exec policies
//
template <size_t BLOCK_SIZE>
struct cuda_exec {};

template <size_t BLOCK_SIZE>
struct cuda_exec_async {};

//
// Reduction policies
//
template <size_t BLOCK_SIZE>
struct cuda_reduce {};


#define RAJA_HOST_DEVICE __host__ __device__
#define RAJA_DEVICE __device__

#else 

#define RAJA_HOST_DEVICE
#define RAJA_DEVICE

#endif

}  // closing brace for RAJA namespace


#endif  // closing endif for header file include guard
