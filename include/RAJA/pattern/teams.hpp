/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   RAJA header file containing user interface for RAJA::kernel
 *
 ******************************************************************************
 */

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef RAJA_pattern_teams_HPP
#define RAJA_pattern_teams_HPP

#include "RAJA/config.hpp"

#include "RAJA/internal/get_platform.hpp"
#include "RAJA/util/plugins.hpp"

#include "camp/camp.hpp"
#include "camp/concepts.hpp"
#include "camp/tuple.hpp"

#include "RAJA/util/macros.hpp"
#include "RAJA/util/types.hpp"

#include "RAJA/policy/loop/policy.hpp"
#include "RAJA/policy/openmp/policy.hpp"
#include "RAJA/policy/cuda/policy.hpp"


#if defined(__CUDA_ARCH__)
#define TEAM_SHARED __shared__
#define TEAM_SYNC() __syncthreads()
#else
#define TEAM_SHARED
#define TEAM_SYNC()
#endif


namespace RAJA
{

//GPU or CPU threads available
#if defined(RAJA_ENABLE_CUDA) && defined(RAJA_ENABLE_OPENMP)
enum ExecPlace { HOST, HOST_THREADS, DEVICE, NUM_PLACES };
#elif defined(RAJA_ENABLE_CUDA)
enum ExecPlace { HOST, DEVICE, NUM_PLACES };
#else
enum ExecPlace { HOST, NUM_PLACES };
#endif

//GPU or CPU threads available
#if defined(RAJA_ENABLE_CUDA) && defined(RAJA_ENABLE_OPENMP)
template <typename HOST_POLICY, typename HOST_THREADS_POLICY, typename DEVICE_POLICY>
struct LoopPolicy {
    using host_policy_t = HOST_POLICY;
    using host_threads_policy_t  = HOST_THREADS_POLICY;
    using device_policy_t = DEVICE_POLICY;
};

template <typename HOST_POLICY, typename HOST_THREADS_POLICY, typename DEVICE_POLICY>
struct LaunchPolicy {
    using host_policy_t = HOST_POLICY;
    using host_threads_policy_t  = HOST_THREADS_POLICY;
    using device_policy_t = DEVICE_POLICY;
};
#elif defined(RAJA_ENABLE_CUDA)
template <typename HOST_POLICY, typename DEVICE_POLICY>
struct LoopPolicy {
    using host_policy_t = HOST_POLICY;
    using device_policy_t = DEVICE_POLICY;
};

template <typename HOST_POLICY, typename DEVICE_POLICY>
struct LaunchPolicy {
    using host_policy_t = HOST_POLICY;
    using device_policy_t = DEVICE_POLICY;
};
#else
template <typename HOST_POLICY>
struct LoopPolicy {
    using host_policy_t = HOST_POLICY;
};

template <typename HOST_POLICY>
struct LaunchPolicy {
    using host_policy_t = HOST_POLICY;
};
#endif


struct Teams {
  int value[3];

  RAJA_INLINE
  RAJA_HOST_DEVICE
  constexpr
  Teams() : value{1, 1, 1} {}

  RAJA_INLINE
  RAJA_HOST_DEVICE
  constexpr
  Teams(int i) : value{i, 1, 1} {}

  RAJA_INLINE
  RAJA_HOST_DEVICE
  constexpr
  Teams(int i, int j) : value{i, j, 1} {}

  RAJA_INLINE
  RAJA_HOST_DEVICE
  constexpr
  Teams(int i, int j, int k) : value{i, j, k} {}
};

struct Threads {
  int value[3];

  RAJA_INLINE
  RAJA_HOST_DEVICE
  constexpr
  Threads() : value{1, 1, 1} {}


  RAJA_INLINE
  RAJA_HOST_DEVICE
  constexpr
  Threads(int i) : value{i, 1, 1} {}

  RAJA_INLINE
  RAJA_HOST_DEVICE
  constexpr
  Threads(int i, int j) : value{i, j, 1} {}

  RAJA_INLINE
  RAJA_HOST_DEVICE
  constexpr
  Threads(int i, int j, int k) : value{i, j, k} {}
};

struct Lanes {
  int value;

  RAJA_INLINE
  RAJA_HOST_DEVICE
  constexpr
  Lanes() : value(0) {}

  RAJA_INLINE
  RAJA_HOST_DEVICE
  constexpr
  Lanes(int i) : value(i) {}
};



struct Resources
{
public:
  Teams teams;
  Threads threads;
  Lanes lanes;

  RAJA_INLINE
  Resources() = default;

  Resources(Teams in_teams, Threads in_threads)
    : teams(in_teams), threads(in_threads) {};

  /*
  template <typename... ARGS>
  RAJA_INLINE
  explicit Resources(ARGS const &... args)
  {
    camp::sink(apply(args)...);
  }
  */
private:
  RAJA_HOST_DEVICE
  RAJA_INLINE
  Teams apply(Teams const &a) { return (teams = a); }

  RAJA_HOST_DEVICE
  RAJA_INLINE
  Threads apply(Threads const &a) { return (threads = a); }

  RAJA_HOST_DEVICE
  RAJA_INLINE
  Lanes apply(Lanes const &a) { return (lanes = a); }
};

//Consolidate to one set of resources?
struct ResourceList
{
  Resources team_resources;
  //Resources host_resources;
#if defined(RAJA_ENABLE_CUDA)
  //Resources device_resources;
#endif
};

class LaunchContext : public Resources
{
public:
  ExecPlace exec_place;

  LaunchContext(Resources const &base, ExecPlace place)
      : Resources(base), exec_place(place)
  {
  }
};


struct seq_launch_t{};

struct omp_launch_t{};

template<bool async, int num_threads=0>
struct cuda_launch_t{};


template <typename LAUNCH_POLICY>
struct LaunchExecute;

template <>
struct LaunchExecute<RAJA::seq_launch_t> {
  template <typename BODY>
  static void exec(LaunchContext const &ctx, BODY const &body)
  {
    body(ctx);
  }
};

//Perhaps just leave this as host?
template <>
struct LaunchExecute<RAJA::omp_launch_t> {
  template <typename BODY>
  static void exec(LaunchContext const &ctx, BODY const &body)
  {
    body(ctx);
  }
};

#ifdef RAJA_ENABLE_CUDA
template <typename BODY>
__global__
void launch_global_fcn(LaunchContext ctx, BODY body)
{
  // printf("Entering global function\n");
  body(ctx);
  // printf("Leaving global function\n");
}


template <bool async>
struct LaunchExecute<RAJA::cuda_launch_t<async,0>> {
  template <typename BODY>
  static void exec(LaunchContext const &ctx, BODY const &body)
  {
    dim3 blocks;
    dim3 threads;

    blocks.x = ctx.teams.value[0];
    blocks.y = ctx.teams.value[1];
    blocks.z = ctx.teams.value[2];

    threads.x = ctx.threads.value[0];
    threads.y = ctx.threads.value[1];
    threads.z = ctx.threads.value[2];
    printf("thread block dim %d %d %d \n", threads.x, threads.y, threads.z);
    printf("grid block dim %d %d %d \n", blocks.x, blocks.y, blocks.z);
    launch_global_fcn<<<blocks, threads>>>(ctx, body);

    if(!async){
      cudaDeviceSynchronize();
    }
  }
};


template <typename BODY, int num_threads>
__launch_bounds__(num_threads, 1)
__global__
void launch_global_fcn_fixed(LaunchContext ctx, BODY body)
{
  // printf("Entering global function\n");
  body(ctx);
  // printf("Leaving global function\n");
}


template <bool async, int nthreads>
struct LaunchExecute<RAJA::cuda_launch_t<async,nthreads>> {
  template <typename BODY>
  static void exec(LaunchContext const &ctx, BODY const &body)
  {
    dim3 blocks;
    dim3 threads;

    blocks.x = ctx.teams.value[0];
    blocks.y = ctx.teams.value[1];
    blocks.z = ctx.teams.value[2];

    threads.x = ctx.threads.value[0];
    threads.y = ctx.threads.value[1];
    threads.z = ctx.threads.value[2];
    launch_global_fcn_fixed<nthreads> <<<blocks, threads>>>(ctx, body);

    if(!async){
      cudaDeviceSynchronize();
    }
  }
};


#endif


template <typename POLICY_LIST, typename BODY>
void launch(ExecPlace place, ResourceList const &resources, BODY const &body)
{

  if(place == HOST){
    using launch_t = LaunchExecute<typename POLICY_LIST::host_policy_t>;

    launch_t::exec(LaunchContext(resources.team_resources, HOST), body);
  }
#ifdef RAJA_ENABLE_OPENMP
  else if(place == HOST_THREADS)
  {
    printf("Launching OMP code ! \n");
    using launch_t = LaunchExecute<typename POLICY_LIST::host_threads_policy_t>;
    launch_t::exec(LaunchContext(resources.team_resources, HOST_THREADS), body);
  }
#endif
#ifdef RAJA_ENABLE_CUDA
  else if(place == DEVICE){
    using launch_t = LaunchExecute<typename POLICY_LIST::device_policy_t>;

    launch_t::exec(LaunchContext(resources.team_resources, DEVICE), body);
  }
#endif
  else {
    throw "unknown launch place!";
  }
}


template <typename POLICY, typename SEGMENT>
struct LoopExecute;

template <typename SEGMENT>
struct LoopExecute<loop_exec, SEGMENT> {

  template <typename BODY>
  static RAJA_INLINE RAJA_HOST_DEVICE void exec(LaunchContext const RAJA_UNUSED_ARG(&ctx),
                                    SEGMENT const &segment,
                                    BODY const &body)
  {

    // block stride loop
    const int len = segment.end() - segment.begin();
    for (int i = 0; i < len; i++) {

      body(*(segment.begin() + i));
    }
  }

  template <typename BODY>
  static RAJA_INLINE RAJA_HOST_DEVICE void exec(LaunchContext const RAJA_UNUSED_ARG(&ctx),
                                    SEGMENT const &segment0,
                                    SEGMENT const &segment1,
                                    BODY const &body)
  {

    // block stride loop
    const int len1 = segment1.end() - segment1.begin();
    const int len0 = segment0.end() - segment0.begin();

    for (int j = 0; j < len1; j++) {
      for (int i = 0; i < len0; i++) {

        body(*(segment0.begin() + i),*(segment1.begin() + j));
      }
    }
  }

  template <typename BODY>
  static RAJA_INLINE RAJA_HOST_DEVICE void exec(LaunchContext const RAJA_UNUSED_ARG(&ctx),
                                    SEGMENT const &segment0,
                                    SEGMENT const &segment1,
                                    SEGMENT const &segment2,
                                    BODY const &body)
  {

    // block stride loop
    const int len2 = segment2.end() - segment2.begin();
    const int len1 = segment1.end() - segment1.begin();
    const int len0 = segment0.end() - segment0.begin();

    for (int k = 0; k < len2; k++) {
      for (int j = 0; j < len1; j++) {
        for (int i = 0; i < len0; i++) {
          body(*(segment0.begin() + i),*(segment1.begin() + j), *(segment2.begin() + k));
        }
      }
    }
  }

};

#if defined(RAJA_ENABLE_OPENMP)
template <typename SEGMENT>
struct LoopExecute<omp_parallel_for_exec, SEGMENT> {

  template <typename BODY>
  static RAJA_INLINE RAJA_HOST_DEVICE void exec(LaunchContext const RAJA_UNUSED_ARG(&ctx),
                                    SEGMENT const &segment,
                                    BODY const &body)
  {

    int len = segment.end() - segment.begin();
#pragma omp parallel for
    for (int i = 0; i < len; i++) {

      body(*(segment.begin() + i));
    }
  }

  template <typename BODY>
  static RAJA_INLINE RAJA_HOST_DEVICE void exec(LaunchContext const RAJA_UNUSED_ARG(&ctx),
                                    SEGMENT const &segment0,
                                    SEGMENT const &segment1,
                                    BODY const &body)
  {

    // block stride loop
    const int len1 = segment1.end() - segment1.begin();
    const int len0 = segment0.end() - segment0.begin();

#pragma omp parallel for collapse (2)
    for (int j = 0; j < len1; j++) {
      for (int i = 0; i < len0; i++) {

        body(*(segment0.begin() + i),*(segment1.begin() + j));
      }
    }
  }

  template <typename BODY>
  static RAJA_INLINE RAJA_HOST_DEVICE void exec(LaunchContext const RAJA_UNUSED_ARG(&ctx),
                                    SEGMENT const &segment0,
                                    SEGMENT const &segment1,
                                    SEGMENT const &segment2,
                                    BODY const &body)
  {

    // block stride loop
    const int len2 = segment2.end() - segment2.begin();
    const int len1 = segment1.end() - segment1.begin();
    const int len0 = segment0.end() - segment0.begin();

#pragma omp parallel for collapse (3)
    for (int k = 0; k < len2; k++) {
      for (int j = 0; j < len1; j++) {
        for (int i = 0; i < len0; i++) {
          body(*(segment0.begin() + i),*(segment1.begin() + j), *(segment2.begin() + k));
        }
      }
    }
  }

};
#endif

#ifdef RAJA_ENABLE_CUDA

template <typename SEGMENT>
struct LoopExecute<cuda_thread_x_loop, SEGMENT> {

  template <typename BODY>
  static RAJA_INLINE RAJA_DEVICE void exec(LaunchContext const RAJA_UNUSED_ARG(&ctx),
                               SEGMENT const &segment,
                               BODY const &body)
  {

    int len = segment.end() - segment.begin();

    for (int i = threadIdx.x; i < len; i += blockDim.x) {
      body(*(segment.begin() + i));
    }
  }
};

template <typename SEGMENT>
struct LoopExecute<cuda_thread_y_loop, SEGMENT> {

  template <typename BODY>
  static RAJA_INLINE RAJA_DEVICE void exec(LaunchContext const RAJA_UNUSED_ARG(&ctx),
                               SEGMENT const &segment,
                               BODY const &body)
  {

    int len = segment.end() - segment.begin();

    for (int i = threadIdx.y; i < len; i += blockDim.y) {
      body(*(segment.begin() + i));
    }
  }
};

template <typename SEGMENT>
struct LoopExecute<cuda_thread_z_loop, SEGMENT> {

  template <typename BODY>
  static RAJA_INLINE RAJA_DEVICE void exec(LaunchContext const RAJA_UNUSED_ARG(&ctx),
                               SEGMENT const &segment,
                               BODY const &body)
  {

    int len = segment.end() - segment.begin();

    for (int i = threadIdx.z; i < len; i += blockDim.z) {
      body(*(segment.begin() + i));
    }
  }
};

template <typename SEGMENT>
struct LoopExecute<cuda_block_x_loop, SEGMENT> {

  template <typename BODY>
  static RAJA_INLINE RAJA_DEVICE void exec(LaunchContext const RAJA_UNUSED_ARG(&ctx),
                               SEGMENT const &segment,
                               BODY const &body)
  {

    int len = segment.end() - segment.begin();

    for (int i = blockIdx.x; i < len; i += gridDim.x) {
      body(*(segment.begin() + i));
    }
  }
};

template <typename SEGMENT>
struct LoopExecute<cuda_block_x_direct, SEGMENT> {

  template <typename BODY>
  static RAJA_INLINE RAJA_DEVICE void exec(LaunchContext const RAJA_UNUSED_ARG(&ctx),
                               SEGMENT const &segment,
                               BODY const &body)
  {

    int len = segment.end() - segment.begin();
    {
      const int i = blockIdx.x;
      body(*(segment.begin() + i));
    }
  }
};

template <typename SEGMENT>
struct LoopExecute<cuda_block_y_direct, SEGMENT> {

  template <typename BODY>
  static RAJA_INLINE RAJA_DEVICE void exec(LaunchContext const RAJA_UNUSED_ARG(&ctx),
                               SEGMENT const &segment,
                               BODY const &body)
  {

    int len = segment.end() - segment.begin();
    {
      const int i = blockIdx.y;
      body(*(segment.begin() + i));
    }
  }
};

//collapsed cuda policies

template <typename SEGMENT>
struct LoopExecute<cuda_block_xyz_direct<2>, SEGMENT> {

  template <typename BODY>
  static RAJA_INLINE RAJA_DEVICE void exec(LaunchContext const RAJA_UNUSED_ARG(&ctx),
                               SEGMENT const &segment0,
                               SEGMENT const &segment1,
                               BODY const &body)
  {
    int len1 = segment1.end() - segment1.begin();
    int len0 = segment0.end() - segment0.begin();
    {
      const int i = blockIdx.x; const int j = blockIdx.y;
      body(*(segment0.begin() + i), *(segment1.begin() + j));
    }
  }
};

template <typename SEGMENT>
struct LoopExecute<cuda_block_xyz_direct<3>, SEGMENT> {

  template <typename BODY>
  static RAJA_INLINE RAJA_DEVICE void exec(LaunchContext const RAJA_UNUSED_ARG(&ctx),
                               SEGMENT const &segment0,
                               SEGMENT const &segment1,
                               SEGMENT const &segment2,
                               BODY const &body)
  {
    int len2 = segment2.end() - segment2.begin();
    int len1 = segment1.end() - segment1.begin();
    int len0 = segment0.end() - segment0.begin();
    {
      const int i = blockIdx.x; 
      const int j = blockIdx.y;
      const int k = blockIdx.z;
      body(*(segment0.begin() + i), 
           *(segment1.begin() + j),
           *(segment2.begin() + k));
    }
  }
};


#endif


template <typename POLICY_LIST, typename CONTEXT, typename SEGMENT, typename BODY>
RAJA_HOST_DEVICE RAJA_INLINE void loop(CONTEXT const &ctx,
                           SEGMENT const &segment,
                           BODY const &body)
{
#ifdef __CUDA_ARCH__
  LoopExecute<typename POLICY_LIST::device_policy_t,
              SEGMENT>::exec(ctx, segment, body);
#else
  switch (ctx.exec_place)
  {
#if defined(RAJA_ENABLE_OPENMP)
     case HOST_THREADS: LoopExecute<typename POLICY_LIST::host_threads_policy_t,
                           SEGMENT>::exec(ctx, segment, body); break;
#endif

     case HOST: LoopExecute<typename POLICY_LIST::host_policy_t,
                            SEGMENT>::exec(ctx, segment, body); break;

     default: RAJA_ABORT_OR_THROW("Back end not support \n"); break;
  }
#endif
}

template <typename POLICY_LIST, typename CONTEXT, typename SEGMENT, typename BODY>
RAJA_HOST_DEVICE RAJA_INLINE void loop(CONTEXT const &ctx,
                           SEGMENT const &segment0,
                           SEGMENT const &segment1,
                           BODY const &body)
{
#ifdef __CUDA_ARCH__
  LoopExecute<typename POLICY_LIST::device_policy_t,
              SEGMENT>::exec(ctx, segment0, segment1, body);
#else
  switch (ctx.exec_place)
  {
#if defined(RAJA_ENABLE_OPENMP)
     case HOST_THREADS: LoopExecute<typename POLICY_LIST::host_threads_policy_t,
                                    SEGMENT>::exec(ctx, segment0, segment1, body); break;
#endif

     case HOST: LoopExecute<typename POLICY_LIST::host_policy_t,
                            SEGMENT>::exec(ctx, segment0, segment1, body); break;
  }
#endif
}

template <typename POLICY_LIST, typename CONTEXT, typename SEGMENT, typename BODY>
RAJA_HOST_DEVICE RAJA_INLINE void loop(CONTEXT const &ctx,
                           SEGMENT const &segment0,
                           SEGMENT const &segment1,
                           SEGMENT const &segment2,
                           BODY const &body)
{

#ifdef __CUDA_ARCH__
  LoopExecute<typename POLICY_LIST::device_policy_t,
              SEGMENT>::exec(ctx, segment0, segment1, segment2, body);
#else
  switch (ctx.exec_place)
  {

#ifdef RAJA_ENABLE_OPENMP
     case HOST_THREADS: LoopExecute<typename POLICY_LIST::host_threads_policy_t,
                                    SEGMENT>::exec(ctx, segment0, segment1, segment2, body); break;
#endif

     case HOST: LoopExecute<typename POLICY_LIST::host_policy_t,
                            SEGMENT>::exec(ctx, segment0, segment1, segment2, body); break;
  }
#endif

}



}  // end namespace RAJA


#endif /* RAJA_pattern_kernel_HPP */
