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

#ifdef RAJA_ENABLE_CUDA
enum ExecPlace { HOST, DEVICE, NUM_PLACES };
#else
enum ExecPlace { HOST, NUM_PLACES };
#endif

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

  template <typename... ARGS>
  RAJA_INLINE
  explicit Resources(ARGS const &... args)
  {
    camp::sink(apply(args)...);
  }

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

struct ResourceList
{
  Resources host_resources;
  Resources device_resources;
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



//template <typename RESOURCE_TUPLE, camp::idx_t I, camp::idx_t IMAX>
//struct LaunchPlaceExtractor {
//
//  template <typename BODY>
//  static void launch(ExecPlace place,
//                     RESOURCE_TUPLE const &resources,
//                     BODY const &body)
//  {
//
//    using resource_t = camp::at_v<typename RESOURCE_TUPLE::TList, I>;
//
//    if (place == resource_t::exec_place) {
//      auto const &resource = camp::get<I>(resources);
//
//      LaunchContext ctx(resource, place);
//
//      LaunchPlaceSwitchboard<resource_t>::exec(place, ctx, body);
//    } else {
//
//      LaunchPlaceExtractor<RESOURCE_TUPLE, I + 1, IMAX>::launch(place,
//                                                                resources,
//                                                                body);
//    }
//  }
//};
//
//
//template <typename RESOURCE_TUPLE, camp::idx_t IMAX>
//struct LaunchPlaceExtractor<RESOURCE_TUPLE, IMAX, IMAX> {
//  template <typename BODY>
//  static void launch(ExecPlace place,
//                     RESOURCE_TUPLE const &resources,
//                     BODY const &body)
//  {
//    printf("Failed to find resource requirements for execution place %d\n",
//           (int)place);
//  }
//};

template <typename POLICY_LIST, typename BODY>
void launch(ExecPlace place, ResourceList const &resources, BODY const &body)
{

  if(place == HOST){
    using launch_t = LaunchExecute<typename POLICY_LIST::host_policy_t>;

    launch_t::exec(LaunchContext(resources.host_resources, HOST), body);
  }
#ifdef RAJA_ENABLE_CUDA
  else if(place == DEVICE){
    using launch_t = LaunchExecute<typename POLICY_LIST::device_policy_t>;

    launch_t::exec(LaunchContext(resources.device_resources, DEVICE), body);
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
  static RAJA_INLINE RAJA_HOST_DEVICE void exec(LaunchContext const &ctx,
                                    SEGMENT const &segment,
                                    BODY const &body)
  {

    // block stride loop
    int len = segment.end() - segment.begin();
    for (int i = 0; i < len; i++) {

      body(*(segment.begin() + i));
    }
  }
};


#ifdef RAJA_ENABLE_CUDA

template <typename SEGMENT>
struct LoopExecute<cuda_thread_x_loop, SEGMENT> {

  template <typename BODY>
  static RAJA_INLINE RAJA_DEVICE void exec(LaunchContext const &ctx,
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
  static RAJA_INLINE RAJA_DEVICE void exec(LaunchContext const &ctx,
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
  static RAJA_INLINE RAJA_DEVICE void exec(LaunchContext const &ctx,
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
  static RAJA_INLINE RAJA_DEVICE void exec(LaunchContext const &ctx,
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
  static RAJA_INLINE RAJA_DEVICE void exec(LaunchContext const &ctx,
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
  static RAJA_INLINE RAJA_DEVICE void exec(LaunchContext const &ctx,
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
#endif

//template <typename POLICY_LIST, camp::idx_t IDX, camp::idx_t MAX_IDX>
//struct LoopPlaceSwitchboard {
//  template <typename SEGMENT, typename BODY>
//  static inline RAJA_HOST_DEVICE void exec(LaunchContext const &ctx,
//                                    SEGMENT const &segment,
//                                    BODY const &body)
//  {
//    if (camp::at_v<POLICY_LIST, IDX>::exec_place == ctx.exec_place) {
//      LoopExecute<typename camp::at_v<POLICY_LIST, IDX>::policy_t,
//                  SEGMENT>::exec(ctx, segment, body);
//    } else {
//      LoopPlaceSwitchboard<POLICY_LIST, IDX + 1, MAX_IDX>::exec(ctx,
//                                                                segment,
//                                                                body);
//    }
//  }
//};
//
//template <typename POLICY_LIST, camp::idx_t MAX_IDX>
//struct LoopPlaceSwitchboard<POLICY_LIST, MAX_IDX, MAX_IDX> {
//  template <typename SEGMENT, typename BODY>
//  static RAJA_HOST_DEVICE void exec(LaunchContext const &ctx,
//                                    SEGMENT const &segment,
//                                    BODY const &body)
//  {
//    printf("whoops!");
//  }
//};


template <typename POLICY_LIST, typename CONTEXT, typename SEGMENT, typename BODY>
RAJA_HOST_DEVICE RAJA_INLINE void loop(CONTEXT const &ctx,
                           SEGMENT const &segment,
                           BODY const &body)
{
#ifndef __CUDA_ARCH__
  LoopExecute<typename POLICY_LIST::host_policy_t,
              SEGMENT>::exec(ctx, segment, body);
#else
  LoopExecute<typename POLICY_LIST::device_policy_t,
              SEGMENT>::exec(ctx, segment, body);
#endif
}


}  // end namespace RAJA


#endif /* RAJA_pattern_kernel_HPP */
